import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch3d.ops import knn_points, knn_gather
import math
import warnings

from .conv_uno import UNO, UNOEncoder
from .sparse_conv_block import SparseConvResBlock
from .sparse_conv_block import convert_to_backend_form, convert_to_backend_form_like, \
    calculate_norm, get_features_from_backend_form, get_normalising_conv

class SparseUNet(nn.Module):
    def __init__(self, channels=3, nf=64, time_emb_dim=256, img_size=128, num_conv_blocks=3, knn_neighbours=3, uno_res=64, 
                 uno_mults=(1,2,4,8), z_dim=None, out_channels=None, conv_type="conv", 
                 depthwise_sparse=True, kernel_size=7, backend="torch_dense", optimise_dense=True,
                 blocks_per_level=(2,2,2,2), attn_res=[16,8], dropout_res=16, dropout=0.1,
                 uno_base_nf=64):
        super().__init__()
        self.backend = backend
        self.img_size = img_size
        self.uno_res = uno_res
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size
        self.optimise_dense = optimise_dense
        # Input projection
        self.linear_in = nn.Linear(channels, nf)
        # Output projection
        self.linear_out = nn.Linear(nf, out_channels if out_channels is not None else channels)

        # Diffusion time MLP
        # TODO: Better to have more features here? 64 by default isn't many
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        uno_coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=uno_res) for _ in range(2)]))
        uno_coords = rearrange(uno_coords, 'c h w -> () (h w) c')
        self.register_buffer("uno_coords", uno_coords) 

        self.normalising_conv = get_normalising_conv(kernel_size=kernel_size, backend=backend)

        self.down_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, time_emb_dim=time_emb_dim, z_dim=z_dim, depthwise=depthwise_sparse, backend=backend
            ))
        self.uno_linear_in = nn.Linear(nf, uno_base_nf)

        self.uno_linear_out = nn.Linear(uno_base_nf, nf)
        self.up_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.up_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, skip_dim=nf, time_emb_dim=time_emb_dim, z_dim=z_dim, depthwise=depthwise_sparse, backend=backend)
            )

        self.uno = UNO(uno_base_nf, uno_base_nf, width=uno_base_nf, mults=uno_mults, blocks_per_level=blocks_per_level, 
                       time_emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, res=uno_res,
                       attn_res=attn_res, dropout_res=dropout_res, dropout=dropout)
    
    def knn_interpolate_to_grid(self, x, coords):
        with torch.no_grad():
            _, assign_index, neighbour_coords = knn_points(self.uno_coords.repeat(x.size(0),1,1), coords, K=self.knn_neighbours, return_nn=True)
            # neighbour_coords: (B, y_length, K, 2)
            diff = neighbour_coords - self.uno_coords.unsqueeze(2) # can probably use dist from knn_points
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16) # (B, y_length, K, 1)

        # See Eqn. 2 in PointNet++. Inverse square distance weighted mean
        neighbours = knn_gather(x, assign_index) # (B, y_length, K, C) 
        out = (neighbours * weights).sum(2) / weights.sum(2)

        return out.to(x.dtype)

    def get_torch_norm_kernel_size(self, img_size, round_down=True):
        if img_size != self.img_size:
            ratio = img_size / self.img_size
            # new kernel_size becomes:
            # 1 -> 1, 1.5 -> 1, 2 -> 1 or 3, 2.5 -> 3, 3 -> 3, 3.5 -> 3, 4 -> 3 or 5, 4.5 -> 5, ...
            # where there are multiple options this is determined by round_down
            new_kernel_size = self.kernel_size * ratio
            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1
            return max(new_kernel_size, 3)
        else:
            return self.kernel_size
    
    def dense_forward(self, x, t, z=None):
        # If x is image shaped (4D) then treat it as a dense tensor for better optimisation
        height = x.size(2)

        coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=height) for _ in range(2)])).to(x.device)
        coords = rearrange(coords, 'c h w -> () (h w) c')
        coords = repeat(coords, "() ... -> b ...", b=x.size(0))

        x = F.conv2d(x, self.linear_in.weight[:,:,None,None], bias=self.linear_in.bias)
        t = self.time_mlp(t)

        # NOTE: Still need to norm to avoid edge artefacts
        mask = torch.ones(x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        kernel_size = self.get_torch_norm_kernel_size(height)
        weight = torch.ones(1, 1, kernel_size, kernel_size, dtype=x.dtype, device=x.device) / (self.kernel_size ** 2)
        norm = F.conv2d(mask, weight, padding=kernel_size//2)

        # 1. Down conv blocks
        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)
        
        # 2. Interpolate to regular grid
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res)

        # 3. UNO
        x = self.uno(x, t, z=z)

        # 4. Interpolate back to sparse coordinates
        x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear')
        x = rearrange(x, "b c (h w) () -> b c h w", h=height)
        x = F.conv2d(x, self.uno_linear_out.weight[:,:,None,None], bias=self.uno_linear_out.bias)

        # 5. Up conv blocks
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)
        
        x = F.conv2d(x, self.linear_out.weight[:,:,None,None], bias=self.linear_out.bias)
    
        return x
    
    def forward(self, x, t, z=None, sample_lst=None, coords=None):
        batch_size = x.size(0)

        # If x is image shaped (4D) then treat it as a dense tensor for better optimisation
        if len(x.shape) == 4 and self.optimise_dense:
            if sample_lst is not None:
                warnings.warn("Ignoring sample_lst: Recieved 4D x and sample_list != None so treating x as a dense Image.")
            if coords is not None:
                warnings.warn("Ignoring coords: Recieved 4D x and coords != None so treating x as a dense Image.")
            return self.dense_forward(x, t, z=z)
        
        # TODO: Re-add the parts of this needed if x is image shape but optimise_dense is False
        # i.e. rearrange and set sample_lst.
      
        assert sample_lst is not None, "In sparse mode sample_lst must be provided"
        if coords is None:
            coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=self.img_size) for _ in range(2)])).to(x.device)
            coords = rearrange(coords, 'c h w -> () (h w) c')
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            coords = torch.gather(coords, 1, sample_lst.unsqueeze(2).repeat(1,1,coords.size(2))).contiguous()

        x = self.linear_in(x)
        t = self.time_mlp(t)

        # 1. Down conv blocks
        # Cache mask and norms
        x = convert_to_backend_form(x, sample_lst, self.img_size, backend=self.backend)
        backend_tensor = x
        norm = calculate_norm(self.normalising_conv, backend_tensor, sample_lst, self.img_size, batch_size, backend=self.backend)

        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)

        # 2. Interpolate to regular grid
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res)

        # 3. UNO
        x = self.uno(x, t, z=z)

        # 4. Interpolate back to sparse coordinates
        x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear')
        x = rearrange(x, "b c l () -> b l c")
        x = self.uno_linear_out(x)
        x = convert_to_backend_form_like(x, backend_tensor, sample_lst=sample_lst, img_size=self.img_size, backend=self.backend)

        # 5. Up conv blocks
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)
    
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend) 
        x = self.linear_out(x)

        return x

class SparseEncoder(nn.Module):
    def __init__(self, out_channels, channels=3, nf=64, img_size=128, num_conv_blocks=3, knn_neighbours=3, uno_res=64, 
                 uno_mults=(1,2,4,8), z_dim=None, conv_type="conv", 
                 depthwise_sparse=True, kernel_size=7, backend="torch_dense", optimise_dense=True,
                 blocks_per_level=(2,2,2,2), attn_res=[16,8], dropout_res=16, dropout=0.1,
                 uno_base_nf=64, stochastic=False):
        super().__init__()
        self.backend = backend
        self.img_size = img_size
        self.uno_res = uno_res
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size
        self.optimise_dense = optimise_dense
        self.stochastic = stochastic
        # Input projection
        self.linear_in = nn.Linear(channels, nf)
        # Output projection
        self.linear_out = nn.Linear(out_channels, out_channels)

        uno_coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=uno_res) for _ in range(2)]))
        uno_coords = rearrange(uno_coords, 'c h w -> () (h w) c')
        self.register_buffer("uno_coords", uno_coords) 

        self.normalising_conv = get_normalising_conv(kernel_size=kernel_size, backend=backend)

        self.down_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, time_emb_dim=nf, z_dim=z_dim, depthwise=depthwise_sparse, backend=backend
            ))
        self.uno_linear_in = nn.Linear(nf, uno_base_nf)

        self.uno = UNOEncoder(uno_base_nf, out_channels, width=uno_base_nf, mults=uno_mults, blocks_per_level=blocks_per_level, 
                       time_emb_dim=nf, z_dim=z_dim, conv_type=conv_type, res=uno_res,
                       attn_res=attn_res, dropout_res=dropout_res, dropout=dropout)
    
        if stochastic:
            self.mu = nn.Linear(out_channels, out_channels)
            self.logvar = nn.Linear(out_channels, out_channels)
        
    
    def knn_interpolate_to_grid(self, x, coords):
        with torch.no_grad():
            _, assign_index, neighbour_coords = knn_points(self.uno_coords.repeat(x.size(0),1,1), coords, K=self.knn_neighbours, return_nn=True)
            # neighbour_coords: (B, y_length, K, 2)
            diff = neighbour_coords - self.uno_coords.unsqueeze(2) # can probably use dist from knn_points
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16) # (B, y_length, K, 1)

        # See Eqn. 2 in PointNet++. Inverse square distance weighted mean
        neighbours = knn_gather(x, assign_index) # (B, y_length, K, C) 
        out = (neighbours * weights).sum(2) / weights.sum(2)

        return out
    
    def forward(self, x, sample_lst=None, coords=None):
        batch_size = x.size(0)
        if len(x.shape) == 4:
            x = rearrange(x, 'b c h w -> b (h w) c')
                
        if coords is None:
            coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, steps=self.img_size, device=x.device) for _ in range(2)]))
            coords = rearrange(coords, 'c h w -> () (h w) c')
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            if sample_lst is not None:
                coords = torch.gather(coords, 1, sample_lst.unsqueeze(2).repeat(1,1,coords.size(2))).contiguous()
        
        if sample_lst is None:
            sample_lst = torch.arange(self.img_size**2, device=x.device)
            sample_lst = repeat(sample_lst, 's -> b s', b=x.size(0))  

        x = self.linear_in(x)
        
        # 1. Down conv blocks
        # Cache mask and norms
        x = convert_to_backend_form(x, sample_lst, self.img_size, backend=self.backend)
        backend_tensor = x
        norm = calculate_norm(self.normalising_conv, backend_tensor, sample_lst, self.img_size, batch_size, backend=self.backend)

        downs = []
        for block in self.down_blocks:
            x = block(x, norm=norm)
            downs.append(x)

        # 2. Interpolate to regular grid
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res)

        # 3. UNO
        x = self.uno(x)
        x = x.mean(dim=(2,3))
    
        x = self.linear_out(x)

        if self.stochastic:
            mu = self.mu(x)
            logvar = self.logvar(x)
            x = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            return x, mu, logvar

        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
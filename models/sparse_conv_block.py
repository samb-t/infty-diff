# import sys
# sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from einops import rearrange, repeat
import math

# spconv
try:
    import spconv as spconv_core
    spconv_core.constants.SPCONV_ALLOW_TF32 = True
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
    SPCONV_AVAILABLE = True
except Exception:
    SPCONV_AVAILABLE = False

# torchsparse
try:
    import torchsparse
    from torchsparse import nn as spnn
    from torchsparse import SparseTensor
    TORCHSPARSE_AVAILABLE = True
except Exception:
    TORCHSPARSE_AVAILABLE = False

# minkowski
try:
    import MinkowskiEngine as ME
    MINKOWSKI_AVAILABLE = True
except Exception:
    MINKOWSKI_AVAILABLE = False


"""
Wrapper around the different backends
"""
# TODO: At test time a dense conv can be used instead
class SparseConvResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None, 
                 epsilon=1e-5, z_dim=None, depthwise=True, backend="torch_dense"):
        super().__init__()
        self.backend = backend
        if self.backend == "spconv":
            assert SPCONV_AVAILABLE, "spconv backend is not detected."
            block = SPConvResBlock
        elif self.backend == "torchsparse":
            assert TORCHSPARSE_AVAILABLE, "torchsparse backend is not detected."
            block = TorchsparseResBlock
        elif self.backend == "minkowski":
            assert MINKOWSKI_AVAILABLE, "Minkowski Engine backend is not detected."
            block =  MinkowskiConvResBlock
        elif self.backend == "torch_dense":
            block = TorchDenseConvResBlock
        else:
            raise Exception("Unrecognised backend.")
        
        self.block = block(img_size, embed_dim, kernel_size=kernel_size, mult=mult, skip_dim=skip_dim, 
                 time_emb_dim=time_emb_dim, epsilon=epsilon, z_dim=z_dim, depthwise=depthwise)
    
    def get_normalising_conv(self):
        return self.block.get_normalising_conv()

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        if isinstance(x, torch.Tensor) and len(x.shape) == 4 and self.backend != "torch_dense":
            # If image shape passed in then use more efficient dense convolution
            return self.block.dense_forward(x, t=t, skip=skip, z=z, norm=norm)
        elif isinstance(x, torch.Tensor) and len(x.shape) == 4 and self.backend == "torch_dense" and not isinstance(norm, tuple):
            # if backend is torch_dense and we input the norm is not a tuple then we can run in dense mode.
            return self.block.dense_forward(x, t=t, skip=skip, z=z, norm=norm)
        else:
            return self.block(x, t=t, skip=skip, z=z, norm=norm)

class SPConvResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None, 
                 epsilon=1e-5, z_dim=None, depthwise=True):
        super().__init__()
        assert not depthwise, "spconv currently does not support depthwise convolutions."
        self.img_size = img_size
        self.spatial_size = img_size ** 2
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = nn.Linear(embed_dim + skip_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv = spconv.SubMConv2d(embed_dim, embed_dim, kernel_size=kernel_size, bias=False, padding=kernel_size//2)
    
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = spconv.SparseSequential(
            nn.Linear(embed_dim, embed_dim*mult),
            nn.GELU(),
            nn.Linear(embed_dim*mult, embed_dim)
        )

        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = None, None, None, None
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
    
    def get_torch_kernel(self, img_size, round_down=True):
        if img_size != self.img_size:
            ratio = img_size / self.img_size
            new_kernel_size = self.kernel_size * ratio
            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1
            new_kernel_size = max(new_kernel_size, 3)
            kernel = self.conv.weight.permute(0,3,1,2)
            kernel = F.interpolate(kernel, size=new_kernel_size, mode="bilinear")
            return kernel
        else:
            return self.conv.weight.permute(0,3,1,2)
    
    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor), "Dense forward expects x to be a torch Tensor"
        assert len(x.shape) == 4, "Dense forward expects x to be 4D: (b, c, h, w)"

        # Skip connection
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        h = rearrange(x, "b c h w -> (b h w) c")
        if skip is not None:
            skip = rearrange(skip, "b c h w -> (b h w) c")
            h = h.cat((h, skip), dim=-1)
            h = self.skip_linear(h)
        x = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)
        
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)
        
        # Conv and norm
        kernel = self.get_torch_kernel(height)
        h = F.conv2d(h, kernel, padding=kernel.size(-1)//2, groups=self.groups)
        h = h / norm
        x = x + h 

        # elementwise MLP
        h = rearrange(x, "b c h w -> (b h w) c")
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        for layer in self.mlp:
            h = layer(h)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        x = x + h

        return x
    
    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        if isinstance(h, spconv.SparseConvTensor):
            feats = h.features
        else:
            feats = h
        feats = norm(feats)
        q_sample = feats.size(0) // t.size(0)
        if t is not None:
            t = t_mlp(t)
            t = repeat(t, "b c -> (b l) c", l=q_sample)
            t_scale, t_shift = t.chunk(2, dim=-1)
            feats = feats * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)
            z_scale = repeat(z_scale, "b c -> (b l) c", l=q_sample)
            feats = feats * (1 + z_scale)
        if isinstance(h, spconv.SparseConvTensor):
            h = h.replace_feature(feats)
        else:
            h = feats
        return h

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, spconv.SparseConvTensor)

        # Skip connection
        if skip is not None:
            feats = torch.cat((x.features, skip.features), dim=-1)
            feats = self.skip_linear(feats)
            x = x.replace_feature(feats)
        
        h = x
        if t is not None or z is not None:
            h = self.modulate(x, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        
        h = self.conv(h)
        h = spconv_div(h, norm)
        x = spconv_add(x, h)

        if t is not None or z is not None:
            h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = spconv_add(x, self.mlp(h))

        return x

class TorchsparseResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None, 
                 epsilon=1e-5, z_dim=None, depthwise=True):
        super().__init__()
        self.img_size = img_size
        self.spatial_size = img_size ** 2
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.embed_dim = embed_dim
        self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = nn.Linear(embed_dim + skip_dim, embed_dim)

        # TODO: check where is best to have the 1 dimension.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv = spnn.Conv3d(embed_dim, embed_dim, kernel_size=(1,kernel_size,kernel_size), depthwise=depthwise, bias=False)
        self._custom_kaiming_uniform_(self.conv.kernel, a=math.sqrt(5))
    
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mult),
            nn.GELU(),
            nn.Linear(embed_dim*mult, embed_dim)
        )

        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = None, None, None, None
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
    
    def get_torch_kernel(self, img_size, round_down=True):
        if img_size != self.img_size:
            ratio = img_size / self.img_size
            new_kernel_size = self.kernel_size * ratio
            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1
            new_kernel_size = max(new_kernel_size, 3)
            kernel = rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)
            kernel = F.interpolate(kernel, size=new_kernel_size, mode="bilinear")
            return kernel
        else:
            return rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)

    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor), "Dense forward expects x to be a torch Tensor"
        assert len(x.shape) == 4, "Dense forward expects x to be 4D: (b, c, h, w)"

        # Skip connection
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        h = rearrange(x, "b c h w -> (b h w) c")
        if skip is not None:
            skip = rearrange(skip, "b c h w -> (b h w) c")
            h = torch.cat((h, skip), dim=-1)
            h = self.skip_linear(h)
        x = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)
        
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)
        
        # Conv and norm
        kernel = self.get_torch_kernel(height)
        h = F.conv2d(h, kernel, padding=kernel.size(-1)//2, groups=self.groups)
        h = h / norm
        x = x + h 

        # elementwise MLP
        h = rearrange(x, "b c h w -> (b h w) c")
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        h = self.mlp(h)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        x = x + h

        return x
    
    def _custom_kaiming_uniform_(self, tensor, a=0, nonlinearity='leaky_relu'):
        fan = self.embed_dim * (self.kernel_size ** 2)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(
            3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)
    
    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        if isinstance(h, torchsparse.SparseTensor):
            feats = h.feats
        else:
            feats = h
        feats = norm(feats)
        q_sample = feats.size(0) // t.size(0)
        if t is not None:
            t = t_mlp(t)
            t = repeat(t, "b c -> (b l) c", l=q_sample)
            t_scale, t_shift = t.chunk(2, dim=-1)
            feats = feats * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)
            z_scale = repeat(z_scale, "b c -> (b l) c", l=q_sample)
            feats = feats * (1 + z_scale)
        if isinstance(h, torchsparse.SparseTensor):
            h = convert_to_backend_form_like(feats, h, backend="torchsparse", rearrange_x=False)
        else:
            h = feats
        return h

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torchsparse.SparseTensor)

        # Skip connection
        if skip is not None:
            feats = torch.cat((x.feats, skip.feats), dim=-1)
            feats = self.skip_linear(feats)
            x = convert_to_backend_form_like(feats, x, backend="torchsparse", rearrange_x=False)
        
        h = x
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        
        h = self.conv(h)
        h = ts_div(h, norm)
        x = ts_add(x, h)

        if t is not None or z is not None:
            h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = ts_add(x, self.mlp(h.feats))

        return x

class TorchDenseConvResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None, 
                 epsilon=1e-5, z_dim=None, depthwise=True):
        super().__init__()
        self.img_size = img_size
        self.spatial_size = img_size ** 2
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = nn.Conv2d(embed_dim + skip_dim, embed_dim, 1)

        # TODO: Try using bias
        self.norm1 = ImageLayerNorm(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1, padding='same', groups=self.groups, bias=False)

        self.norm2 = ImageLayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*mult, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim*mult, embed_dim, 1)
        )
        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = None, None, None, None
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        
    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
            x = self.skip_linear(x)

        h = self.modulate(x, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        
        # Conv and norm
        h = self.conv(h) 
        h = h / norm
        x = x + h

        h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = x + self.mlp(h)

        return x 
    
    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        h = norm(h)
        if t is not None:
            t = t_mlp(t)[:,:,None,None]
            t_scale, t_shift = t.chunk(2, dim=1)
            h = h * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)[:,:,None,None]
            h = h * (1 + z_scale)
        return h
    
    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor)

        # For dense conv we also pass in a mask to make the output 'sparse' again
        norm, mask = norm

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
            x = self.skip_linear(x)
            x = x * mask
        
        h = self.modulate(x, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        h =  h * mask
        
        # Assume that x already has 0s for missing pixels so no masking here
        h = self.conv(h) 
        # NOTE: although norm is dense and not masked, everything afterwards is elementwise so this is fine
        h = h / norm 
        x = x + h

        h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = x + self.mlp(h)

        # mask out
        return x * mask

class MinkowskiConvResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None, 
                 epsilon=1e-5, z_dim=None, depthwise=True):
        super().__init__()
        raise NotImplementedError("Latest Minkowski Block not implemented yet.")
        self.img_size = img_size
        self.spatial_size = img_size ** 2
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = ME.MinkowskiLinear(embed_dim+skip_dim, embed_dim)

        if depthwise:
            self.conv = ME.MinkowskiChannelwiseConvolution(embed_dim, kernel_size=kernel_size, bias=False, dimension=2)
        else:
            self.conv = ME.MinkowskiConvolution(embed_dim, embed_dim, kernel_size=kernel_size, bias=False, dimension=2)
    
        self.mlp = nn.Sequential(
            MinkowskiLayerNorm(embed_dim),
            ME.MinkowskiLinear(embed_dim, embed_dim*mult),
            ME.MinkowskiGELU(),
            ME.MinkowskiLinear(embed_dim*mult, embed_dim)
        )

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim)
            )
        if z_dim is not None:
            self.z_mlp = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
    
    def get_kernel_for_torch(self):
        raise NotImplementedError

    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        raise NotImplementedError
    
    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, ME.SparseTensor) or isinstance(x, ME.TensorField)

        if skip is not None:
            x = ME.cat(x, skip)
            x = self.skip_linear(x)

        conditioning = 0
        if t is not None:
            conditioning += self.time_mlp(t)
        if z is not None:
            conditioning += self.z_mlp(z)

        # x is flat i.e. (b l) c so we need to repeat the conditioning in this way
        if t is not None or z is not None:
            q_sample = x.features.size(0) // t.size(0)
            conditioning = repeat(conditioning, "b c -> (b l) c", l=q_sample)
            # instead make function make_sparse_like(tensor=conditioning, sparse=x)
            conditioning = ME.SparseTensor(
                features=conditioning,
                minkowski_algorithm=x.coordinate_manager.minkowski_algorithm,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager
            )
        
        out = self.conv(x) / norm

        x = x + out
        if t is not None or z is not None:
            x = x + conditioning
        x = x + self.mlp(x)

        return x


##############################
###### HELPER FUNCTIONS ######
##############################

# TODO: This and `convert_to_backend_form_like` could be merged into a single funciton
def convert_to_backend_form(x, sample_lst, img_size, backend="torch_dense"):
    if backend == "spconv":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size)
        x = SparseConvTensor(
                    features=rearrange(x, "b l c -> (b l) c"),
                    indices=sparse_indices,
                    spatial_shape=(img_size, img_size), 
                    batch_size=x.size(0),
                )
    elif backend == "torchsparse":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size, ndims=3)
        x = torchsparse.SparseTensor(
            coords=sparse_indices,
            feats=rearrange(x, "b l c -> (b l) c")
        )
    elif backend == "minkowski":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size)
        x = ME.SparseTensor(
                features=rearrange(x, "b l c -> (b l) c"),
                coordinates=sparse_indices,
                # TODO: allow this to be changed externally
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, # or MEMORY_OPTIMIZED
            )
    elif backend == "torch_dense":
        x_full = torch.zeros(x.size(0), img_size**2, x.size(2), device=x.device, dtype=x.dtype)
        x_full.scatter_(1, repeat(sample_lst, "b l -> b l c", c=x.size(-1)), x)
        x = rearrange(x_full, "b (h w) c -> b c h w", h=img_size, w=img_size)
    else:
        raise Exception("Unrecognised backend.")

    return x

def convert_to_backend_form_like(x, backend_tensor, sample_lst=None, img_size=None, backend="torch_dense", rearrange_x=True):
    if backend == "spconv":
        assert img_size is not None
        x = SparseConvTensor(
                    features=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
                    indices=backend_tensor.indices,
                    spatial_shape=(img_size, img_size), 
                    batch_size=x.size(0),
                )
    elif backend == "torchsparse":
        x = torchsparse.SparseTensor(
            coords=backend_tensor.coords,
            feats=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
            stride=backend_tensor.stride
        )
        x.cmaps = backend_tensor.cmaps
        x.kmaps = backend_tensor.kmaps
    elif backend == "minkowski":
        x = ME.SparseTensor(
                features=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
                # TODO: allow this to be changed externally
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, # or MEMORY_OPTIMIZED
                coordinate_map_key=backend_tensor.coordinate_map_key,
                coordinate_manager=backend_tensor.coordinate_manager
            )
    elif backend == "torch_dense":
        # Don't bother using the backend_tensor, just use sample_lst
        assert img_size is not None
        assert sample_lst is not None
        x_full = torch.zeros(x.size(0), img_size**2, x.size(2), device=x.device, dtype=x.dtype)
        x_full.scatter_(1, repeat(sample_lst, "b l -> b l c", c=x.size(-1)), x)
        x = rearrange(x_full, "b (h w) c -> b c h w", h=img_size, w=img_size)
    else:
        raise Exception("Unrecognised backend.")

    return x

def get_features_from_backend_form(x, sample_lst, backend="torch_dense"):
    if backend == "spconv":
        return rearrange(x.features, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "torchsparse":
        return rearrange(x.feats, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "minkowski":
        return rearrange(x.features, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "torch_dense":
        x = rearrange(x, "b c h w -> b (h w) c")
        x = torch.gather(x, 1, sample_lst.unsqueeze(2).repeat(1,1,x.size(2)))
        return x
    else:
        raise Exception("Unrecognised backend.")

def calculate_norm(conv, backend_tensor, sample_lst, img_size, batch_size, backend="torch_dense"):
    if backend == "spconv":
        device, dtype = backend_tensor.features.device, backend_tensor.features.dtype
        ones = torch.ones(backend_tensor.features.size(0), 1, device=device, dtype=dtype)
        mask = SparseConvTensor(
                features=ones,
                indices=backend_tensor.indices,
                spatial_shape=(img_size, img_size), 
                batch_size=batch_size,
            )
        norm = conv(mask) 
    elif backend == "torchsparse":
        device, dtype = backend_tensor.feats.device, backend_tensor.feats.dtype
        ones = torch.ones(backend_tensor.feats.size(0), 1, device=device, dtype=dtype)
        mask = torchsparse.SparseTensor(
                coords=backend_tensor.coords,
                feats=ones,
                stride=backend_tensor.stride
            )
        mask.cmaps = backend_tensor.cmaps
        mask.kmaps = backend_tensor.kmaps
        norm = conv(mask) 
    elif backend == "minkowski":
        device, dtype = backend_tensor.features.device, backend_tensor.features.dtype
        ones = torch.ones(backend_tensor.features.size(0), 1, device=device, dtype=dtype)
        mask = ME.SparseTensor(
                features=ones,
                # TODO: allow this to be changed externally
                minkowski_algorithm=backend_tensor.coordinate_manager.minkowski_algorithm, # or MEMORY_OPTIMIZED
                coordinate_map_key=backend_tensor.coordinate_map_key,
                coordinate_manager=backend_tensor.coordinate_manager
            )
        norm = conv(mask)
    elif backend == "torch_dense":
        device, dtype = backend_tensor.device, backend_tensor.dtype
        mask = torch.zeros(sample_lst.size(0), img_size**2, device=device, dtype=dtype)
        mask.scatter_(1, sample_lst, torch.ones(sample_lst.size(0), sample_lst.size(1), device=sample_lst.device, dtype=dtype))
        mask = rearrange(mask, "b (h w) -> b () h w", h=img_size, w=img_size)
        norm = conv(mask)
        norm[norm < 1e-5] = 1.0
        norm = (norm, mask)
    else:
        raise Exception("Unrecognised backend.")
    
    return norm

# TODO: This can be clearned up a bit
def get_normalising_conv(kernel_size, backend="torch_dense"):
    if backend == "spconv":
        assert SPCONV_AVAILABLE, "spconv backend is not detected."
        weight = torch.ones(1, kernel_size, kernel_size, 1) / (kernel_size ** 2)
        conv = spconv.SubMConv2d(1, 1, kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        conv.weight.data = weight
        conv.weight.requires_grad_(False)
    elif backend == "torchsparse":
        assert TORCHSPARSE_AVAILABLE, "spconv backend is not detected."
        weight = torch.ones(kernel_size**2, 1, 1) / (kernel_size ** 2)
        conv = spnn.Conv3d(1, 1, kernel_size=(1,kernel_size,kernel_size), bias=False)
        conv.kernel.data = weight
        conv.kernel.requires_grad_(False)
    elif backend == "minkowski":
        assert MINKOWSKI_AVAILABLE, "Minkowski Engine backend is not detected."
        weight = torch.ones(kernel_size**2, 1, 1) / (kernel_size ** 2)
        conv = ME.MinkowskiConvolution(1, 1, kernel_size=kernel_size, bias=False, dimension=2)
        conv.kernel.data = weight
        conv.kernel.requires_grad_(False)
    elif backend == "torch_dense":
        weight = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        conv.weight.data = weight
        conv.weight.requires_grad_(False)
        return conv
    else:
        raise Exception("Unrecognised backend.")

    return conv

"""
sample_lst is a tensor of shape (B, L)
which can be used to index flattened 2D images.
This functions converts it to a tensor of shape (BxL, 3)
    indices[:,0] is the number of the item in the batch
    indices[:,1] is the number of the item in the y direction
    indices[:,2] is the number of the item in the x direction
"""
# TODO: Any chance to get better performance by sorting in a specific way?
def sample_lst_to_sparse_indices(sample_lst, img_size, ndims=2, dtype=torch.int32):
    # number of the item in the batch - (B,)
    batch_idx = torch.arange(sample_lst.size(0), device=sample_lst.device, dtype=torch.int32)
    batch_idx = repeat(batch_idx, "b -> b l", l=sample_lst.size(1))
    # pixel number in vertical direction - (B,L)
    sample_lst_h = sample_lst.div(img_size, rounding_mode='trunc').to(dtype)
    # pixel number in horizontal direction - (B,L)
    sample_lst_w =  (sample_lst % img_size).to(dtype)

    if ndims == 2:
        indices = torch.stack([batch_idx, sample_lst_h, sample_lst_w], dim=2)
        indices = rearrange(indices, "b l three -> (b l) three")
    else:
        zeros = torch.zeros_like(sample_lst_h)
        indices = torch.stack([zeros, sample_lst_h, sample_lst_w, batch_idx], dim=2)
        indices = rearrange(indices, "b l four -> (b l) four")

    return indices

def ts_add(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats + b.feats
    else:
        feats = a.feats + b
    out = SparseTensor(
        coords=a.coords,
        feats=feats,
        stride=a.stride
    )
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out

def ts_div(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats / b.feats
    else:
        feats = a.feats / b
    out = SparseTensor(
        coords=a.coords,
        feats=feats,
        stride=a.stride
    )
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out

def spconv_add(a, b):
    if isinstance(b, SparseConvTensor):
        return a.replace_feature(a.features + b.features)
    else:
        return a.replace_feature(a.features + b)

def spconv_div(a, b):
    if isinstance(b, SparseConvTensor):
        return a.replace_feature(a.features / b.features)
    else:
        return a.replace_feature(a.features / b)

def spconv_clamp(a, min=None, max=None):
    return a.replace_feature(a.features.clamp(min=min, max=max))

class MinkowskiLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, input):
        output = self.norm(input.F)
        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

def minkowski_clamp(x, min=None, max=None):
    output = x.features.clamp(min=min, max=max)
    if isinstance(x, ME.TensorField):
        return ME.TensorField(
            output,
            coordinate_field_map_key=x.coordinate_field_map_key,
            coordinate_manager=x.coordinate_manager,
            quantization_mode=x.quantization_mode,
        )
    else:
        return ME.SparseTensor(
            output,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

class ImageLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


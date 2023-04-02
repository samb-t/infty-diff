import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np
from flash_attn.flash_attention import FlashAttention

class UNO(nn.Module):
    def __init__(self, nin, nout, width=64, mults=(1,2,4,8), blocks_per_level=(2,2,2,2), time_emb_dim=None, 
                 z_dim=None, conv_type="conv", res=64, attn_res=[16,8], dropout_res=16, dropout=0.1):
        super().__init__()
        self.width = width
        self.conv_type = conv_type
        self.fc0 = nn.Conv2d(nin+2 if conv_type == "spectral" else nin, width, 1)

        dims = [width, *map(lambda m: width * m, mults)]
        in_out = list(zip(dims[:-1], dims[1:], blocks_per_level))
        cur_res = res

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out, num_blocks) in enumerate(in_out):
            is_last = ind == len(mults) - 1
            cur_dropout = dropout if cur_res <= dropout_res else 0.0
            layers = nn.ModuleList([])
            for _ in range(num_blocks):
                layers.append(nn.ModuleList([
                    ConvBlock(dim_in, dim_in, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout),
                    FlashAttnBlock(dim_in, emb_dim=time_emb_dim, z_dim=z_dim) if cur_res in attn_res else Identity(),
                ]))
            
            downsample = get_conv(conv_type, dim_in, dim_out, downsample=True) if not is_last else get_conv(conv_type, dim_in, dim_out)
            self.downs.append(nn.ModuleList([layers, downsample]))
            cur_res = cur_res // 2 if not is_last else cur_res
        
        self.mid_block1 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)
        self.mid_attn = FlashAttnBlock(dim_out, emb_dim=time_emb_dim, z_dim=z_dim)
        self.mid_block2 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out, num_blocks) in enumerate(reversed(in_out)):
            is_last = ind == len(mults) - 1
            cur_dropout = dropout if cur_res <= dropout_res else 0.0
            layers = nn.ModuleList([])
            for _ in range(num_blocks):
                layers.append(nn.ModuleList([
                    ConvBlock(dim_out + dim_in, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout),
                    FlashAttnBlock(dim_out, emb_dim=time_emb_dim, z_dim=z_dim) if cur_res in attn_res else Identity(),
                ]))

            upsample = get_conv(conv_type, dim_out, dim_in, upsample=True) if not is_last else get_conv(conv_type, dim_out, dim_in)
            self.ups.append(nn.ModuleList([layers, upsample]))
            cur_res = cur_res * 2 if not is_last else cur_res

        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, nout, 1)

    def forward(self, x, emb=None, z=None):
        # NOTE: Get grid can probably be replaced with fourier features or something?
        if self.conv_type == "spectral":
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)

        h = []
        for level, down in self.downs:
            for layers in level:
                for layer in layers:
                    x = layer(x, emb, z=z)
                h.append(x)
            x = down(x)
        
        x = self.mid_block1(x, emb, z=z)
        x = self.mid_attn(x, emb, z=z)
        x = self.mid_block2(x, emb, z=z)

        for level, up in self.ups:
            for layers in level:
                h_pop = h.pop()
                x = torch.cat((x, h_pop), dim=1)
                for layer in layers:
                    x = layer(x, emb, z=z)
            x = up(x)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

class UNOEncoder(nn.Module):
    def __init__(self, nin, nout, width=64, mults=(1,2,4,8), blocks_per_level=(2,2,2,2), time_emb_dim=None, 
                 z_dim=None, conv_type="conv", res=64, attn_res=[16,8], dropout_res=16, dropout=0.1):
        super().__init__()
        self.width = width
        self.conv_type = conv_type
        self.fc0 = nn.Conv2d(nin+2 if conv_type == "spectral" else nin, width, 1)

        dims = [width, *map(lambda m: width * m, mults)]
        in_out = list(zip(dims[:-1], dims[1:], blocks_per_level))
        cur_res = res

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out, num_blocks) in enumerate(in_out):
            is_last = ind == len(mults) - 1
            cur_dropout = dropout if cur_res <= dropout_res else 0.0
            layers = nn.ModuleList([])
            for _ in range(num_blocks):
                layers.append(nn.ModuleList([
                    ConvBlock(dim_in, dim_in, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout),
                    FlashAttnBlock(dim_in, emb_dim=time_emb_dim, z_dim=z_dim) if cur_res in attn_res else Identity(),
                ]))
            
            downsample = get_conv(conv_type, dim_in, dim_out, downsample=True) if not is_last else get_conv(conv_type, dim_in, dim_out)
            self.downs.append(nn.ModuleList([layers, downsample]))
            cur_res = cur_res // 2 if not is_last else cur_res
        
        self.mid_block1 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)
        self.mid_attn = FlashAttnBlock(dim_out, emb_dim=time_emb_dim, z_dim=z_dim)
        self.mid_block2 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)

        self.fc1 = nn.Conv2d(dim_out, nout, 1)

    def forward(self, x, emb=None, z=None):
        # NOTE: Get grid can probably be replaced with fourier features or something?
        if self.conv_type == "spectral":
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)

        for level, down in self.downs:
            for layers in level:
                for layer in layers:
                    x = layer(x, emb, z=z)
            x = down(x)
        
        x = self.mid_block1(x, emb, z=z)
        x = self.mid_attn(x, emb, z=z)
        x = self.mid_block2(x, emb, z=z)

        x = self.fc1(x)

        return x

    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

def get_conv(conv_type, nin, nout, downsample=False, upsample=False):
    if conv_type == "conv":
        if downsample:
            return nn.Conv2d(nin, nout, 3, stride=2, padding=1)
        elif upsample:
            return nn.ConvTranspose2d(nin, nout, 4, stride=2, padding=1)
        else:
            return nn.Conv2d(nin, nout, 3, 1, 1)
    elif conv_type == "spectral":
        # Same as conv but the kernel is defined in fourier space, 
        # allowing better generalisation to different resolutions.
        # Does not support AMP.
        # TODO: 4 modes or 3???
        if downsample:
            return SpectralConv2d(nin, nout, 4, 4, out_mult=0.5)
        elif upsample:
            return SpectralConv2d(nin, nout, 4, 4, out_mult=2.0)
        else:
            return SpectralConv2d(nin, nout, 4, 4)
    else:
        raise Exception("Unknown Convolution name. Expected either 'conv' or 'spectral'")


class ConvBlock(nn.Module):
    def __init__(self, nin, nout, emb_dim=None, z_dim=None, dropout=0.0, conv_type="conv"):
        super().__init__()
        self.in_layers = nn.Sequential(
            LayerNorm((1, nin, 1, 1)), # TODO: Use GroupNorm instead like BeatGANs? Spatial though
            nn.GELU(), # TODO: Use SiLU instead like BeatGANs?
            get_conv(conv_type, nin, nout),
        )
        self.norm = LayerNorm((1, nout, 1, 1))
        self.out_layers = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=dropout),
            get_conv(conv_type, nout, nout)
        )
        self.res_conv = nn.Conv2d(nin, nout, 1) if nin != nout else nn.Identity()

        if emb_dim is not None:
            self.time = nn.Sequential(nn.GELU(), nn.Linear(emb_dim, nout*2))
        if z_dim is not None:
            self.z_mlp = nn.Sequential(nn.Linear(z_dim, nout), nn.GELU(), nn.Linear(nout, nout))
    
    def forward(self, x, t=None, z=None):
        # TODO: Should be blocks be made more like FNO blocks???
        h = self.in_layers(x)

        # Condition on t and z
        h = self.norm(h)
        if t is not None:
            t_scale, t_shift = self.time(t)[:,:,None,None].chunk(2, dim=1)
            h = h * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = self.z_mlp(z)[:,:,None,None]
            h = h * (1 + z_scale)
        
        h = self.out_layers(h)

        return h + self.res_conv(x)

# class GroupNorm(nn.GroupNorm):
#     def forward(self, x):
#         return super().forward(x.float()).type(x.dtype)

class FlashAttnBlock(nn.Module):
    def __init__(self, dim, min_heads=4, dim_head=32, mult=2, emb_dim=None, z_dim=None):
        super().__init__()
        self.num_heads = num_heads = max(dim // dim_head, min_heads)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, num_heads*dim_head*3)
        self.attn_linear = nn.Linear(num_heads*dim_head, dim)
        self.attn = FlashAttention()

    def forward(self, x, t=None, z=None):
        height = x.size(2)
        h = rearrange(x, "b c h w -> b (h w) c")
        qkv = self.qkv(self.norm(h))
        # split qkv and separate heads
        qkv = rearrange(qkv, "b l (three h c) -> b l three h c", three=3, h=self.num_heads)

        # Do Flash Attention
        h, _ = self.attn(qkv)

        h = rearrange(h, "b l h c -> b l (h c)")
        h = self.attn_linear(h)
        h = rearrange(h, "b (h w) c -> b c h w", h=height)

        return x + h

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x

class LayerNorm(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.weight + self.bias

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, out_shape=None, out_mult=None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        assert not (out_shape is not None and out_mult is not None), "Both out_shape or out_mult can't be set at once"
        self.out_shape, self.out_mult = None, None
        if out_shape is not None:
            self.out_shape = out_shape if isinstance(out_shape, tuple) else (out_shape, out_shape)
        if out_mult is not None:
            self.out_mult = out_mult if isinstance(out_mult, tuple) else (out_mult, out_mult)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input.to(weights.dtype), weights).to(input.dtype) # or input to float32?

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x) # TODO: Make this norm="forward"?

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        if self.out_shape is not None:
            # change shape to self.out_shape
            x = torch.fft.irfft2(out_ft, s=self.out_shape)
        elif self.out_mult:
            # change shape to multiple of current shape
            out_shape = (int(x.size(-2) * self.out_mult[0]), int(x.size(-1) * self.out_mult[1]))
            x = torch.fft.irfft2(out_ft, s=out_shape)
        else:
            # keep shape the same
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
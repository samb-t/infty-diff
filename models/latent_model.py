import torch
import torch.nn as nn
import math

class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers
    default MLP for the latent DPM in the Diffusion Autoencoders paper
    """
    def __init__(self, channels, hid_channels=2048, num_layers=10, time_emb_channels=64, dropout=0.0):
        super().__init__()
        self.time_emb_channels = time_emb_channels

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_channels, channels), 
            nn.GELU(), 
            nn.Linear(channels, channels))

        self.in_layer = MLPLNAct(channels, hid_channels, cond_channels=channels, dropout=dropout)

        self.layers = nn.ModuleList([
                MLPLNAct(channels+hid_channels, hid_channels, cond_channels=channels, dropout=dropout) 
            for _ in range(num_layers - 2)])
    
        self.out_layer = nn.Linear(channels+hid_channels, channels)

    def forward(self, x, t):
        t = timestep_embedding(t, self.time_emb_channels)
        cond = self.time_embed(t)
        h = self.in_layer(x, cond=cond)
        for layer in self.layers:
            h = torch.cat((h, x), dim=1)
            h = layer(h, cond=cond)
        h = torch.cat((h, x), dim=1)
        h = self.out_layer(h)
        return h

class MLPLNAct(nn.Module):
    def __init__(self, in_channels, out_channels, use_cond=True, cond_channels=64, dropout=0.1):
        super().__init__()
        self.use_cond = use_cond
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.GELU()
        if self.use_cond:
            self.cond_layers = nn.Sequential(self.act, nn.Linear(cond_channels, out_channels))
        self.norm = nn.LayerNorm(out_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            x = x * (1.0 + cond)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
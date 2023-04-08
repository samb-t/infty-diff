import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torchvision
from torchvision import transforms
import numpy as np
import os
import imageio
import json
import pickle
from PIL import Image
import random
from pathlib import Path
from ml_collections import ConfigDict
import wandb
from tqdm import tqdm

################################################################
# CelebA Dataset
################################################################

class ImageFolder(Dataset):
    def __init__(self, root_path, split_file=None, split_key='train', first_k=None,
                 repeat=1, cache='bin', augment=False, flip_p=0.5):
        self.repeat = repeat
        self.cache = cache
        self.augment = augment
        self.flip_p = flip_p

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            x = transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255

        if self.augment:
            vflip = random.random() < self.flip_p
            augment = lambda x: x.flip(-1) if vflip else x
            x = augment(x)
        
        return x

def flatten_collection(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(flatten_collection(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class EnumerateDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return idx, self.dataset[idx]

def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset

def get_data_loader(H, enumerate_data=False, override_img_size=None, flip_p=0.5, 
                    drop_last=True, shuffle=True, train_val_split_ratio=0.95):
    img_size = H.data.img_size if override_img_size is None else override_img_size
    if H.data.name == 'mnist':
        transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)
    elif H.data.name == 'celeba':
        dataset = ImageFolder(H.data.root_dir+f"{img_size}", split_file=H.data.root_dir+f"split.json", cache='bin', augment=True, flip_p=flip_p)
        val_dataset = ImageFolder(H.data.root_dir+f"{img_size}", split_file=H.data.root_dir+f"split.json", cache='bin', split_key='val', augment=True, flip_p=flip_p)
    elif H.data.name == 'churches':
        transform = transforms.Compose([
            transforms.CenterCrop(256), 
            transforms.RandomHorizontalFlip(flip_p), 
            transforms.Resize(img_size), 
            transforms.ToTensor()])
        dataset = torchvision.datasets.LSUN(H.data.root_dir, classes=["church_outdoor_train"], transform=transform)
        val_dataset = torchvision.datasets.LSUN(H.data.root_dir, classes=["church_outdoor_val"], transform=transform)
    elif H.data.name == 'ffhq':
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.RandomHorizontalFlip(flip_p),
            torchvision.transforms.ToTensor()
        ])
        dataset = torchvision.datasets.ImageFolder(H.data.root_dir, transform=transform)
        dataset, val_dataset = train_val_split(dataset, train_val_split_ratio)
    else:
        raise Exception("Dataset not recognised")

    if enumerate_data:
        dataset = EnumerateDataset(dataset)
    
    dataloader = DataLoader(dataset, batch_size=H.train.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=H.train.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=4)

    return dataloader, val_dataloader

def optim_warmup(step, optim, lr, warmup_iters):
    lr = lr * float(step) / warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def plot_images(H, x, title='', norm=True, vis=None):
    x = (x + 1) / 2 if norm else x
    x = torch.clamp(x, 0, 1)

    # visdom
    # if H.run.enable_visdom and vis is not None:
    #     vis.images(x, win=title, opts=dict(title=title))
    
    # wandb
    x = wandb.Image(x, caption=title)
    return {title: x}
    
def update_ema(model, ema_model, ema_rate):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))

def load_latents(H, Encoder=None):
    latent_path = f"checkpoints/{H.run.experiment}/latents.pkl"
    if os.path.exists(latent_path):
        return torch.load(latent_path)
    else:
        print("Latent file not found so generating...")
        os.makedirs(latent_path.replace("latents.pkl", ""), exist_ok=True)
        decoder_checkpoint_path = f'checkpoints/{H.decoder.run.experiment}/checkpoint.pkl'
        decoder_state_dict = torch.load(decoder_checkpoint_path, map_location="cpu")
        encoder = Encoder(
            out_channels=H.decoder.model.z_dim,
            channels=H.decoder.data.channels,
            nf=H.decoder.model.nf,
            img_size=H.decoder.data.img_size,
            num_conv_blocks=H.decoder.model.num_conv_blocks,
            knn_neighbours=H.decoder.model.knn_neighbours,
            uno_res=H.decoder.model.uno_res,
            uno_mults=H.decoder.model.uno_mults,
            conv_type=H.decoder.model.uno_conv_type,
            depthwise_sparse=H.decoder.model.depthwise_sparse,
            kernel_size=H.decoder.model.kernel_size,
            backend=H.decoder.model.backend,
            blocks_per_level=H.decoder.model.uno_blocks_per_level,
            attn_res=H.decoder.model.uno_attn_resolutions,
            dropout_res=H.decoder.model.uno_dropout_from_resolution,
            dropout=H.decoder.model.uno_dropout,
            uno_base_nf=H.decoder.model.uno_base_channels,
            stochastic=H.decoder.model.stochastic_encoding
        )
        encoder.load_state_dict(decoder_state_dict["encoder_state_dict"])
        encoder = encoder.to("cuda")

        dataloader, val_dataloader = get_data_loader(H.decoder, flip_p=0.0, drop_last=False, shuffle=False)
        flipped_dataloader, val_flipped_dataloader = get_data_loader(H.decoder, flip_p=1.0, drop_last=False, shuffle=False)

        latents = []
        mus = []
        logvars = []
        for x, x_flip in tqdm(zip(dataloader, flipped_dataloader), total=len(dataloader)):
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            if isinstance(x_flip, tuple) or isinstance(x_flip, list):
                x_flip = x_flip[0]
            x, x_flip = x.to("cuda"), x_flip.to("cuda")
            
            x = torch.cat((x, x_flip), dim=0)
            x = x * 2 - 1
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=H.decoder.train.amp):
                    if H.decoder.model.stochastic_encoding:
                        z, mu, logvar = encoder(x)
                        latents.append(z.cpu())
                        mus.append(mu.cpu())
                        logvars.append(logvar.cpu())
                    else:
                        z = encoder(x)
                        latents.append(z.cpu())
        latents = torch.cat(latents, dim=0)
        if H.decoder.model.stochastic_encoding:
            mus = torch.cat(mus, dim=0)
            logvars = torch.cat(logvars, dim=0)

        mean = latents.mean(dim=0, keepdim=True) # BUG: Results in NaNs even though they should be 0.
        mean = torch.zeros_like(mean)
        std = latents.std(dim=0, keepdim=True)
        print(f"Mean: {mean} ({mean.mean()}), std: {std} ({std.mean()})")
        latents = (latents - mean) / std

        # Validaton set
        val_latents = []
        val_mus = []
        val_logvars = []
        for x, x_flip in tqdm(zip(val_dataloader, val_flipped_dataloader), total=len(val_dataloader)):
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            if isinstance(x_flip, tuple) or isinstance(x_flip, list):
                x_flip = x_flip[0]
            x, x_flip = x.to("cuda"), x_flip.to("cuda")
            
            x = torch.cat((x, x_flip), dim=0)
            x = x * 2 - 1
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=H.decoder.train.amp):
                    if H.decoder.model.stochastic_encoding:
                        z, mu, logvar = encoder(x)
                        val_latents.append(z.cpu())
                        val_mus.append(mu.cpu())
                        val_logvars.append(logvar.cpu())
                    else:
                        z = encoder(x)
                        val_latents.append(z.cpu())
        val_latents = torch.cat(val_latents, dim=0)
        if H.decoder.model.stochastic_encoding:
            val_mus = torch.cat(val_mus, dim=0)
            val_logvars = torch.cat(val_logvars, dim=0)
        val_latents = (val_latents - mean) / std

        if H.decoder.model.stochastic_encoding:
            torch.save((mus, logvars, val_mus, val_logvars, mean, std), latent_path)
            return (mus, logvars, val_mus, val_logvars, mean, std)
        else:
            torch.save((latents, val_latents, mean, std), latent_path)
            return (latents, val_latents, mean, std)
    

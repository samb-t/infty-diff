import sys
sys.path.append('.')

import torch
from torch.utils.data import ConcatDataset
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
from utils import get_data_loader
from diffusion import DCTGaussianBlur

import numpy as np
import os
import imageio

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_float("inv_snr", 0.05, "Wiener filter inverse signal to noise ratio.", lower_bound=0.0)
flags.DEFINE_integer("sample_img_size", None, "The image size to sample at.")
flags.DEFINE_string("model", "clip_vit_b_32", "The model to get features from.")
flags.DEFINE_string("sampling_steps", None, "Number of sampling steps.")
flags.mark_flags_as_required(["config"])

class DeblurDataset(torch.utils.data.Dataset):
    def __init__(self, folder, mollifier, deblurrer=None):
        self.folder = folder
        self.image_paths = os.listdir(folder)
        self.mollifier = mollifier
        self.deblurrer = deblurrer

    def __getitem__(self, index):
        path = self.folder+self.image_paths[index]
        x = torch.load(path).to(torch.float32)
        if self.deblurrer is not None:
            x = self.deblurrer(x.unsqueeze(0)).squeeze(0)
        else:
            x = self.mollifier.undo_wiener(x)
        x = (x + 1) / 2
        x = (x * 255).clamp(0, 255).to(torch.uint8)
        return x.permute(1,2,0).numpy() # to channels last

    def __len__(self):
        return len(self.image_paths)

class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)

    def __getitem__(self, index):
        x = self.dataset[index]
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]
        x = x.mul(255).clamp_(0, 255).to(torch.uint8)
        return x.permute(1,2,0).numpy() # to channels last

    def __len__(self):
        return self.length

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        return img

    def __len__(self):
        return len(self.image_paths)

from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance, get_batch_features
from cleanfid.resize import build_resizer
import torchvision
from tqdm import tqdm

class ResizeDataset(torch.utils.data.Dataset):
    def __init__(self, dset, mode, size=(299, 299)):
        self.dset = dset
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.custom_image_tranform = lambda x: x
        self.fn_resize = build_resizer(mode)
    
    def __len__(self):
        return len(self.dset)
    
    def __getitem__(self, i):
        img_np = self.dset[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)
        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t

def get_dset_features(dset, model=None, num_workers=12, num=None,
            shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
            mode="clean", custom_fn_resize=None, description="", verbose=True,
            custom_image_transform=None):
    
    dataset = ResizeDataset(dset, mode=mode)
    if custom_image_transform is not None:
        dataset.custom_image_tranform=custom_image_transform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize
    
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers)
    
    # collect all features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader
    
    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats

def compute_fid_from_datasets(dset1, dset2, 
            mode="clean", model_name="clip_vit_b_32", num_workers=12,
            batch_size=32, device=torch.device("cuda"),
            custom_feat_extractor=None, verbose=True,
            custom_image_tranform=None, custom_fn_resize=None, use_dataparallel=True):
    if custom_feat_extractor is None and model_name=="inception_v3":
        feat_model = build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
    elif custom_feat_extractor is None and model_name=="clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        feat_model = custom_feat_extractor
    # compare_folders
    # -> get_folder_features
    np_feats1 = get_dset_features(dset1, feat_model, num_workers=num_workers,
                                  batch_size=batch_size, device=device, mode=mode,
                                  verbose=verbose, custom_image_transform=custom_image_tranform,
                                  custom_fn_resize=custom_fn_resize)
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)

    np_feats2 = get_dset_features(dset2, feat_model, num_workers=num_workers,
                                  batch_size=batch_size, device=device, mode=mode,
                                  verbose=verbose, custom_image_transform=custom_image_tranform,
                                  custom_fn_resize=custom_fn_resize)
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid



def main(argv):
    H = FLAGS.config
    sample_img_size = FLAGS.sample_img_size
    inv_snr = FLAGS.inv_snr
    feat_model = FLAGS.model
    sampling_steps = FLAGS.sampling_steps

    img_size = sample_img_size if sample_img_size is not None else H.data.img_size

    if H.diffusion.gaussian_filter_std > 0.0:
        if sample_img_size is not None:
            new_std = H.diffusion.gaussian_filter_std * (sample_img_size / H.data.img_size)
            mollifier = DCTGaussianBlur(sample_img_size, std=new_std, inv_snr=inv_snr)
        else:
            mollifier = DCTGaussianBlur(H.data.img_size, H.diffusion.gaussian_filter_std, inv_snr=inv_snr)

    # TODO: Merge train+test datasets for celeba and ffhq
    train_loader, test_loader = get_data_loader(H, override_img_size=img_size)
    if H.data.name in ["ffhq"]:
        real_dataset = ConcatDataset((train_loader.dataset, test_loader.dataset))
    else:
        real_dataset = train_loader.dataset
    real_dataset = NoClassDataset(real_dataset)

    if sampling_steps is not None:
        samples_folder = f"checkpoints/{H.run.experiment}_samples_{img_size}_{sampling_steps}/"
    else:
        samples_folder = f"checkpoints/{H.run.experiment}_samples_{img_size}/"
    if H.diffusion.gaussian_filter_std > 0.0:
        samples_dataset = DeblurDataset(samples_folder, mollifier)
    else:
        samples_dataset = BigDataset(samples_folder)

    fid = compute_fid_from_datasets(real_dataset, samples_dataset, model_name=feat_model)
    print(fid)

if __name__ == '__main__':
    app.run(main)

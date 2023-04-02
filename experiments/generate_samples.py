import sys
sys.path.append('.')

import torch

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
from tqdm import tqdm

from models import SparseUNet, SparseEncoder, MLPSkipNet
from utils import load_latents
import diffusion as gd
from diffusion import GaussianDiffusion, get_named_beta_schedule, SpacedDiffusion, space_timesteps

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("decoder_config", None, "Decoder training configuration.", lock_config=True)
flags.DEFINE_integer("sample_img_size", None, "The image size to sample at.")
flags.DEFINE_string("sampling_steps", "100", "Number of diffusion steps when sampling.")
flags.mark_flags_as_required(["config", "decoder_config"])

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda')

def make_samples(H, latent_model, decoder, diffusion, decoder_diffusion, 
                 latents_mean, latents_std, sample_img_size, sampling_steps):
    img_size = sample_img_size if sample_img_size is not None else H.decoder.data.img_size
    if sample_img_size is not None:
        new_std = H.decoder.diffusion.gaussian_filter_std * (sample_img_size / H.decoder.data.img_size)
        decoder_diffusion.mollifier = gd.DCTGaussianBlur(img_size, std=new_std).to(device)
    os.makedirs(f"checkpoints/{H.run.experiment}_samples_{img_size}_{sampling_steps}", exist_ok=True)
    noise_mul = img_size / H.decoder.data.img_size
    idx = 0
    for _ in tqdm(range(H.data.fid_samples // H.train.sample_size + 1)):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=H.train.amp):
                samples, _ = diffusion.p_sample_loop(
                        latent_model, 
                        (H.train.sample_size, 
                        H.decoder.model.z_dim), 
                        progress=False, 
                        return_all=False, 
                        clip_denoised=False
                     )
                samples = samples * latents_std.to(samples.dtype) + latents_mean.to(samples.dtype)
                img, _ = decoder_diffusion.ddim_sample_loop(
                        decoder, 
                        (H.train.sample_size, H.decoder.data.channels, img_size, img_size), 
                        progress=False, 
                        model_kwargs=dict(z=samples), 
                        return_all=False,
                        noise_mul=noise_mul
                    )
                for x in img:
                    torch.save(x.cpu().to(torch.float16), f"checkpoints/{H.run.experiment}_samples_{img_size}_{sampling_steps}/{idx:05}.pkl")
                    idx += 1
                    if idx == H.data.fid_samples:
                        exit()


def main(argv):
    sample_img_size = FLAGS.sample_img_size
    sampling_steps = FLAGS.sampling_steps
    H = FLAGS.config
    H.decoder = FLAGS.decoder_config
    train_kwargs = {}

    latent_model = MLPSkipNet(
        H.decoder.model.z_dim, 
        hid_channels=H.model.hid_channels, 
        num_layers=H.model.num_layers, 
        time_emb_channels=H.model.time_embed_dim, 
        dropout=H.model.dropout
    )
    latent_checkpoint_path = f'checkpoints/{H.run.experiment}/latent_checkpoint.pkl'
    latent_state_dict = torch.load(latent_checkpoint_path, map_location="cpu")
    print(f"Loading Latent Model from step {latent_state_dict['global_step']}")
    latent_model.load_state_dict(latent_state_dict["model_ema_state_dict"])

    decoder = SparseUNet(
            channels=H.decoder.data.channels,
            nf=H.decoder.model.nf,
            time_emb_dim=H.decoder.model.time_emb_dim,
            img_size=H.decoder.data.img_size,
            num_conv_blocks=H.decoder.model.num_conv_blocks,
            knn_neighbours=H.decoder.model.knn_neighbours,
            uno_res=H.decoder.model.uno_res,
            uno_mults=H.decoder.model.uno_mults,
            z_dim=H.decoder.model.z_dim,
            conv_type=H.decoder.model.uno_conv_type,
            depthwise_sparse=H.decoder.model.depthwise_sparse,
            kernel_size=H.decoder.model.kernel_size,
            backend=H.decoder.model.backend,
            blocks_per_level=H.decoder.model.uno_blocks_per_level,
            attn_res=H.decoder.model.uno_attn_resolutions,
            dropout_res=H.decoder.model.uno_dropout_from_resolution,
            dropout=H.decoder.model.uno_dropout,
            uno_base_nf=H.decoder.model.uno_base_channels,
            # continuous_conv=H.decoder.model.continuous_conv
        )
    decoder_checkpoint_path = f'checkpoints/{H.decoder.run.experiment}/checkpoint.pkl'
    decoder_state_dict = torch.load(decoder_checkpoint_path, map_location="cpu")
    print(f"Loading Decoder from step {decoder_state_dict['global_step']}")
    decoder.load_state_dict(decoder_state_dict["model_ema_state_dict"])

    latent_model = latent_model.to(device)
    decoder = decoder.to(device)

    ## Load encodings
    _, _, latents_mean, latents_std = load_latents(H, Encoder=SparseEncoder)[:4]
    latents_mean = latents_mean.to(device)
    latents_std = latents_std.to(device)

    ## Setup diffusion for latent model
    betas = get_named_beta_schedule(H.diffusion.noise_schedule, H.diffusion.steps)
    if H.diffusion.model_mean_type == "epsilon":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif H.diffusion.model_mean_type == "v":
        model_mean_type = gd.ModelMeanType.V
    elif H.diffusion.model_mean_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    else:
        raise Exception("Unknown model mean type. Expected value in [epsilon, v, xstart]")
    model_var_type=((gd.ModelVarType.FIXED_LARGE if not H.model.sigma_small else gd.ModelVarType.FIXED_SMALL) if not H.model.learn_sigma else gd.ModelVarType.LEARNED_RANGE)
    loss_type = gd.LossType.MSE if H.diffusion.loss_type == 'mse' else gd.LossType.RESCALED_MSE
    diffusion = GaussianDiffusion(
            betas, 
            model_mean_type, 
            model_var_type, 
            loss_type, 
            rescale_timesteps=True
        ).to(device)

    ## Setup diffusion for decoder model
    betas = get_named_beta_schedule(H.decoder.diffusion.noise_schedule, H.decoder.diffusion.steps, resolution=H.decoder.data.img_size)
    if H.decoder.diffusion.model_mean_type == "epsilon":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif H.decoder.diffusion.model_mean_type == "v":
        model_mean_type = gd.ModelMeanType.V
    elif H.decoder.diffusion.model_mean_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    elif H.decoder.diffusion.model_mean_type == "mollified_epsilon":
        assert H.decoder.diffusion.gaussian_filter_std > 0, "Error: Predicting mollified_epsilon but gaussian_filter_std == 0."
        model_mean_type = gd.ModelMeanType.MOLLIFIED_EPSILON
    else:
        raise Exception("Unknown model mean type. Expected value in [epsilon, v, xstart]")
    model_var_type=((gd.ModelVarType.FIXED_LARGE if not H.model.sigma_small else gd.ModelVarType.FIXED_SMALL) if not H.model.learn_sigma else gd.ModelVarType.LEARNED_RANGE)
    loss_type = gd.LossType.MSE if H.decoder.diffusion.loss_type == 'mse' else gd.LossType.RESCALED_MSE
    skipped_timestep_respacing = sampling_steps
    skipped_use_timesteps = space_timesteps(H.decoder.diffusion.steps, skipped_timestep_respacing)
    decoder_diffusion = SpacedDiffusion(
            skipped_use_timesteps, 
            betas=betas, 
            model_mean_type=model_mean_type, 
            model_var_type=model_var_type, 
            loss_type=loss_type,
            gaussian_filter_std=H.decoder.diffusion.gaussian_filter_std,
            img_size=H.decoder.data.img_size,
            rescale_timesteps=True,
            multiscale_loss=H.decoder.diffusion.multiscale_loss, 
            multiscale_max_img_size=H.decoder.diffusion.multiscale_max_img_size,
            mollifier_type=H.decoder.diffusion.mollifier_type,
        ).to(device)
    
    make_samples(
            H, 
            latent_model, 
            decoder, 
            diffusion, 
            decoder_diffusion, 
            latents_mean=latents_mean, 
            latents_std=latents_std,
            sample_img_size=sample_img_size,
            sampling_steps=sampling_steps
        )

if __name__ == '__main__':
    app.run(main)

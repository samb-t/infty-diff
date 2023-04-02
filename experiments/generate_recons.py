import sys
sys.path.append('.')

import torch
from torchvision.utils import save_image

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
from tqdm import tqdm

from models import SparseUNet, SparseEncoder
from utils import get_data_loader
import diffusion as gd
from diffusion import get_named_beta_schedule, SpacedDiffusion, space_timesteps


# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.DEFINE_integer("sample_img_size", None, "The image size to sample at.")
flags.mark_flags_as_required(["config"])

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda')

def make_samples(H, decoder, encoder, decoder_diffusion, train_loader, sample_img_size):
    img_size = sample_img_size if sample_img_size is not None else H.data.img_size
    if sample_img_size is not None:
        new_std = H.diffusion.gaussian_filter_std * (sample_img_size / H.data.img_size)
        decoder_diffusion.mollifier = gd.DCTGaussianBlur(img_size, std=new_std, inv_snr=0.05).to(device)
    os.makedirs(f"checkpoints/{H.run.experiment}_samples_{img_size}", exist_ok=True)
    noise_mul = img_size / H.data.img_size
    idx = 0
    for x in tqdm(train_loader, total=(H.data.fid_samples//H.train.batch_size+1)):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]
        x = x.to(device) * 2 - 1
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=H.train.amp):
                encoding = encoder(x)
                img, _ = decoder_diffusion.ddim_sample_loop(
                        decoder, 
                        (encoding.size(0), H.data.channels, img_size, img_size), 
                        progress=False, 
                        model_kwargs=dict(z=encoding), 
                        return_all=False,
                        noise_mul=noise_mul
                    )
                for x in img:
                    if H.diffusion.gaussian_filter_std > 0.0:
                        torch.save(x.cpu().to(torch.float16), f"checkpoints/{H.run.experiment}_samples_{img_size}/{idx:05}.pkl")
                    else:
                        save_image(torch.clamp((x + 1) / 2, 0, 1), f"checkpoints/{H.run.experiment}_samples_{img_size}/{idx:05}.png")
                    idx += 1
                    if idx == H.data.fid_samples:
                        exit()


def main(argv):
    sample_img_size = FLAGS.sample_img_size
    H = FLAGS.config
    train_kwargs = {}

    decoder = SparseUNet(
            channels=H.data.channels,
            nf=H.model.nf,
            time_emb_dim=H.model.time_emb_dim,
            img_size=H.data.img_size,
            num_conv_blocks=H.model.num_conv_blocks,
            knn_neighbours=H.model.knn_neighbours,
            uno_res=H.model.uno_res,
            uno_mults=H.model.uno_mults,
            z_dim=H.model.z_dim,
            conv_type=H.model.uno_conv_type,
            depthwise_sparse=H.model.depthwise_sparse,
            kernel_size=H.model.kernel_size,
            backend=H.model.backend,
            blocks_per_level=H.model.uno_blocks_per_level,
            attn_res=H.model.uno_attn_resolutions,
            dropout_res=H.model.uno_dropout_from_resolution,
            dropout=H.model.uno_dropout,
            uno_base_nf=H.model.uno_base_channels,
            # continuous_conv=H.model.continuous_conv
        )
    encoder = SparseEncoder(
            out_channels=H.model.z_dim,
            channels=H.data.channels,
            nf=H.model.nf,
            img_size=H.data.img_size,
            num_conv_blocks=H.model.num_conv_blocks,
            knn_neighbours=H.model.knn_neighbours,
            uno_res=H.model.uno_res,
            uno_mults=H.model.uno_mults,
            conv_type=H.model.uno_conv_type,
            depthwise_sparse=H.model.depthwise_sparse,
            kernel_size=H.model.kernel_size,
            backend=H.model.backend,
            blocks_per_level=H.model.uno_blocks_per_level,
            attn_res=H.model.uno_attn_resolutions,
            dropout_res=H.model.uno_dropout_from_resolution,
            dropout=H.model.uno_dropout,
            uno_base_nf=H.model.uno_base_channels,
            stochastic=H.model.stochastic_encoding
        )
    decoder_checkpoint_path = f'checkpoints/{H.run.experiment}/checkpoint.pkl'
    decoder_state_dict = torch.load(decoder_checkpoint_path, map_location="cpu")
    print(f"Loading Decoder from step {decoder_state_dict['global_step']}")
    decoder.load_state_dict(decoder_state_dict["model_ema_state_dict"])
    encoder.load_state_dict(decoder_state_dict["encoder_state_dict"])

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    ## Load encodings
    train_loader, _ = get_data_loader(H)

    ## Setup diffusion for decoder model
    betas = get_named_beta_schedule(H.diffusion.noise_schedule, H.diffusion.steps, resolution=H.data.img_size)
    if H.diffusion.model_mean_type == "epsilon":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif H.diffusion.model_mean_type == "v":
        model_mean_type = gd.ModelMeanType.V
    elif H.diffusion.model_mean_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    elif H.diffusion.model_mean_type == "mollified_epsilon":
        assert H.diffusion.gaussian_filter_std > 0, "Error: Predicting mollified_epsilon but gaussian_filter_std == 0."
        model_mean_type = gd.ModelMeanType.MOLLIFIED_EPSILON
    else:
        raise Exception("Unknown model mean type. Expected value in [epsilon, v, xstart]")
    model_var_type=((gd.ModelVarType.FIXED_LARGE if not H.model.sigma_small else gd.ModelVarType.FIXED_SMALL) if not H.model.learn_sigma else gd.ModelVarType.LEARNED_RANGE)
    loss_type = gd.LossType.MSE if H.diffusion.loss_type == 'mse' else gd.LossType.RESCALED_MSE
    skipped_timestep_respacing = "100"
    skipped_use_timesteps = space_timesteps(H.diffusion.steps, skipped_timestep_respacing)
    decoder_diffusion = SpacedDiffusion(
            skipped_use_timesteps, 
            betas=betas, 
            model_mean_type=model_mean_type, 
            model_var_type=model_var_type, 
            loss_type=loss_type,
            gaussian_filter_std=H.diffusion.gaussian_filter_std,
            img_size=H.data.img_size,
            rescale_timesteps=True,
            multiscale_loss=H.diffusion.multiscale_loss, 
            multiscale_max_img_size=H.diffusion.multiscale_max_img_size,
            mollifier_type=H.diffusion.mollifier_type,
        ).to(device)
    
    make_samples(
            H, 
            decoder, 
            encoder,
            decoder_diffusion, 
            train_loader,
            sample_img_size=sample_img_size,
        )

if __name__ == '__main__':
    app.run(main)

import torch
import numpy as np
from scipy.stats.qmc import Halton

import wandb
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import time
import os

from models import SparseUNet, SparseEncoder
from utils import get_data_loader, flatten_collection, optim_warmup, \
    plot_images, update_ema, create_named_schedule_sampler, LossAwareSampler
import diffusion as gd
from diffusion import GaussianDiffusion, get_named_beta_schedule

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda')

def train(H, model, ema_model, encoder, train_loader, optim, diffusion, schedule_sampler, vis=None, checkpoint_path='', global_step=0):
    halton = Halton(2)
    scaler = torch.cuda.amp.GradScaler()

    mean_loss = 0
    mean_step_time = 0
    mean_total_norm = 0
    skip = 0
    while True:
        for x in train_loader:
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            start_time = time.time()

            if global_step < H.optimizer.warmup_steps:
                optim_warmup(global_step, optim, H.optimizer.learning_rate, H.optimizer.warmup_steps)

            global_step += 1
            x = x.to(device, non_blocking=True)
            x = x * 2 - 1 

            t, weights = schedule_sampler.sample(x.size(0), device)

            if H.mc_integral.type == 'uniform':
                sample_lst = torch.stack([torch.from_numpy(np.random.choice(H.data.img_size**2, H.mc_integral.q_sample, replace=False)) for _ in range(H.train.batch_size)]).to(device)
            elif H.mc_integral.type == 'halton':
                sample_lst = torch.stack([torch.from_numpy((halton.random(H.mc_integral.q_sample) * H.data.img_size).astype(np.int64)) for _ in range(H.train.batch_size)]).to(device)
                sample_lst = sample_lst[:,:,0] * H.data.img_size + sample_lst[:,:,1]
            else:
                raise Exception('Unknown Monte Carlo Integral type')

            with torch.cuda.amp.autocast(enabled=H.train.amp):
                losses = diffusion.training_losses(model, x, t, sample_lst=sample_lst, encoder=encoder)

                if H.diffusion.multiscale_loss:
                    loss = (losses["multiscale_loss"] * weights).mean()
                else:
                    loss = (losses["loss"] * weights).mean()
            
            optim.zero_grad()
            if H.train.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                encoder_total_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                if H.optimizer.gradient_skip and max(model_total_norm, encoder_total_norm) >= H.optimizer.gradient_skip_threshold:
                    skip += 1
                else:
                    scaler.step(optim)
                    scaler.update()
            else:
                loss.backward()
                model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                encoder_total_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                if H.optimizer.gradient_skip and max(model_total_norm, encoder_total_norm) >= H.optimizer.gradient_skip_threshold:
                    skip += 1
                else:
                    optim.step()

            if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            if global_step % H.train.ema_update_every == 0:
                update_ema(model, ema_model, H.train.ema_decay)
            
            mean_loss += loss.item()
            mean_step_time += time.time() - start_time
            mean_total_norm += max(model_total_norm, encoder_total_norm).item()

            wandb_dict = dict()
            if global_step % H.train.plot_graph_steps == 0 and global_step > 0:
                norm = H.train.plot_graph_steps
                print(f"Step: {global_step}, Loss {mean_loss / norm:.5f}, Step Time: {mean_step_time / norm:.5f}, Skip: {skip / norm:.5f}, Gradient Norm: {mean_total_norm / norm:.5f}")
                wandb_dict |= {'Step Time': mean_step_time / norm, 'Loss': mean_loss / norm, 'Skip': skip / norm, "Gradient Norm": mean_total_norm / norm}
                mean_loss = 0
                mean_step_time = 0
                skip = 0
                mean_total_norm = 0
            
            if global_step % H.train.plot_samples_steps == 0 and global_step > 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        if H.model.stochastic_encoding:
                            encoding = encoder(x[:H.train.sample_size])[0]
                        else:
                            encoding = encoder(x[:H.train.sample_size])
                        samples, _ = diffusion.p_sample_loop(ema_model, (encoding.size(0), H.data.channels, H.data.img_size, H.data.img_size), progress=True, model_kwargs=dict(z=encoding), return_all=False)

                wandb_dict |= plot_images(H, samples, title='samples', vis=vis)

                if H.diffusion.model_mean_type == "mollified_epsilon":
                    wandb_dict |= plot_images(H, diffusion.mollifier.undo_wiener(samples), title=f'deblurred_samples', vis=vis)
            
            if wandb_dict:
                wandb.log(wandb_dict, step=global_step)

            if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
                torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'encoder_state_dict': encoder.state_dict(),
                        'model_ema_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                    }, checkpoint_path)
                

def main(argv):
    H = FLAGS.config
    train_kwargs = {}

    # wandb can be disabled by passing in --config.run.wandb_mode=disabled
    wandb.init(project=H.run.name, config=flatten_collection(H), save_code=True, dir=H.run.wandb_dir, mode=H.run.wandb_mode)
    
    model = SparseUNet(
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
        uno_base_nf=H.model.uno_base_channels
    )
    # NOTE: deepcopy doesn't work on Minkowksi
    ema_model = SparseUNet(
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
        uno_base_nf=H.model.uno_base_channels
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

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in encoder.parameters())}")

    if H.run.experiment != '':
        checkpoint_path = f'checkpoints/{H.run.experiment}/'
    else:
        checkpoint_path = 'checkpoints/'
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = checkpoint_path + 'checkpoint.pkl'
    train_kwargs['checkpoint_path'] = checkpoint_path

    model = model.to(device)
    encoder = encoder.to(device)
    ema_model = ema_model.to(device)
    train_loader, _ = get_data_loader(H)
    optim = torch.optim.Adam(
            list(model.parameters()) + list(encoder.parameters()), 
            lr=H.optimizer.learning_rate, 
            betas=(H.optimizer.adam_beta1, H.optimizer.adam_beta2)
        )

    if H.train.load_checkpoint and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        print(f"Loading Model from step {state_dict['global_step']}")
        train_kwargs['global_step'] = state_dict['global_step']
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        encoder.load_state_dict(state_dict['encoder_state_dict'], strict=False)
        ema_model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
        try:
            optim.load_state_dict(state_dict['optimizer_state_dict'])
        except ValueError:
            print("Failed to load optim params.")
    
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
    diffusion = GaussianDiffusion(
            betas, 
            model_mean_type, 
            model_var_type, 
            loss_type, 
            H.diffusion.gaussian_filter_std, 
            H.data.img_size,
            rescale_timesteps=True, 
            multiscale_loss=H.diffusion.multiscale_loss, 
            multiscale_max_img_size=H.diffusion.multiscale_max_img_size,
            mollifier_type=H.diffusion.mollifier_type,
            stochastic_encoding=H.model.stochastic_encoding
        ).to(device)

    schedule_sampler = create_named_schedule_sampler(H.diffusion.schedule_sampler, diffusion)

    train(H, model, ema_model, encoder, train_loader, optim, diffusion, schedule_sampler, **train_kwargs)

if __name__ == '__main__':
    app.run(main)

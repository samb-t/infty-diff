import torch
from torch.utils.data import DataLoader

import wandb
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import time
import copy
import os
from tqdm import tqdm

from models import SparseUNet, SparseEncoder, MLPSkipNet
from utils import flatten_collection, optim_warmup, \
    plot_images, update_ema, create_named_schedule_sampler, LossAwareSampler, load_latents
import diffusion as gd
from diffusion import GaussianDiffusion, get_named_beta_schedule, SpacedDiffusion, space_timesteps

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("decoder_config", None, "Decoder training configuration.", lock_config=True)
flags.mark_flags_as_required(["config", "decoder_config"])

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda')

def train(H, model, ema_model, decoder, train_loader, test_loader, optim, diffusion, 
          decoder_diffusion, schedule_sampler, vis=None, checkpoint_path='', 
          latents_mean=0.0, latents_std=1.0):
    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    mean_loss = 0
    mean_step_time = 0
    mean_total_norm = 0
    skip = 0
    best_test_loss = float("inf")

    while True:
        for x in train_loader:
            if isinstance(x, tuple) or isinstance(x, list):
                mu, logvar = x
                mu, logvar = mu.to(device), logvar.to(device)
                x = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
                x = (x - latents_mean) / latents_std
            else:
                x = x.to(device)
            start_time = time.time()

            if global_step < H.optimizer.warmup_steps:
                optim_warmup(global_step, optim, H.optimizer.learning_rate, H.optimizer.warmup_steps)

            global_step += 1
            x = x.to(device, non_blocking=True)
            t, weights = schedule_sampler.sample(x.size(0), device)

            with torch.cuda.amp.autocast(enabled=H.train.amp):
                losses = diffusion.training_losses(model, x, t)
                loss = (losses["loss"] * weights).mean()
            
            optim.zero_grad()
            if H.train.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if H.optimizer.gradient_skip and model_total_norm >= H.optimizer.gradient_skip_threshold:
                    skip += 1
                else:
                    scaler.step(optim)
                    scaler.update()
            else:
                loss.backward()
                model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if H.optimizer.gradient_skip and model_total_norm >= H.optimizer.gradient_skip_threshold:
                    skip += 1
                else:
                    optim.step()
            
            if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            if global_step % H.train.ema_update_every == 0:
                update_ema(model, ema_model, H.train.ema_decay)
            
            mean_loss += loss.item()
            mean_step_time += time.time() - start_time
            mean_total_norm += model_total_norm.item()

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
                        samples, _ = diffusion.p_sample_loop(
                                ema_model, 
                                (H.train.sample_size, x.size(1)), 
                                progress=True, 
                                return_all=False,
                                clip_denoised=False
                            )
                        samples = samples * latents_std + latents_mean
                        img, _ = decoder_diffusion.ddim_sample_loop(
                                decoder, 
                                (H.train.sample_size, H.decoder.data.channels, H.decoder.data.img_size, H.decoder.data.img_size), 
                                progress=True, 
                                model_kwargs=dict(z=samples), 
                                return_all=False
                            )
                
                wandb_dict |= plot_images(H, img, title='samples', vis=vis)
                if H.decoder.diffusion.model_mean_type == "mollified_epsilon":
                    wandb_dict |= plot_images(H, decoder_diffusion.mollifier.undo_wiener(img), title=f'deblurred_samples', vis=vis)
            
            if global_step % H.train.calculate_test_loss_steps == 0 and global_step > 0:
                # Aapproximation of test loss to assess for overfitting.
                total_loss, count = 0.0, 0
                for x in tqdm(test_loader):
                    if isinstance(x, tuple) or isinstance(x, list):
                        mu, logvar = x
                        mu, logvar = mu.to(device), logvar.to(device)
                        x = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
                        x = (x - latents_mean) / latents_std
                    else:
                        x = x.to(device)
                    for _ in range(H.train.test_loss_repeats):
                        t, weights = schedule_sampler.sample(x.size(0), device)
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=H.train.amp):
                                losses = diffusion.training_losses(ema_model, x, t)
                                loss = (losses["loss"] * weights).mean()
                        total_loss += loss.item()
                        count += 1
                print(f"Test Loss: {total_loss/count}")
                wandb_dict |= {'Test Loss': total_loss/count}

                if total_loss/count < best_test_loss:
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'model_ema_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                    }, f'checkpoints/{H.run.experiment}/best_latent_checkpoint.pkl')
                    best_test_loss = total_loss/count
            
            if wandb_dict:
                wandb.log(wandb_dict, step=global_step)

            if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
                torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'model_ema_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                    }, checkpoint_path)

def main(argv):
    H = FLAGS.config
    H.decoder = FLAGS.decoder_config
    train_kwargs = {}

    # wandb can be disabled by passing in --config.run.wandb_mode=disabled
    wandb.init(project=H.run.name, config=flatten_collection(H), save_code=True, dir=H.run.wandb_dir, mode=H.run.wandb_mode)

    model = MLPSkipNet(
        H.decoder.model.z_dim, 
        hid_channels=H.model.hid_channels, 
        num_layers=H.model.num_layers, 
        time_emb_channels=H.model.time_embed_dim, 
        dropout=H.model.dropout
    )
    ema_model = copy.deepcopy(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    optim = torch.optim.AdamW(
            model.parameters(), 
            lr=H.optimizer.learning_rate, 
            betas=(H.optimizer.adam_beta1, H.optimizer.adam_beta2),
            weight_decay=H.optimizer.weight_decay
        )

    if H.run.experiment != '':
        checkpoint_path = f'checkpoints/{H.run.experiment}/'
    else:
        checkpoint_path = 'checkpoints/'
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = checkpoint_path + 'latent_checkpoint.pkl'
    train_kwargs['checkpoint_path'] = checkpoint_path

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
        )
    decoder_checkpoint_path = f'checkpoints/{H.decoder.run.experiment}/checkpoint.pkl'
    decoder_state_dict = torch.load(decoder_checkpoint_path, map_location="cpu")
    print(f"Loading Decoder from step {decoder_state_dict['global_step']}")
    decoder.load_state_dict(decoder_state_dict["model_ema_state_dict"])

    model = model.to(device)
    ema_model = ema_model.to(device)
    decoder = decoder.to(device)

    ## Load encodings
    if H.decoder.model.stochastic_encoding:
        train_mus, train_logvars, val_mus, val_logvars, latents_mean, latents_std = load_latents(H, Encoder=SparseEncoder)
        train_dataset = torch.utils.data.TensorDataset(train_mus, train_logvars)
        train_loader = DataLoader(train_dataset, batch_size=H.train.batch_size, shuffle=True, drop_last=True, num_workers=4)
        test_dataset = torch.utils.data.TensorDataset(val_mus, val_logvars)
        test_loader = DataLoader(test_dataset, batch_size=H.train.batch_size, shuffle=True, drop_last=True, num_workers=4)
    else:
        train_dataset, test_dataset, latents_mean, latents_std = load_latents(H, Encoder=SparseEncoder)[:4]
        train_loader = DataLoader(train_dataset, batch_size=H.train.batch_size, shuffle=True, drop_last=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=H.train.batch_size, shuffle=True, drop_last=True, num_workers=4)
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
    if H.diffusion.loss_type == "mse":
        loss_type = gd.LossType.MSE 
    elif H.diffusion.loss_type == "l1":
        loss_type = gd.LossType.L1
    elif H.diffusion.loss_type == "rescaled_mse":
        loss_type = gd.LossType.RESCALED_MSE
    else:
        raise Exception("Unknown loss type.")
    diffusion = GaussianDiffusion(
            betas, 
            model_mean_type, 
            model_var_type, 
            loss_type, 
            rescale_timesteps=True
        ).to(device)
    schedule_sampler = create_named_schedule_sampler(H.diffusion.schedule_sampler, diffusion)

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
    skipped_timestep_respacing = "100"
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
    
    train(
            H, 
            model, 
            ema_model, 
            decoder, 
            train_loader, 
            test_loader,
            optim, 
            diffusion, 
            decoder_diffusion, 
            schedule_sampler, 
            vis=None, 
            checkpoint_path=checkpoint_path, 
            latents_mean=latents_mean, 
            latents_std=latents_std
        )

if __name__ == '__main__':
    app.run(main)

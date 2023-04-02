from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference
import pwd
import os

USERNAME = pwd.getpwuid(os.getuid())[0]

def get_config():
    config = ConfigDict()

    config.run = run = ConfigDict()
    run.name = 'infty_diff_sampler'
    run.experiment = 'latent_experiment'
    run.wandb_dir = ''
    run.wandb_mode = 'online'

    config.data = data = ConfigDict()
    data.test_ratio = 0.05
    data.fid_samples = 50000

    config.train = train = ConfigDict()
    train.amp = True
    train.batch_size = 256
    train.sample_size = 8
    train.plot_graph_steps = 100
    train.plot_samples_steps = 20000
    train.calculate_test_loss_steps = 10000
    train.test_loss_repeats = 10000
    train.checkpoint_steps = 10000
    train.ema_update_every = 10
    train.ema_decay = 0.995

    config.model = model = ConfigDict()
    model.hid_channels = 2048
    model.num_layers = 10
    model.time_embed_dim = 128
    model.dropout = 0.0
    model.learn_sigma = False
    model.sigma_small = False

    config.diffusion = diffusion = ConfigDict()
    diffusion.steps = 1000
    diffusion.noise_schedule = 'const0.008'
    diffusion.schedule_sampler = 'uniform'
    diffusion.loss_type = 'l1'
    diffusion.model_mean_type = "epsilon"

    config.optimizer = optimizer = ConfigDict()
    optimizer.learning_rate = 1e-4
    optimizer.adam_beta1 = 0.9
    optimizer.adam_beta2 = 0.99
    optimizer.weight_decay = 0.04
    optimizer.warmup_steps = 0
    optimizer.gradient_skip = False
    optimizer.gradient_skip_threshold = 500.

    return config

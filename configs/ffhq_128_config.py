from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference
import pwd
import os

USERNAME = pwd.getpwuid(os.getuid())[0]

def get_config():
    config = ConfigDict()

    config.run = run = ConfigDict()
    run.name = 'infty_diff'
    run.experiment = 'ffhq_mollified_128'
    run.wandb_dir = ''
    run.wandb_mode = 'online'
    run.enable_visdom = False
    run.visdom_server = 'http://localhost' if USERNAME == 'sam' else 'http://ncc1.clients.dur.ac.uk'
    run.visdom_port = 8097 if USERNAME == 'sam' else 9275

    config.data = data = ConfigDict()
    data.name = 'ffhq'
    data.root_dir = '/home/sam/workspace/data/FFHQ-256'
    data.img_size = FieldReference(128)
    data.channels = 3
    
    config.train = train = ConfigDict()
    train.load_checkpoint = False
    train.amp = True
    train.batch_size = 16
    train.sample_size = 16
    train.plot_graph_steps = 100
    train.plot_samples_steps = 5000
    train.checkpoint_steps = 10000
    train.ema_update_every = 10
    train.ema_decay = 0.995

    config.model = model = ConfigDict()
    model.nf = 64
    model.time_emb_dim = 256
    model.num_conv_blocks = 3
    model.knn_neighbours = 3
    model.depthwise_sparse = True
    model.kernel_size = 7
    model.backend = "torchsparse"
    model.uno_res = 64
    model.uno_base_channels = 64
    model.uno_mults = (1,2,4,8)
    model.uno_blocks_per_level = (2,2,2,2)
    model.uno_attn_resolutions = [16,8]
    model.uno_dropout_from_resolution = 16
    model.uno_dropout = 0.1
    model.uno_conv_type = "conv"
    model.z_dim = 256
    model.learn_sigma = False
    model.sigma_small = False
    model.stochastic_encoding = False
    model.kld_weight = 1e-4

    config.diffusion = diffusion = ConfigDict()
    diffusion.steps = 1000
    diffusion.noise_schedule = 'cosine'
    diffusion.schedule_sampler = 'uniform'
    diffusion.loss_type = 'mse'
    diffusion.gaussian_filter_std = 1.0
    diffusion.model_mean_type = "mollified_epsilon"
    diffusion.multiscale_loss = False
    diffusion.multiscale_max_img_size = config.data.get_ref('img_size') // 2
    diffusion.mollifier_type = "dct"

    config.mc_integral = mc_integral = ConfigDict()
    mc_integral.type = 'uniform'
    mc_integral.q_sample = (config.data.get_ref('img_size') ** 2) // 4

    config.optimizer = optimizer = ConfigDict()
    optimizer.learning_rate = 1e-4
    optimizer.adam_beta1 = 0.9
    optimizer.adam_beta2 = 0.99
    optimizer.warmup_steps = 0
    optimizer.gradient_skip = False
    optimizer.gradient_skip_threshold = 500.

    return config

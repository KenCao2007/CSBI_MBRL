import ml_collections
import torch


def get_mdp_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 0
  config.train_bs_x_dsm = 8
  config.train_bs_t_dsm = 1
  config.train_bs_x = 3
  config.train_bs_t = 3
  config.num_stage = 20
  config.num_epoch = 6
  config.num_itr = 120
  config.T = 1.0
  # config.train_method = 'dsm_imputation_v2'
  config.train_method = 'mdp_train'
  # config.train_method = "join_train"
  # config.train_method = "alternate_imputation_v2"
  config.t0 = 1e-3
  config.FID_freq = 1
  config.snapshot_freq = 1
  config.ckpt_freq = 1
  config.num_FID_sample = 2000
  config.problem_name = 'mdp'
  config.num_itr_dsm = 10000
  config.backward_warmup_epoch = 12
  config.DSM_warmup = True
  config.log_tb = True

  # sampling
  config.snr = 0.08
  config.samp_bs = 50

  config.interval = 40
  config.sigma_min = 0.001
  config.sigma_max = 20
  config.beta_min = 0.001
  config.beta_max = 20

  # optimization
  # config.optim = optim = ml_collections.ConfigDict()
  config.optimizer = 'AdamW'
  config.lr = 5e-3
  # config.l2_norm = 1e-6
  config.grad_clip = 1.
  config.lr_gamma = 0.99
  config.ema_decay = 0.99


  #env_config
  config.state_dim = 17
  config.action_dim = 6
  config.size = config.state_dim * 2 + config.action_dim + 1
  config.input_size = (1, config.size)
  config.out_size = config.size
  config.in_size = config.input_size
  # config.input_size = 7
  # config.input_size = (1,11)
  # config.out_size = 11
  # config.in_size = (1,11)
  config.permute_batch = True
  config.imputation_eval = True
#   config.output_layer = 'conv1d'  # 'conv1d_silu'

  model_configs={
      'mlp':get_mlp_config(config.input_size, config.out_size),
      'Unetv2':get_Unetv2_config(config.in_size),
      'Transformerv2':get_Transformerv2_config(config.in_size)

  }
  return config, model_configs


def get_Unetv2_config(input_size=None):
  config = ml_collections.ConfigDict()
  config.name = 'Unetv2'
  # config.attention_resolutions='16,8'
  config.attention_layers=[1,2]
  config.in_channels = 1
  config.out_channel = 1
  config.num_head = 2
  config.num_res_blocks = 2
  config.num_channels = 32
  config.num_norm_groups = 32  # num_channels is divisible by num_norm_groups.
  config.dropout = 0.0
  config.channel_mult = (1, 1, 1)  # (1, 1, 2, 2)
  config.input_size = input_size # since we have padding=2
  return config


def get_Transformerv2_config(input_size=None, output_layer='conv1d'):  # Default model.
  config = ml_collections.ConfigDict()
  config.name = 'Transformerv2'
  config.layers = 4
  config.nheads = 8
  config.channels = 64
  config.diffusion_embedding_dim = 128
  config.timeemb = 128
  config.featureemb = 16
  config.input_size = input_size
  config.output_layer = output_layer  # conv1d as default.
  return config


def get_mlp_config(input_size=None, out_size = None):
  config = ml_collections.ConfigDict()
  config.name = "mlp"
  config.name = 'Toyv3'
  config.inputdim = input_size
  config.outputdim = out_size
  config.hiddendim = 64
  return config







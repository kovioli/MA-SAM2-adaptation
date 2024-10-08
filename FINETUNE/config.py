import torch

# settings
DEVICE_NUM = 0
DEVICE = torch.device(f"cuda:{DEVICE_NUM}" if torch.cuda.is_available() else "cpu")

# hyperparams
EPOCH = 80
LR = 1e-5
BS = 1
MODEL_TYPE = 'tiny' # 'tiny', 'small', 'base', 'large'
MODEL_DICT = {
    'tiny': {
        'config': 'sam2_hiera_t.yaml',
        'ckpt': '/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt'
    },
    'small': {
        'config': 'sam2_hiera_s.yaml',
        'ckpt': '/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_small.pt'
    },
    'base': {
        'config': 'sam2_hiera_b+.yaml',
        'ckpt': '/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_base_plus.pt'
    },
    'large': {
        'config': 'sam2_hiera_l.yaml',
        'ckpt': '/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_large.pt'
    }
}

# logging
LOG_EVERY_STEP = 100
PATIENCE = 10
MIN_DELTA = 0.01

# data
THRESHOLD = (0.1, 0.3, 0.5, 0.7, 0.9)
# TRAIN_IDs = ['model_0', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7']
# TRAIN_IDs = ['model_0', 'model_1', 'model_2', 'model_3']
# TRAIN_IDs = ['model_0', 'model_1']
TRAIN_IDs = ['model_0']
VAL_IDs = ['model_8']
NOISE_VAR = 0.0
PROMPT_GRID = False

PARTICLE_CLASS = 13


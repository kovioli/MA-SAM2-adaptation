import torch

# technical
DEVICE_NUM = 2
DEVICE = torch.device(f"cuda:{DEVICE_NUM}" if torch.cuda.is_available() else "cpu")

# hyperparameters
EPOCH = 80
LR = 1e-5
BS = 1

# data
TRAIN_ID = "TS_0001"
VAL_DS_CONF = {
    "ds_id": "TS_0010",
    "s": 64,
    "p": 2,
}
VAL_ID = "TS_0010"
TRAIN_RATIO = 0.8
NR_SLICES = 32
PROMPT_GRID = False

# logging
LOG_EVERY_STEP = 100
PATIENCE = 10
MIN_DELTA = 0.01
THRESHOLD = (0.1, 0.3, 0.5, 0.7, 0.9)

# model
MODEL_TYPE = "tiny"  # 'tiny', 'small', 'base', 'large'
MODEL_DICT = {
    "tiny": {
        "config": "sam2_hiera_t.yaml",
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt",
    },
    "small": {
        "config": "sam2_hiera_s.yaml",
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_small.pt",
    },
    "base": {
        "config": "sam2_hiera_b+.yaml",
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_base_plus.pt",
    },
    "large": {
        "config": "sam2_hiera_l.yaml",
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_large.pt",
    },
}

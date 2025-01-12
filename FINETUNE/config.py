import torch

# technical
DEVICE_NUM = 3  # GPU number
DEVICE = torch.device(f"cuda:{DEVICE_NUM}" if torch.cuda.is_available() else "cpu")

# hyperparameters
EPOCH = 80
LR = 1e-5
BS = 1

# data
DATA_DIR = "/media/hdd1/oliver/EMPIAR_clean"  # path to the data directory
PREDICTION_CLASS = "ribosomes"  # class to predict (based on the folder structure)
TRAIN_ID = "TS_0001"  # training dataset ID
VAL_ID = "TS_0010"  # validation dataset ID
SAVE_DIR = "/media/hdd1/oliver/ASDF"  # checkpoint, log, and prediction save directory

# list of datasets to evaluate
# - name: dataset ID
# - z_offset: offset to apply for particle coordinate matching (if dataset is cut out along the z-axis from the original tomogram)
# - target_shape: target shape for the prediction (the input tomogram shape)
# example:
# EVAL_DS_LIST = [{"name": "TS_0010", "z_offset": 350, "target_shape": (290, 927, 927)}],
# where the original tomogram has been cut as orig_tomo[350:350+290, :927, :927]
EVAL_DS_LIST = [
    {"name": "TS_0010", "z_offset": 350, "target_shape": (290, 927, 927)},
]


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
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt",  # path to the checkpoint
    },
    "small": {
        "config": "sam2_hiera_s.yaml",
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_small.pt",  # path to the checkpoint
    },
    "base": {
        "config": "sam2_hiera_b+.yaml",
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_base_plus.pt",  # path to the checkpoint
    },
    "large": {
        "config": "sam2_hiera_l.yaml",
        "ckpt": "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_large.pt",  # path to the checkpoint
    },
}

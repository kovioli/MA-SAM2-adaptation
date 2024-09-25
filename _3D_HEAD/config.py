import torch

DEVICE_NUM = 1
DEVICE = torch.device(f"cuda:{DEVICE_NUM}" if torch.cuda.is_available() else "cpu")
THRESHOLD = (0.1, 0.3, 0.5, 0.7, 0.9)
EPOCH = 80
LR = 1e-5
BS = 1
# TRAIN_IDs = ['model_0', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7']
TRAIN_IDs = ['model_0', 'model_1']
VAL_IDs = ['model_8']

TRAIN_SLICES = (102, 166)
LOG_EVERY_STEP = 100
PATIENCE = 10
MIN_DELTA = 0.01
TRAIN_RATIO = 0.8
MODEL_TYPE = 'tiny' # 'tiny', 'small', 'base', 'large'
EIGHTH = 8

NOISE_VAR = 1.0
import torch

DEVICE_NUM = 3
DEVICE = torch.device(f"cuda:{DEVICE_NUM}" if torch.cuda.is_available() else "cpu")
THRESHOLD = (0.1, 0.3, 0.5, 0.7, 0.9)
EPOCH = 80
LR = 1e-5
BS = 1
TRAIN_ID = 'TS_0001'
LOG_EVERY_STEP = 100
PATIENCE = 10
MIN_DELTA = 0.01
TRAIN_RATIO = 0.8
MODEL_TYPE = 'tiny' # 'tiny', 'small', 'base', 'large'
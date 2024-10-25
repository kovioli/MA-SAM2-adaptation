# %%
import os
import torch
import numpy as np
import json
from model import SAM2_finetune
from dataset import PNGDataset
import mrcfile
from skimage.transform import resize
import torch.nn.functional as F
from config import DEVICE, PROMPT_GRID, MODEL_TYPE, MODEL_DICT, TRAIN_ID
from torch.cuda.amp import autocast
from matplotlib import pyplot as plt


def compute_dice(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return dice.item()


def compute_iou(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()


training_folder = os.path.join("/media", "hdd1", "oliver", "SAM2_EMPIAR_DCR")

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


training_folder = os.path.join("/media", "hdd1", "oliver", "SAM2_EMPIAR_FINETUNE")
TRAIN_TIMESTAMP = "19102024_07:00"

model_info = MODEL_DICT[MODEL_TYPE]

model = SAM2_finetune(
    model_cfg=model_info["config"],
    ckpt_path=model_info["ckpt"],
    device=DEVICE,
    use_point_grid=PROMPT_GRID,
)

model_path = os.path.join(
    training_folder, "checkpoints", TRAIN_TIMESTAMP, "best_model.pth"
)
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

pred_tomogram_info_list = [
    {"name": "TS_0001", "z_offset": 390, "target_shape": (210, 927, 927)},
    {"name": "TS_0002", "z_offset": 380, "target_shape": (240, 927, 927)},
    {"name": "TS_0003", "z_offset": 380, "target_shape": (250, 927, 927)},
    {"name": "TS_0004", "z_offset": 340, "target_shape": (300, 927, 927)},
    {"name": "TS_0005", "z_offset": 110, "target_shape": (280, 928, 928)},
    {"name": "TS_0006", "z_offset": 170, "target_shape": (140, 928, 928)},
    {"name": "TS_0007", "z_offset": 200, "target_shape": (150, 928, 928)},
    {"name": "TS_0008", "z_offset": 100, "target_shape": (400, 928, 928)},
    {"name": "TS_0009", "z_offset": 120, "target_shape": (250, 928, 928)},
    {"name": "TS_0010", "z_offset": 350, "target_shape": (290, 927, 927)},
]


DS_LIST = [
    "TS_0001",
    "TS_0002",
    "TS_0003",
    "TS_0004",
    "TS_0005",
    "TS_0006",
    "TS_0007",
    "TS_0008",
    "TS_0009",
    "TS_0010",
]

json_results = []

for DS_ID in DS_LIST:
    pred_tomogram_info = [x for x in pred_tomogram_info_list if x["name"] == DS_ID][0]
    if DS_ID == TRAIN_ID:
        print(f"Skipping {DS_ID}")
        continue
    print(f"Predicting {DS_ID}")
    ds = PNGDataset(
        main_folder=os.path.join("/media", "hdd1", "oliver", "EMPIAR_png"),
        DS_ID=DS_ID,
        device=DEVICE,
    )
    all_predictions = []
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for i in range(len(ds)):
            if i % 50 == 0:
                print(f"Slice {i}")
            input_slice, label_slice = ds[i]
            input_slice = input_slice.unsqueeze(0)
            with autocast():
                pred = model(input_slice)
            pred = pred.squeeze(0)
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = pred_sigmoid > 0.99

            dice = compute_dice(pred_binary, label_slice)
            dice_scores.append(dice)

            iou = compute_iou(pred_binary, label_slice)
            iou_scores.append(iou)

            pred_cpu = pred.cpu().numpy().squeeze(0)
            all_predictions.append(pred_cpu)

    average_dice = np.mean(dice_scores)
    average_iou = np.mean(iou_scores)
    print(f"Dice: {average_dice:.3f}, IoU: {average_iou:.3f}")
    all_predictions = np.stack(all_predictions, axis=0)
    resized_pred = resize(
        all_predictions,
        pred_tomogram_info.get("target_shape"),
        mode="reflect",
        anti_aliasing=True,
    )
    pred_save_folder = os.path.join(training_folder, "PREDICT", f"{TRAIN_TIMESTAMP}")
    if not os.path.exists(pred_save_folder):
        os.makedirs(pred_save_folder)

    pred_save_path = os.path.join(pred_save_folder, f"{DS_ID}.mrc")

    with mrcfile.new(pred_save_path, overwrite=True) as mrc:
        mrc.set_data(resized_pred)
    print(f"Prediction saved at {pred_save_path}")
    result = {"DS_ID": DS_ID, "dice": average_dice, "iou": average_iou}
    json_results.append(result)

with open(
    os.path.join(training_folder, "PREDICT", f"{TRAIN_TIMESTAMP}", "seg_results.json"),
    "w",
) as f:
    json.dump(json_results, f)

# %%

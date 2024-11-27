# %%
import os
import torch
import sys
import pandas as pd

sys.path.append("..")
import numpy as np
import json
from _3D_HEAD.model_3D import HeadModel

from FINETUNE.dataset_mrc import MRCDataset
import mrcfile
from skimage.transform import resize
import torch.nn.functional as F
from config import DEVICE, MODEL_DICT
from torch.cuda.amp import autocast
from matplotlib import pyplot as plt

DS_ID = "TS_0001"
NR_SLICES = 8

with open(f".../log_s{NR_SLICES}.csv", "r") as file:
    csv_data = file.read()

# Convert to DataFrame, filtering relevant columns and parsing iou/dice values
df = pd.DataFrame(
    [x.split(",") for x in csv_data.splitlines()],
    columns=["timestamp", "size", "series", "s32", "position", "run", "iou", "dice"],
)
best_idx = df.groupby("position")["dice"].idxmax()

# Create new dataframe with only the best runs
best_runs = df.loc[best_idx]


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


training_folder = os.path.join("...")


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


for idx, row in best_runs.iterrows():
    timestamp = row["timestamp"]
    position = row["position"]
    model_size = row["size"]
    print(f"EVALUATING POSITION {position} WITH TIMESTAMP {timestamp}")
    model_info = MODEL_DICT[model_size]

    model = HeadModel(model_type=model_size, device=DEVICE)
    model_path = os.path.join(
        training_folder, "checkpoints", timestamp, "best_model.pth"
    )
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    json_results = []
    for DS in pred_tomogram_info_list:
        if DS["name"] == DS_ID:
            # skip training DS
            continue
        print(f"EVALUATING TOMOGRAM {DS['name']}")
        ds = MRCDataset(
            main_folder=".../EMPIAR_clean",
            ds_id=DS["name"],
            device=DEVICE,
            full_tomo=True,
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
                    pred, denoised_img, _ = model(input_slice)
                pred = pred.squeeze(0)
                pred_sigmoid = torch.sigmoid(pred)
                pred_binary = pred_sigmoid > 0.5

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
            DS.get("target_shape"),
            mode="reflect",
            anti_aliasing=True,
        )
        pred_save_folder = os.path.join(training_folder, "PREDICT", f"{timestamp}")
        if not os.path.exists(pred_save_folder):
            os.makedirs(pred_save_folder)

        pred_save_path = os.path.join(pred_save_folder, f"{DS['name']}.mrc")

        with mrcfile.new(pred_save_path, overwrite=True) as mrc:
            mrc.set_data(resized_pred)
        print(f"Prediction saved at {pred_save_path}")
        result = {"DS_ID": DS_ID, "dice": average_dice, "iou": average_iou}
        json_results.append(result)

    with open(
        os.path.join(training_folder, "PREDICT", f"{timestamp}", "seg_results.json"),
        "w",
    ) as f:
        json.dump(json_results, f)


# %%

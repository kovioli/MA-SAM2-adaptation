# %%
import os
import torch
import numpy as np
import json
from _3D_HEAD.model_3D import HeadFinetuneModel
from shrec_dataset import MRCDataset
import mrcfile
from skimage.transform import resize
import torch.nn.functional as F
from config import DEVICE, PROMPT_GRID, MODEL_TYPE, MODEL_DICT
from torch.cuda.amp import autocast


# Functions to compute Dice coefficient and IoU
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


particle_ts_mapping = {
    "grandmodel_8ds_large": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "31102024_17:49"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "31102024_22:47"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "01112024_03:26"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "01112024_08:15"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "01112024_11:53"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "01112024_15:50"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "01112024_22:46"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "02112024_02:54"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "02112024_06:23"},
        {"particle_id": 10, "particle_name": "3d2f", "ts": "02112024_11:33"},
        {"particle_id": 11, "particle_name": "4d8q", "ts": "02112024_17:14"},
        {"particle_id": 12, "particle_name": "1bxn", "ts": "02112024_20:24"},
    ],
    "reconstruction_8ds_large": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "31102024_18:40:07"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "31102024_21:51:00"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "31102024_23:35:14"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "01112024_01:29:46"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "01112024_03:43:32"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "01112024_05:38:12"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "01112024_08:01:39"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "01112024_09:46:33"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "01112024_12:09:56"},
        {"particle_id": 10, "particle_name": "3d2f", "ts": "01112024_14:23:48"},
        {"particle_id": 11, "particle_name": "4d8q", "ts": "01112024_16:27:55"},
        {"particle_id": 12, "particle_name": "1bxn", "ts": "01112024_20:46:32"},
    ],
    "reconstruction_1ds_large": [
        {"particle_id": 12, "particle_name": "1bxn", "ts": "22112024_14:37:17"},
    ],
    "reconstruction_half_ds_large": [
        {"particle_id": 12, "particle_name": "1bxn", "ts": "22112024_17:15:52"},
    ],
    "reconstruction_s64_large": [
        {"particle_id": 12, "particle_name": "1bxn", "ts": "22112024_21:51:49"},
    ],
    "reconstruction_1ds_final_large": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "22112024_22:18:19"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "22112024_22:46:29"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "22112024_23:09:14"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "22112024_23:31:06"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "23112024_00:05:01"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "23112024_00:28:59"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "23112024_00:50:49"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "23112024_01:12:42"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "23112024_01:34:34"},
        {"particle_id": 10, "particle_name": "3d2f", "ts": "23112024_01:58:37"},
        {"particle_id": 11, "particle_name": "4d8q", "ts": "23112024_02:20:29"},
        {"particle_id": 12, "particle_name": "1bxn", "ts": "23112024_02:52:32"},
    ],
}
test_ds_name = "model_9"


# Initialize the model
model_info = MODEL_DICT[MODEL_TYPE]

model = HeadFinetuneModel(model_type=MODEL_TYPE, device=DEVICE)


run_id = "reconstruction_1ds_final_large"
base_folder = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "SAM2_SHREC_HEADFINETUNE",
    f"shrec2020_headfinetune_ce_{run_id}",
)

all_results = []
for particle_training in particle_ts_mapping.get(run_id):
    print(
        f"Processing particle {particle_training.get('particle_id')} with TS {particle_training.get('ts')}"
    )

    model_path = os.path.join(
        base_folder,
        "checkpoints",
        particle_training.get("ts"),
        "best_model.pth",  # Update the model filename if different
    )
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    ds = MRCDataset(
        main_folder="/media/hdd1/oliver/shrec2020_full_dataset",
        DS_ID=test_ds_name,
        device=DEVICE,
        particle_id=particle_training.get("particle_id"),
        input_type="reconstruction",
    )

    all_predictions = []
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for i in range(len(ds)):
            # if i % 20 == 0:
            #     print(f"Processing slice {i}/{len(ds)}")
            input_slice, label_slice = ds[i]  # Get input and label
            input_slice = input_slice.unsqueeze(0).to(
                DEVICE
            )  # Add batch dimension and move to device
            label_slice = label_slice.to(DEVICE)

            # Make prediction
            with autocast():
                pred, _ = model(input_slice)
            pred = pred.squeeze(0)  # Remove batch dimension

            # Apply sigmoid
            pred_sigmoid = torch.sigmoid(pred)

            # Threshold at 0.5 to get binary mask
            pred_binary = (pred_sigmoid > 0.99).float()

            # Compute Dice coefficient
            dice = compute_dice(pred_binary, label_slice)
            dice_scores.append(dice)

            # Compute IoU
            iou = compute_iou(pred_binary, label_slice)
            iou_scores.append(iou)

            # Collect predictions for saving
            pred_cpu = pred_binary.cpu().numpy().squeeze()  # Shape: (H, W)
            all_predictions.append(pred_cpu)

    # Calculate average metrics
    average_dice = np.mean(dice_scores)
    average_iou = np.mean(iou_scores)
    print(f"Dice / IoU:\n{average_dice:.3f} / {average_iou:.3f}")
    all_predictions = np.stack(all_predictions, axis=0)
    resized_pred = resize(
        all_predictions, (len(ds), 512, 512), mode="reflect", anti_aliasing=True
    )
    resized_pred = resized_pred.astype(np.float32)
    if not os.path.exists(os.path.join(base_folder, "PREDICT")):
        os.makedirs(os.path.join(base_folder, "PREDICT"))
    pred_save_path = os.path.join(
        base_folder,
        "PREDICT",
        f"{test_ds_name}_particle_{particle_training.get('particle_id')}.mrc",
    )
    with mrcfile.new(pred_save_path, overwrite=True) as mrc:
        mrc.set_data(resized_pred)
        print(f"Saved prediction to {pred_save_path}")
    result = {
        "particle_id": particle_training.get("particle_id"),
        "particle_name": particle_training.get("particle_name"),
        "ts": particle_training.get("ts"),
        "dice": average_dice,
        "iou": average_iou,
    }
    all_results.append(result)

json_result_path = os.path.join(base_folder, f"class_exploration_seg_results.json")
with open(json_result_path, "w") as f:
    json.dump(all_results, f)

# %%

# %%
import os
import torch
import numpy as np
import json
from model import SAM2_finetune
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


particle_ts_mapping_grandmodel = {
    "2ds": [
        {"particle_id": 1, "ts": "08102024_06:23", "particle_name": "3cf3"},
        {"particle_id": 2, "ts": "08102024_07:00", "particle_name": "1s3x"},
        {"particle_id": 3, "ts": "08102024_07:15", "particle_name": "1u6g"},
        {"particle_id": 4, "ts": "08102024_07:33", "particle_name": "4cr2"},
        {"particle_id": 5, "ts": "08102024_07:50", "particle_name": "1qvr"},
        {"particle_id": 6, "ts": "08102024_08:09", "particle_name": "3h84"},
        {"particle_id": 7, "ts": "08102024_08:26", "particle_name": "2cg9"},
        {"particle_id": 8, "ts": "08102024_08:40", "particle_name": "3qm1"},
        {"particle_id": 9, "ts": "08102024_08:53", "particle_name": "3gl1"},
        {"particle_id": 10, "ts": "08102024_09:13", "particle_name": "3d2f"},
        {"particle_id": 11, "ts": "08102024_09:30", "particle_name": "4d8q"},
        {"particle_id": 12, "ts": "08102024_10:08", "particle_name": "1bxn"},
    ],
    "8ds": [
        {"particle_id": 1, "ts": "08102024_06:26", "particle_name": "3cf3"},
        {"particle_id": 2, "ts": "08102024_07:51", "particle_name": "1s3x"},
        {"particle_id": 3, "ts": "08102024_08:58", "particle_name": "1u6g"},
        {"particle_id": 4, "ts": "08102024_10:08", "particle_name": "4cr2"},
        {"particle_id": 5, "ts": "08102024_11:45", "particle_name": "1qvr"},
        {"particle_id": 6, "ts": "08102024_13:25", "particle_name": "3h84"},
        {"particle_id": 7, "ts": "08102024_14:49", "particle_name": "2cg9"},
        {"particle_id": 8, "ts": "08102024_15:56", "particle_name": "3qm1"},
        {"particle_id": 9, "ts": "08102024_16:58", "particle_name": "3gl1"},
        {"particle_id": 10, "ts": "08102024_18:24", "particle_name": "3d2f"},
        {"particle_id": 11, "ts": "08102024_19:47", "particle_name": "4d8q"},
        {"particle_id": 12, "ts": "08102024_20:55", "particle_name": "1bxn"},
    ],
}


particle_ts_mapping_reconstruction = {
    "8ds_tiny_OLD": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "16102024_16:26"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "16102024_17:16"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "16102024_17:59"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "16102024_19:15"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "16102024_20:06"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "16102024_20:52"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "16102024_21:36"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "16102024_22:20"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "16102024_23:43"},
        {"particle_id": 10, "particle_name": "3d2f", "ts": "17102024_00:26"},
        {"particle_id": 11, "particle_name": "4d8q", "ts": "17102024_01:15"},
        {"particle_id": 12, "particle_name": "1bxn", "ts": "17102024_02:30"},
    ],
    "8ds_tiny": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "21102024_19:46"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "21102024_20:32"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "21102024_21:43"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "21102024_22:17"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "21102024_23:08"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "21102024_23:48"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "22102024_00:40"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "22102024_01:27"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "22102024_02:18"},
        {"particle_id": 10, "particle_name": "3d2f", "ts": "22102024_03:19"},
        {"particle_id": 11, "particle_name": "4d8q", "ts": "22102024_04:17"},
        {"particle_id": 12, "particle_name": "1bxn", "ts": "22102024_05:13"},
    ],
    "8ds_large": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "16102024_16:25"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "16102024_18:15"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "16102024_19:51"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "16102024_21:26"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "16102024_23:10"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "17102024_00:44"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "17102024_02:30"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "17102024_04:17"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "17102024_06:38"},
        {"particle_id": 10, "particle_name": "3d2f", "ts": "17102024_08:41"},
        {"particle_id": 11, "particle_name": "4d8q", "ts": "17102024_10:41"},
        {"particle_id": 12, "particle_name": "1bxn", "ts": "17102024_13:57"},
    ],
    "denoised_8ds_tiny": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "17102024_09:04"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "17102024_10:02"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "17102024_10:52"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "17102024_11:32"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "17102024_12:13"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "17102024_12:58"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "17102024_13:39"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "17102024_14:20"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "17102024_15:07"},
        {"particle_id": 10, "particle_name": "3d2f", "ts": "17102024_15:50"},
        {"particle_id": 11, "particle_name": "4d8q", "ts": "17102024_16:31"},
        {"particle_id": 12, "particle_name": "1bxn", "ts": "17102024_17:29"},
    ],
    "denoised_g1_8ds_tiny": [
        {"particle_id": 1, "particle_name": "3cf3", "ts": "18102024_13:35"},
        {"particle_id": 2, "particle_name": "1s3x", "ts": "18102024_14:20"},
        {"particle_id": 3, "particle_name": "1u6g", "ts": "18102024_15:18"},
        {"particle_id": 4, "particle_name": "4cr2", "ts": "18102024_15:58"},
        {"particle_id": 5, "particle_name": "1qvr", "ts": "18102024_16:38"},
        {"particle_id": 6, "particle_name": "3h84", "ts": "18102024_17:19"},
        {"particle_id": 7, "particle_name": "2cg9", "ts": "18102024_18:03"},
        {"particle_id": 8, "particle_name": "3qm1", "ts": "18102024_18:48"},
        {"particle_id": 9, "particle_name": "3gl1", "ts": "18102024_20:02"},
        {"particle_id": 10, "particle_nName": "3d2f", "ts": "18102024_20:47"},
        {"particle_id": 11, "particle_nName": "4d8q", "ts": "18102024_21:34"},
        {"particle_id": 12, "particle_nName": "1bxn", "ts": "18102024_22:56"},
    ],
}
test_ds_name = "model_9"


# Initialize the model
model_info = MODEL_DICT[MODEL_TYPE]

model = SAM2_finetune(
    model_cfg=model_info["config"],
    ckpt_path=model_info["ckpt"],
    device=DEVICE,
    use_point_grid=PROMPT_GRID,
)


run_id = "8ds_tiny"
base_folder = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "SAM2_SHREC_FINETUNE",
    f"shrec2020_finetune_class_exploration_reconstruction_{run_id}",
)

all_results = []
for particle_training in particle_ts_mapping_reconstruction.get(run_id):
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
                pred = model(input_slice)
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

json_result_path = os.path.join(base_folder, f"class_exploration_results.json")
with open(json_result_path, "w") as f:
    json.dump(all_results, f)

# %%

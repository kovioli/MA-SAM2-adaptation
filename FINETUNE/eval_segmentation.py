import os
import torch
import numpy as np
import json
from _3D_HEAD.model_3D import HeadFinetuneModel
from dataset import MRCDataset
import mrcfile
from skimage.transform import resize
import argparse
from config import (
    DEVICE,
    MODEL_TYPE,
    SAVE_DIR,
    DATA_DIR,
    EVAL_DS_LIST,
    PREDICTION_CLASS,
)
from torch.cuda.amp import autocast


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


def main(train_timestamp):
    model = HeadFinetuneModel(
        model_type=MODEL_TYPE,
        device=DEVICE,
    )

    model_path = os.path.join(
        SAVE_DIR, "checkpoints", train_timestamp, "best_model.pth"
    )
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    json_results = []

    for DS_INFO in EVAL_DS_LIST:
        ds_id = DS_INFO["name"]
        target_shape = DS_INFO["target_shape"]

        print(f"Predicting {ds_id}")
        ds = MRCDataset(
            data_folder=DATA_DIR,
            pred_class=PREDICTION_CLASS,
            ds_id=ds_id,
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
                    pred, _ = model(input_slice)
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
            target_shape,
            mode="reflect",
            anti_aliasing=True,
        )
        pred_save_folder = os.path.join(SAVE_DIR, "PREDICT", train_timestamp)
        if not os.path.exists(pred_save_folder):
            os.makedirs(pred_save_folder)

        pred_save_path = os.path.join(pred_save_folder, f"{ds_id}.mrc")

        with mrcfile.new(pred_save_path, overwrite=True) as mrc:
            mrc.set_data(resized_pred)
        print(f"Prediction saved at {pred_save_path}")
        result = {"DS_ID": ds_id, "dice": average_dice, "iou": average_iou}
        json_results.append(result)

    with open(
        os.path.join(SAVE_DIR, "PREDICT", train_timestamp, "seg_results.json"),
        "w",
    ) as f:
        json.dump(json_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_timestamp",
        type=str,
        required=True,
        help="Timestamp of the training run (format: DDMMYYYY_HH:MM:SS)",
    )
    args = parser.parse_args()
    main(args.train_timestamp)

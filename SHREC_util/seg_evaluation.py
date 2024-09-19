import os
import torch
import numpy as np
import sys
import json
sys.path.append('..')
from _3D_HEAD.model_3D import HeadModel
from _3D_HEAD.shrec_dataset import MRCDataset
import mrcfile
from skimage.transform import resize
import torch.nn.functional as F

# Set the device
DEVICE = 'cuda:3'  # Change as needed
ds_list = ['model_0', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9']

json_result_folder = os.path.join(
    '/oliver',
    'SAM2',
    'SHREC_segmentation_results'
)

# Initialize the model
model = HeadModel('tiny', DEVICE, in_chan=3)
TS = '18092024_13:48'
model_path = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'SAM2_SHREC',
    'checkpoints',
    TS,
    'best_model.pth'  # Update the model filename if different
)
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

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

# List to store results for all datasets
all_results = []

for DS_ID in ds_list:
    print(f"Processing dataset: {DS_ID}")
    
    # Create the dataset
    ds = MRCDataset(main_folder='/media/hdd1/oliver/SHREC', DS_ID=DS_ID, device=DEVICE)

    # Prepare to collect predictions and metrics
    all_predictions = []
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for i in range(len(ds)):
            if i % 20 == 0:
                print(f"Processing slice {i}/{len(ds)}")
            input_slice, label_slice = ds[i]  # Get input and label
            input_slice = input_slice.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
            label_slice = label_slice.to(DEVICE)

            # Make prediction
            pred, _ = model(input_slice)
            pred = pred.squeeze(0)  # Remove batch dimension

            # Apply sigmoid
            pred_sigmoid = torch.sigmoid(pred)

            # Threshold at 0.5 to get binary mask
            pred_binary = (pred_sigmoid > 0.5).float()

            # Compute Dice coefficient
            dice = compute_dice(pred_binary, label_slice)
            dice_scores.append(dice)

            # Compute IoU
            iou = compute_iou(pred_binary, label_slice)
            iou_scores.append(iou)

            # Collect predictions for saving
            pred_cpu = pred.cpu().numpy().squeeze()  # Shape: (H, W)
            all_predictions.append(pred_cpu)

    # Calculate average metrics
    average_dice = np.mean(dice_scores)
    average_iou = np.mean(iou_scores)

    print(f"Average Dice coefficient for {DS_ID}: {average_dice}")
    print(f"Average IoU for {DS_ID}: {average_iou}")

    # Stack all predictions along the depth axis
    all_predictions = np.stack(all_predictions, axis=0)  # Shape: (D, H, W)

    # Resize predictions to target shape (180, 512, 512)
    print("Resizing predictions to target shape (180, 512, 512)...")
    target_shape = (180, 512, 512)
    resized_pred = resize(all_predictions, target_shape, mode='reflect', anti_aliasing=True)

    # Ensure the data type is float32 for saving
    resized_pred = resized_pred.astype(np.float32)

    # Save predictions to MRC file
    pred_save_path = os.path.join(
        '/media',
        'hdd1',
        'oliver',
        'SAM2_SHREC',
        'PREDICT',
        TS,
        f'{DS_ID}_prediction.mrc'
    )
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)

    with mrcfile.new(pred_save_path, overwrite=True) as mrc:
        mrc.set_data(resized_pred)
        print(f"Predictions saved to {pred_save_path}")

    # Store results for this dataset
    result = {
        "ds_name": DS_ID,
        "dice": average_dice,
        "IoU": average_iou
    }
    all_results.append(result)

# Save all results to a JSON file
json_file_path = os.path.join(json_result_folder, f'segmentation_results_{TS}.json')
os.makedirs(json_result_folder, exist_ok=True)

with open(json_file_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Results saved to {json_file_path}")
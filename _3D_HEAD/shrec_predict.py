# %%
import os
import torch
import numpy as np
from model_3D import HeadModel
from shrec_dataset import MRCDataset
import mrcfile
from skimage.transform import resize
import torch.nn.functional as F

# Set the device
DEVICE = 'cuda:0'  # Change as needed

# Initialize the model
model = HeadModel('tiny', DEVICE, in_chan=3)
TS = '18092024_09:37'
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

# Specify the dataset ID (e.g., 'model_0' to 'model_9')
DS_ID = 'model_1'  # Change to the desired dataset

# Create the dataset
ds = MRCDataset(main_folder='/media/hdd1/oliver/SHREC', DS_ID=DS_ID, device=DEVICE)

# Prepare to collect predictions
all_predictions = []

with torch.no_grad():
    for i in range(len(ds)):
        if i % 20 == 0:
            print(f"Processing slice {i}/{len(ds)}")
        input_slice, _ = ds[i]  # We don't need labels for prediction
        input_slice = input_slice.unsqueeze(0)  # Add batch dimension
        input_slice = input_slice.to(DEVICE)

        # Make prediction
        pred, _ = model(input_slice)

        # Detach and move to CPU
        pred_cpu = pred.cpu().numpy().squeeze()  # Shape: (1, H, W) -> (H, W)
        all_predictions.append(pred_cpu)

# %%
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
    f'{DS_ID}_prediction.mrc'
)
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)

with mrcfile.new(pred_save_path, overwrite=True) as mrc:
    mrc.set_data(resized_pred)
    print(f"Predictions saved to {pred_save_path}")

# %%

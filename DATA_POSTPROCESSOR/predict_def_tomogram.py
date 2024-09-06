# %%
import mrcfile
import os
import numpy as np
import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from _3D_HEAD.model_3D import HeadModel
from _3D_HEAD.config import DEVICE
import matplotlib.pyplot as plt

def read_mrc(file_path: str):
    with mrcfile.open(file_path, mode='r', permissive=True) as mrc:
        return mrc.data
    
def load_model_from_path(model_path: str):
    model = HeadModel('tiny', DEVICE, in_chan=3)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model
# %%

tomo_path = os.path.join(
    '/oliver',
    'EMPIAR_DEF',
    'tomograms',
    'TS_026.rec'
)

tomogram_data = read_mrc(tomo_path)
# rib = read_mrc(os.path.join('/oliver', 'EMPIAR_DEF', 'ribosomes', 'TS_026_cyto_ribosomes.mrc'))
# %%
PREDICTION_DIR = os.path.join(
    '/oliver',
    'SAM2',
    'PREDICT',
    'TS_29082024_15:06'
)
pred_save_path = os.path.join(PREDICTION_DIR, 'TS_026.mrc')
model_path = os.path.join(
    '/oliver',
    'SAM2',
    'checkpoints',
    '29082024_15:06',
    'best_model.pth'
)
model = load_model_from_path(model_path)
# %%
all_predictions = []
with torch.no_grad():
    for i in range(430, 550):
        if i % 20 == 0:
            print(f"Slice {i}")
        
        # Extract the slice and convert to tensor
        slice_data = torch.tensor(tomogram_data[i]).float().to(DEVICE)
        
        # Stack the slice to simulate 3-channel RGB image
        input_image = slice_data.unsqueeze(0).repeat(3, 1, 1)  # [3, 928, 960]
        
        # Pad the image to [3, 1024, 1024]
        # Only padding bottom and right sides
        padding = (0, 1024 - 960, 0, 1024 - 928)  # (left, right, top, bottom)
        input_image_padded = F.pad(input_image, padding, "constant", 0)
        
        # Prepare for model input (add batch dimensions)
        model_input = input_image_padded.unsqueeze(0)  # [1, 3, 1024, 1024]
        
        # Get prediction
        pred, _ = model(model_input)
        
        # Store the prediction
        all_predictions.append(pred.squeeze().cpu().numpy())

# %%
pred_tensor = torch.tensor(all_predictions)
# %%

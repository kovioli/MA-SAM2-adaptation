# %%
import os
import torch
import numpy as np
from model_3D import HeadModel
from dataset import PNGDataset
import matplotlib.pyplot as plt
import mrcfile
from skimage.transform import resize
# %%

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

TS = "11092024_08:33"
DEVICE = 'cuda:3'
model = HeadModel('base', DEVICE, in_chan=3)
model_path = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'SAM2',
    'checkpoints',
    TS,
    'best_model.pth'
)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

# %%
PRED_ID = 'TS_0002'
pred_tomogram_info = [x for x in pred_tomogram_info_list if x['name'] == PRED_ID][0]
ds = PNGDataset(main_folder='/media/hdd1/oliver/EMPIAR_png', DS_ID=PRED_ID, device=DEVICE)
# %%
all_predictions = []
with torch.no_grad():
    for i in range(len(ds)):
        if i % 20 == 0:
            print(f"Slice {i}")
        # memory = None # for single slice training
        pred, _ = model(ds[i][0].unsqueeze(0))
        all_predictions.append(pred.cpu().numpy().squeeze(0))
all_predictions = np.concatenate(all_predictions, axis=0)
print("resizing...")
resized_pred = resize(all_predictions, pred_tomogram_info.get('target_shape'), mode='reflect', anti_aliasing=True)

pred_save_path = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'SAM2',
    'PREDICT',
    f'TS_{TS}',
    f'{PRED_ID}.mrc'
)
if not os.path.exists(os.path.dirname(pred_save_path)):
    os.makedirs(os.path.dirname(pred_save_path))
with mrcfile.new(pred_save_path, overwrite=True) as mrc:
    mrc.set_data(resized_pred)
# %%
plt.imshow(ds[100][1][0].cpu())
plt.axis('off')
# %%
plt.imshow(ds[100][0][0].cpu())
plt.axis('off')
# %%
with torch.no_grad():
    asdf = model(ds[100][0].unsqueeze(0))
asdf_norm = (asdf - asdf.min()) / (asdf.max() - asdf.min())
plt.imshow(asdf_norm.cpu().squeeze() > 0.7)
plt.axis('off')
# %%

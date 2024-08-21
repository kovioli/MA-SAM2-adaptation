# %%
import os
import torch
import numpy as np
from model_3D import HeadModel
from dataset import PNGDataset
import matplotlib.pyplot as plt
import mrcfile
from skimage.transform import resize


TS = "21082024_13:58" # tiny
DEVICE = 'cuda:1'
PRED_ID = 'TS_0002'
model = HeadModel('tiny', DEVICE, in_chan=3)

ds = PNGDataset(main_folder='/oliver/EMPIAR_png', DS_ID='TS_0002', device=DEVICE)
# %%
model_path = os.path.join(
    '/oliver',
    'SAM2',
    'checkpoints',
    TS,
    'best_model.pth'
)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
# %%
all_predictions = []
memory = None
with torch.no_grad():
    for i in range(len(ds)):
        if i % 20 == 0:
            print(f"Slice {i}")
        pred, memory = model(ds[i][0].unsqueeze(0), memory)
        all_predictions.append(pred.cpu().numpy().squeeze(0))
all_predictions = np.concatenate(all_predictions, axis=0)
resized_pred = resize(all_predictions, (240, 927, 927), mode='reflect', anti_aliasing=True)

pred_save_path = os.path.join(
    '/oliver',
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

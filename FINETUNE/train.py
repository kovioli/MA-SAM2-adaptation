# %%
import os
import torch
from monai.losses import GeneralizedDiceLoss
from shrec_dataset import MRCDataset
from model import SAM2_finetune
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    EPOCH,
    LR,
    BS,
    MODEL_TYPE,
    LOG_EVERY_STEP,
    PATIENCE,
    MIN_DELTA,
    THRESHOLD,
    TRAIN_IDs,
    VAL_IDs,
    MODEL_DICT
)



ds = MRCDataset(
    main_folder=os.path.join('/media', 'hdd1', 'oliver', 'SHREC'),
    DS_ID=TRAIN_IDs[0],
    device=DEVICE
)
lossfunc = GeneralizedDiceLoss(sigmoid=True, reduction='mean')
model = SAM2_finetune(
    model_cfg=MODEL_DICT[MODEL_TYPE]['config'],
    ckpt_path=MODEL_DICT[MODEL_TYPE]['ckpt'],
    device=DEVICE
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.eval()


# %%
# plt.imshow(ds[0][0][0].cpu())
pred = model(ds[120][0].unsqueeze(0))
plt.imshow(pred.squeeze().cpu().detach().numpy())
# %%

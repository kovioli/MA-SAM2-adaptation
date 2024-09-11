# %%
import os
import mrcfile
import matplotlib.pyplot as plt

pred_timestamp = '11092024_14:40'
DS_NAME = 'TS_0001'
PRED_PATH = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'SAM2',
    'PREDICT',
    f"TS_{pred_timestamp}",
    f"{DS_NAME}.mrc"   
)

TOMOGRAM_PATH = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'EMPIAR_clean',
    'tomograms',
    f"{DS_NAME}.mrc"
)

FAS_PATH = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'EMPIAR_clean',
    'fas',
    f"{DS_NAME}.mrc"
)

with mrcfile.open(PRED_PATH) as mrc, \
    mrcfile.open(TOMOGRAM_PATH) as mrc_tomo, \
    mrcfile.open(FAS_PATH) as mrc_fas:
        
    pred = mrc.data
    tomo = mrc_tomo.data
    fas = mrc_fas.data
    print(f"PRED SHAPE: {pred.shape} | TOMO SHAPE: {tomo.shape} | FAS SHAPE: {fas.shape}")
# %%
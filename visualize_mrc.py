# %%
import os
import mrcfile
import matplotlib.pyplot as plt

pred_timestamp = "19102024_07:00"
DS_NAME = "TS_0002"
PRED_PATH = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "SAM2_EMPIAR_FINETUNE",
    "PREDICT",
    f"{pred_timestamp}",
    f"{DS_NAME}.mrc",
)

TOMOGRAM_PATH = os.path.join(
    "/media", "hdd1", "oliver", "EMPIAR_clean", "tomograms", f"{DS_NAME}.mrc"
)

FAS_PATH = os.path.join(
    "/media", "hdd1", "oliver", "EMPIAR_clean", "ribosomes", f"{DS_NAME}.mrc"
)

with mrcfile.open(PRED_PATH) as mrc, mrcfile.open(TOMOGRAM_PATH) as mrc_tomo:

    pred = mrc.data
    tomo = mrc_tomo.data
    print(f"PRED SHAPE: {pred.shape} | TOMO SHAPE: {tomo.shape}")
# %%

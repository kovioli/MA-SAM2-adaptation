# %%
import os
import mrcfile
import matplotlib.pyplot as plt

pred_timestamp = "24102024_21:20:21"
DS_NAME = "TS_0002"
PRED_PATH = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "SAM2_EMPIAR_DCR",
    "PREDICT",
    f"{pred_timestamp}",
    f"{DS_NAME}.mrc",
)


with mrcfile.open(PRED_PATH) as mrc:
    pred = mrc.data
    print(f"PRED SHAPE: {pred.shape}")

# PRED_PATH = os.path.join(
#     "/media",
#     "hdd1",
#     "oliver",
#     "DeePiCt",
#     "PREDICT",
#     "predictions",
#     "test_holdout_1_best",
#     "TS_0002",
#     "ribo",
#     "probability_map.mrc",
# )
# %%


TOMOGRAM_PATH = os.path.join("/media", "hdd1", "oliver", "EMPIAR", f"{DS_NAME}.rec")

RIBO_PATH = os.path.join(
    "/media", "hdd1", "oliver", "EMPIAR", f"{DS_NAME}_cyto_ribosomes.mrc"
)

with mrcfile.open(RIBO_PATH) as mrc, mrcfile.open(TOMOGRAM_PATH) as mrc_tomo:

    ribo = mrc.data
    tomo = mrc_tomo.data
    print(f"RIBO SHAPE: {ribo.shape} | TOMO SHAPE: {tomo.shape}")
# %%

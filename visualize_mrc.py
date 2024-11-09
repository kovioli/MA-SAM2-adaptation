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

MODEL_NAME = "TS_0001_s8-64_p0_r2"
model_name = f"{MODEL_NAME}_best"

# TS_0001_s16_p10_r1_best
# TS_0001_s16_p4_r4_best
# TS_0001_s16_p6_r2_best
# TS_0001_s16_p8_r3_best
deepict_pred_folder = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "DeePiCt",
    "PREDICT",
    "predictions",
    model_name,
    "TS_0002",
    "ribo",
    "probability_map.mrc",
)


with mrcfile.open(deepict_pred_folder) as mrc:
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

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


TOMOGRAM_PATH = os.path.join(
    "/media", "hdd1", "oliver", "EMPIAR_DCR", "TS_0001", "16-64", "0", "tomogram.mrc"
)

RIBO_PATH = os.path.join(
    "/media", "hdd1", "oliver", "EMPIAR_DCR", "TS_0001", "16-64", "0", "ribosome.mrc"
)

with mrcfile.open(RIBO_PATH) as mrc, mrcfile.open(TOMOGRAM_PATH) as mrc_tomo:

    ribo = mrc.data
    tomo = mrc_tomo.data
    print(f"RIBO SHAPE: {ribo.shape} | TOMO SHAPE: {tomo.shape}")
# %%

# %%
import os
import mrcfile
import numpy as np

position = 0
ribo_path = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "EMPIAR_DCR",
    "TS_0001",
    "8-64",
    str(position),
    "ribosome.mrc",
)
with mrcfile.open(ribo_path) as mrc:
    ribo = mrc.data
    print(f"RIBO SHAPE: {ribo.shape}")

print(f"Total voxels: {ribo.size}")
print(f"Total ribo voxels: {int(np.sum(ribo))}")

# %%
# SHREC
import os
import mrcfile
import numpy as np
import matplotlib.pyplot as plt

model_nr = 0
class_mask = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "shrec2020_full_dataset",
    f"model_{model_nr}",
    "class_mask.mrc",
)

grandmodel_path = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "shrec2020_full_dataset",
    f"model_{model_nr}",
    "grandmodel.mrc",
)

reconstruction_path = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "shrec2020_full_dataset",
    f"model_{model_nr}",
    "reconstruction.mrc",
)
with mrcfile.open(class_mask, permissive=True) as mrc:
    class_mask = mrc.data
    print(f"Class mask SHAPE: {class_mask.shape}")

with mrcfile.open(grandmodel_path, permissive=True) as mrc:
    grandmodel = mrc.data
    print(f"Grandmodel SHAPE: {grandmodel.shape}")

with mrcfile.open(reconstruction_path, permissive=True) as mrc:
    reconstruction = mrc.data
    reconstruction = reconstruction[156:356]
    print(f"Reconstruction SHAPE: {reconstruction.shape}")

# %%

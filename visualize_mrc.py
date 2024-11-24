# %%
import os
import mrcfile
import matplotlib.pyplot as plt
import struct
import numpy as np

# TOMOGRAM_PATH = os.path.join(
#     "/media",
#     "hdd1",
#     "oliver",
#     "shrec2020_full_dataset",
#     "model_0",
#     f"reconstruction.mrc",
# )

# TOMOGRAM_PATH = os.path.join(
#     "/media",
#     "hdd1",
#     "oliver",
#     "SAM2_SHREC_FINETUNE",
#     "shrec2020_finetune_class_exploration_reconstruction_8ds_tiny",
#     "PREDICT",
#     "model_9_particle_1.mrc",
# )

# prediction_dir=f"/media/hdd1/oliver/DeePiCt/PREDICT/predictions/shrec_p{p_map['particle_id']:02d}_reconstruction_best/model_9_p{p_map['particle_id']}_reconstruction/ribo/",

# TOMOGRAM_PATH = os.path.join(
#     "/media",
#     "hdd1",
#     "oliver",
#     "DeePiCt",
#     "PREDICT",
#     "predictions",
#     "shrec_p01_reconstruction_best",
#     "model_9_p1_reconstruction",
#     "ribo",
#     "probability_map.mrc",
# )
DEEPICT_PRED_PATH = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "DeePiCt",
    "PREDICT",
    "predictions",
    "TS_0001_s8-64_p1_r4_best",
    "TS_0002",
    "ribo",
    "probability_map.mrc",
)

INPUT_PATH = os.path.join(
    "/media", "hdd1", "oliver", "EMPIAR_clean", "tomograms", "TS_0002.mrc"
)
LABEL_PATH = os.path.join(
    "/media", "hdd1", "oliver", "EMPIAR_clean", "ribosomes", "TS_0002.mrc"
)

# TOMOGRAM_PATH = os.path.join(
#     "/media",
#     "hdd1",
#     "oliver",
#     "DEEPICT_SHREC",
#     "model_0",
#     "reconstruction_cropped.mrc",
# )


def read_mrc(file_path):
    with open(file_path, "rb") as f:
        # Read the header (1024 bytes)
        header = f.read(1024)
        nx, ny, nz = struct.unpack("3i", header[0:12])
        mode = struct.unpack("i", header[12:16])[0]

        # Determine the data type based on the mode
        if mode == 0:
            dtype = np.int8
        elif mode == 1:
            dtype = np.int16
        elif mode == 2:
            dtype = np.float32
        elif mode == 6:
            dtype = np.uint16
        else:
            raise ValueError("Unsupported MRC mode: {}".format(mode))

        # Read the data
        data = np.fromfile(f, dtype=dtype, count=nx * ny * nz).reshape((nz, ny, nx))
    return data


deepict_pred = read_mrc(DEEPICT_PRED_PATH)
input_tomo = read_mrc(INPUT_PATH)
label_tomo = read_mrc(LABEL_PATH)

# %%

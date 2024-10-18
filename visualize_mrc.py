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

TOMOGRAM_PATH = os.path.join(
    "/media",
    "hdd1",
    "oliver",
    "topaz_test",
    f"reconstruction.mrc",
)


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


tomo = read_mrc(TOMOGRAM_PATH)
# %%

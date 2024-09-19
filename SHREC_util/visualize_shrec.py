# %%
import os
import mrcfile
import matplotlib.pyplot as plt
import struct
import numpy as np

def read_mrc(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (1024 bytes)
        header = f.read(1024)
        nx, ny, nz = struct.unpack('3i', header[0:12])
        mode = struct.unpack('i', header[12:16])[0]
        
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
            raise ValueError('Unsupported MRC mode: {}'.format(mode))
        
        # Read the data
        data = np.fromfile(f, dtype=dtype, count=nx*ny*nz).reshape((nz, ny, nx))
    return data

# %%
model_nr: int = 1
filename: str = 'class_mask.mrc'
FULL_PATH = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'SHREC',
    f'model_{model_nr}',
    filename
)
data = read_mrc(FULL_PATH)
print(f"Model {model_nr} shape: {data.shape}")
# %%

# PREDICTION
TS = '19092024_08:21'
FULL_PATH = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'SAM2_SHREC',
    'PREDICT',
    TS,
    f'model_{model_nr}_prediction.mrc',
)
pred_data = read_mrc(FULL_PATH)
print(f"Prediction {model_nr} shape: {pred_data.shape}")    
# %%

# %%
import os
import sys

sys.path.append("..")
from torch.utils.data import Dataset, Subset, ConcatDataset
import torch
import numpy as np
from torchvision.transforms import ToTensor
from FINETUNE.config import NOISE_VAR, PARTICLE_CLASS
import mrcfile
import torchvision.transforms.functional as TF
import struct


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


def create_multi_ds(
    main_folder, train_DS_IDs, val_DS_IDs, particle_id: int, device="cuda"
):
    train_datasets = []
    val_datasets = []
    for DS_ID in train_DS_IDs:
        ds = MRCDataset(
            main_folder=main_folder, DS_ID=DS_ID, particle_id=particle_id, device=device
        )
        train_datasets.append(ds)

    for DS_ID in val_DS_IDs:
        ds = MRCDataset(
            main_folder=main_folder, DS_ID=DS_ID, particle_id=particle_id, device=device
        )
        val_datasets.append(ds)

    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)

    print(f"Full train-DS length: {len(train_ds)}")

    val_indices = np.arange(len(val_ds))
    val_indices = val_indices[::4]

    val_ds = Subset(val_ds, val_indices)

    print(f"Train DS length: {len(train_ds)}")
    print(f"Val DS length: {len(val_ds)}")

    return train_ds, val_ds


class MRCDataset(Dataset):
    def __init__(self, main_folder, DS_ID, particle_id: int, device="cuda"):
        """
        main_folder: str (e.g., '/media/hdd1/oliver/SHREC')
        DS_ID: str (e.g., 'model_0')
        NOISE_VAR: float (variance of the Gaussian noise relative to the data's standard deviation)
        """
        self.device = device

        # Paths to the mrc files
        data_dir = os.path.join(main_folder, DS_ID)
        # input_file = os.path.join(data_dir, 'grandmodel.mrc')
        input_file = os.path.join(data_dir, "reconstruction.mrc")
        label_file = os.path.join(data_dir, "class_mask.mrc")

        # Read the mrc files
        uncropped_data = read_mrc(input_file).copy().astype(np.float32)
        self.input_volume = uncropped_data[156:356]  # shape (200, H, W)

        self.label_volume = read_mrc(label_file).copy()

        self.label_volume = (self.label_volume == particle_id).astype(np.uint8)
        self.input_volume = (self.input_volume - np.mean(self.input_volume)) / np.std(
            self.input_volume
        )
        assert (
            self.input_volume.shape == self.label_volume.shape
        ), "Input and label volumes must have the same shape"

        # Store the depth (number of slices)
        self.depth = self.input_volume.shape[0]

    def __len__(self):
        return self.depth

    def __getitem__(self, idx):
        # Get the slice at idx
        input_slice = self.input_volume[idx]  # shape (H, W)
        label_slice = self.label_volume[idx]  # shape (H, W)

        # Convert to torch tensors
        input_slice_tensor = torch.from_numpy(input_slice).unsqueeze(
            0
        )  # shape (1, H, W)
        label_slice_tensor = torch.from_numpy(label_slice).unsqueeze(
            0
        )  # shape (1, H, W)

        # Resize input to (1, 1024, 1024)
        input_slice_tensor = TF.resize(
            input_slice_tensor,
            [1024, 1024],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        # normalize to [0, 1]
        # input_slice_tensor = (input_slice_tensor - input_slice_tensor.min()) / (input_slice_tensor.max() - input_slice_tensor.min())

        # Resize label to (1, 256, 256)
        label_slice_tensor = TF.resize(
            label_slice_tensor, [256, 256], interpolation=TF.InterpolationMode.NEAREST
        )

        # Repeat input_slice_tensor to 3 channels
        input_slice_tensor = input_slice_tensor.repeat(3, 1, 1)

        # Convert to float32 for precision
        input_slice_tensor = input_slice_tensor.float()

        # Add Gaussian noise scaled to the data's standard deviation
        # noise_std = NOISE_VAR ** 0.5
        noise = torch.randn_like(input_slice_tensor) * NOISE_VAR**0.5
        input_slice_tensor = input_slice_tensor + noise

        # Clip the values to [0, 1] to maintain valid image range
        # input_slice_tensor = torch.clamp(input_slice_tensor, 0.0, 1.0)

        # Move to device
        input_slice_tensor = input_slice_tensor.to(self.device)
        label_slice_tensor = label_slice_tensor.float().to(self.device)

        return input_slice_tensor, label_slice_tensor


# ds = MRCDataset(
#     main_folder='/media/hdd1/oliver/SHREC',
#     DS_ID='model_1',
#     device='cpu'
# )
# import matplotlib.pyplot as plt
# plt.imshow(ds[140][0][1].cpu())
# plt.axis('off')
# %%

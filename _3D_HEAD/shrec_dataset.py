# %%
import os
import sys
sys.path.append('..')
from torch.utils.data import Dataset, Subset
import torch
import numpy as np
from torchvision.transforms import ToTensor
from _3D_HEAD.config import EIGHTH
import mrcfile
import torchvision.transforms.functional as TF

def create_train_val_datasets(main_folder, DS_ID, device='cuda', eighth=EIGHTH):
    # Load the full dataset
    full_dataset = MRCDataset(
        main_folder=main_folder,
        DS_ID=DS_ID,
        device=device
    )

    print(f"Full dataset length: {len(full_dataset)}")

    indices = np.arange(len(full_dataset))

    # Shuffle the indices to ensure randomness
    # np.random.shuffle(indices) -> TODO: LIKELY NOT SUPPOSED TO

    val_size = len(indices) // 8

    val_indices = indices[::8]

    train_indices = np.setdiff1d(indices, val_indices)

    if eighth == 1:
        train_indices = train_indices[::8]
    elif eighth == 2:
        train_indices = train_indices[::4]
    elif eighth == 4:
        train_indices = train_indices[::2]
    elif eighth == 8:
        pass
    else:
        raise ValueError("The 'eighth' parameter must be either 1, 2, 4, or 8.")

    # Verify the indices are within the valid range
    if max(train_indices) >= len(full_dataset) or max(val_indices) >= len(full_dataset):
        raise ValueError("The defined indices are out of the range of the dataset length.")

    # Create the training and validation datasets using Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    return train_dataset, val_dataset

class MRCDataset(Dataset):
    def __init__(self, main_folder, DS_ID, device='cuda'):
        """
        main_folder: str (e.g., '/media/hdd1/oliver/SHREC')
        DS_ID: str (e.g., 'model_0')
        """
        self.device = device

        # Paths to the mrc files
        data_dir = os.path.join(main_folder, DS_ID)
        input_file = os.path.join(data_dir, 'grandmodel.mrc')
        label_file = os.path.join(data_dir, 'class_mask.mrc')

        # Read the mrc files
        with mrcfile.open(input_file, permissive=True) as mrc:
            self.input_volume = mrc.data.copy()
        with mrcfile.open(label_file, permissive=True) as mrc:
            self.label_volume = mrc.data.copy()

        # From the class_mask, extract voxels with value 13
        # Create a binary mask where voxels with value 13 are 1, else 0
        self.label_volume = (self.label_volume == 13).astype(np.uint8)

        # Do Gaussian normalization on the input volume
        self.input_volume = (self.input_volume - np.mean(self.input_volume)) / np.std(self.input_volume)

        # Verify the volumes have the same shape
        assert self.input_volume.shape == self.label_volume.shape, "Input and label volumes must have the same shape"

        # Store the depth (number of slices)
        self.depth = self.input_volume.shape[0]

    def __len__(self):
        return self.depth

    def __getitem__(self, idx):
        # Get the slice at idx
        input_slice = self.input_volume[idx]  # shape (H, W)
        label_slice = self.label_volume[idx]  # shape (H, W)

        # Convert to torch tensors
        input_slice_tensor = torch.from_numpy(input_slice).unsqueeze(0)  # shape (1, H, W)
        label_slice_tensor = torch.from_numpy(label_slice).unsqueeze(0)  # shape (1, H, W)

        # Resize input to (1, 1024, 1024)
        input_slice_tensor = TF.resize(input_slice_tensor, [1024, 1024], interpolation=TF.InterpolationMode.BILINEAR)

        # Resize label to (1, 256, 256)
        label_slice_tensor = TF.resize(label_slice_tensor, [256, 256], interpolation=TF.InterpolationMode.NEAREST)

        # Repeat input_slice_tensor to 3 channels
        input_slice_tensor = input_slice_tensor.repeat(3, 1, 1)

        # Move to device
        input_slice_tensor = input_slice_tensor.to(self.device)
        label_slice_tensor = label_slice_tensor.float().to(self.device)

        return input_slice_tensor, label_slice_tensor


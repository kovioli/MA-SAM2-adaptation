# %%
import os
import sys

sys.path.append("..")
from pathlib import Path
import torch
import numpy as np
import mrcfile
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class MRCDataset(Dataset):
    """A PyTorch Dataset for loading and processing MRC (Medical Research Council) file format tomographic data.
    
    This dataset is designed to handle 3D tomographic data stored in MRC format, particularly for deep learning applications
    in cryo-electron microscopy. It processes volumetric data by creating a stack of three consecutive slices (previous,
    current, and next) for each position, enabling the model to consider spatial context in the z-dimension.

    Args:
        data_folder (str): Root directory containing the tomogram and label data.
        pred_class (str): Name of the prediction class/category (e.g., 'ribosomes', 'membranes').
        ds_id (str): Dataset identifier used to locate specific tomogram and label files.
        device (str, optional): Device to load tensors to ('cuda' or 'cpu'). If None, automatically selects
            CUDA if available, else CPU. Defaults to None.
        input_size (tuple, optional): Desired size (height, width) for input images. Defaults to (1024, 1024).
        label_size (tuple, optional): Desired size (height, width) for label images. Defaults to (256, 256).

    The dataset expects the following directory structure:
        data_folder/
        ├── tomograms/
        │   └── {ds_id}.mrc
        └── {pred_class}/
            └── {ds_id}.mrc

    Each tomogram slice is processed as follows:
        1. Input data is normalized using Gaussian normalization
        2. For each slice, a stack of 3 consecutive slices is created
        3. Both input and label data are resized to specified dimensions
        4. Data is converted to PyTorch tensors and moved to specified device
    """
    def __init__(
        self,
        data_folder,
        pred_class,
        ds_id,
        device=None,
        input_size=(1024, 1024),
        label_size=(256, 256)
    ):
        # Auto-select device if none specified
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.label_size = label_size
        self._load_full_tomogram(data_folder, pred_class, ds_id)

    def _load_full_tomogram(self, data_folder, pred_class, ds_id):
        tomogram_path = Path(data_folder) / "tomograms" / f"{ds_id}.mrc"
        label_path = Path(data_folder) / pred_class / f"{ds_id}.mrc"

        # Check if files exist
        if not tomogram_path.exists():
            raise FileNotFoundError(f"Tomogram file not found: {tomogram_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        try:
            # Load input tomogram
            with mrcfile.open(tomogram_path, permissive=True) as mrc:
                self.input_volume = mrc.data.copy().astype(np.float32)

            # Load labels
            with mrcfile.open(label_path, permissive=True) as mrc:
                self.label_volume = mrc.data.copy()

        except Exception as e:
            raise RuntimeError(f"Error loading MRC files: {str(e)}")

        # Validate volumes have same shape
        if self.input_volume.shape != self.label_volume.shape:
            raise ValueError(
                f"Input and label volumes must have the same shape. "
                f"Got {self.input_volume.shape} and {self.label_volume.shape}"
            )

        # Gaussian normalize the input volume
        self.input_volume = self._normalize_volume(self.input_volume)
        self.depth = len(self.input_volume)

        print(f"Volume shape: {self.input_volume.shape}")

    def _normalize_volume(self, volume):
        """Normalize volume using Gaussian normalization"""
        eps = 1e-6  # Prevent division by zero
        return (volume - np.mean(volume)) / (np.std(volume) + eps)

    def __len__(self):
        return self.depth

    def __getitem__(self, idx):
        if not 0 <= idx < self.depth:
            raise IndexError(f"Index {idx} out of range [0, {self.depth})")

        prev_idx = max(0, idx - 1)
        next_idx = min(self.depth - 1, idx + 1)
        label_slice = self.label_volume[idx].copy()

        input_stack = np.stack(
            [
                self.input_volume[prev_idx],
                self.input_volume[idx],
                self.input_volume[next_idx],
            ]
        )

        try:
            input_tensor = torch.from_numpy(input_stack)
            label_tensor = torch.from_numpy(label_slice).unsqueeze(0)

            input_tensor = TF.resize(
                input_tensor,
                self.input_size,
                interpolation=TF.InterpolationMode.BILINEAR,
            )

            label_tensor = TF.resize(
                label_tensor,
                self.label_size,
                interpolation=TF.InterpolationMode.BILINEAR,
            )

            input_tensor = input_tensor.float().to(self.device)
            label_tensor = label_tensor.float().to(self.device)

            return input_tensor, label_tensor

        except Exception as e:
            raise RuntimeError(f"Error processing slice {idx}: {str(e)}")
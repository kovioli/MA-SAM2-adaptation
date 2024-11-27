# %%
import os
import sys

sys.path.append("..")
import torch
import numpy as np
import mrcfile
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def train_val_ds(main_folder, s, p, device="cuda"):
    train_ds = MRCDataset(main_folder=main_folder, s=s, p=p, device=device)
    val_ds = MRCDataset(main_folder=main_folder, s=s, p=p, device=device)

    return train_ds, val_ds


class MRCDataset(Dataset):
    def __init__(
        self, main_folder, ds_id, s=None, p=None, device="cuda", full_tomo=False
    ):
        self.device = device
        self.full_tomo = full_tomo

        if full_tomo:
            # For testing: load and process full tomogram and labels
            self._load_full_tomogram(main_folder, ds_id)
        else:
            # For training: load sliced data with caching
            if s is None or p is None:
                raise ValueError("s and p must be provided when full_tomo=False")

            self.cache_dir = os.path.join(
                "...",
                ds_id,
                str(s),
                self.stringify_p(p),
            )

            if self._load_from_cache():
                print("Loaded from cache")
                return

            self._load_and_process_volumes(main_folder, ds_id, s, p)
            self._save_to_cache()

    def stringify_p(self, p):
        if isinstance(p, int) or p.is_integer():
            return str(int(p))
        return str(float(p)).replace(".", "_")

    def _calculate_slice_indices(self, total_slices, slice_count, position):
        """Calculate start and end indices for slicing, supporting float positions"""
        # Convert position to actual slice index
        position_float = float(position)
        start_idx = int(round(position_float * slice_count))
        end_idx = int(round((position_float + 1) * slice_count))

        # Ensure we don't exceed volume boundaries
        start_idx = max(0, min(start_idx, total_slices - slice_count))
        end_idx = min(total_slices, start_idx + slice_count)

        return start_idx, end_idx

    def _load_full_tomogram(self, main_folder, ds_id):
        tomogram_path = os.path.join(main_folder, "tomograms", f"{ds_id}.mrc")
        label_path = os.path.join(main_folder, "ribosomes", f"{ds_id}.mrc")

        # Load input tomogram
        with mrcfile.open(tomogram_path, permissive=True) as mrc:
            self.input_volume = mrc.data.copy().astype(np.float32)

        # Load labels - now required for testing
        with mrcfile.open(label_path, permissive=True) as mrc:
            self.label_volume = mrc.data.copy()

        # Validate volumes have same shape
        if self.input_volume.shape != self.label_volume.shape:
            raise ValueError(
                "Input and label volumes must have the same shape for testing"
            )

        # Gaussian normalize the input volume
        self.input_volume = self._normalize_volume(self.input_volume)
        self.depth = len(self.input_volume)

        print(f"Loaded full tomogram with {self.depth} slices")
        print(f"Volume shape: {self.input_volume.shape}")

    def _load_from_cache(self):
        if not os.path.exists(self.cache_dir):
            return False

        cache_tomogram = os.path.join(self.cache_dir, "tomogram.mrc")
        cache_ribosome = os.path.join(self.cache_dir, "ribosome.mrc")

        if not (os.path.exists(cache_tomogram) and os.path.exists(cache_ribosome)):
            return False

        with mrcfile.open(cache_tomogram, permissive=True) as mrc:
            self.input_volume = mrc.data

        with mrcfile.open(cache_ribosome, permissive=True) as mrc:
            self.label_volume = mrc.data

        self.depth = len(self.input_volume)
        return True

    def _load_and_process_volumes(self, main_folder, ds_id, slice_count, position):
        tomogram_path = os.path.join(main_folder, "tomograms", f"{ds_id}.mrc")
        label_path = os.path.join(main_folder, "ribosomes", f"{ds_id}.mrc")

        with mrcfile.open(tomogram_path, permissive=True) as mrc:
            input_volume = mrc.data.copy().astype(np.float32)

        with mrcfile.open(label_path, permissive=True) as mrc:
            label_volume = mrc.data.copy()

        self._validate_dimensions(input_volume, slice_count, position)

        norm_input_volume = self._normalize_volume(input_volume)
        start_idx, end_idx = self._calculate_slice_indices(
            len(input_volume), slice_count, position
        )

        self.input_volume = norm_input_volume[start_idx:end_idx]
        self.label_volume = label_volume[start_idx:end_idx]

        self._validate_volumes()
        self.depth = len(self.input_volume)

    def _normalize_volume(self, volume):
        return (volume - np.mean(volume)) / np.std(volume)

    def _validate_dimensions(self, volume, slice_count, position):
        if len(volume) < slice_count:
            raise ValueError(f"Slice size {slice_count} exceeds volume dimensions")

        if len(volume) < (position + 1) * slice_count:
            raise ValueError(
                f"Slices {slice_count * position} to {(position + 1) * slice_count} exceed volume dimensions"
            )

    def _validate_volumes(self):
        if self.input_volume.shape != self.label_volume.shape:
            raise ValueError("Input and label volumes must have the same shape")

    def _save_to_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)

        with mrcfile.new(
            os.path.join(self.cache_dir, "tomogram.mrc"), overwrite=True
        ) as mrc:
            mrc.set_data(self.input_volume)

        with mrcfile.new(
            os.path.join(self.cache_dir, "ribosome.mrc"), overwrite=True
        ) as mrc:
            mrc.set_data(self.label_volume)

    def __len__(self):
        return self.depth

    def __getitem__(self, idx):
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

        input_tensor = torch.from_numpy(input_stack)
        label_tensor = torch.from_numpy(label_slice).unsqueeze(0)

        input_tensor = TF.resize(
            input_tensor,
            [1024, 1024],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        label_tensor = TF.resize(
            label_tensor,
            [256, 256],
            interpolation=TF.InterpolationMode.BILINEAR,  # TODO: Change to NEAREST
        )

        input_tensor = input_tensor.float().to(self.device)
        label_tensor = label_tensor.float().to(self.device)

        return input_tensor, label_tensor


# Example usage:
# ds = MRCDataset(
#     main_folder=".../EMPIAR_clean",
#     ds_id="TS_0010",
#     s=64,
#     p=2,
#     device="cpu",
# )
# %%

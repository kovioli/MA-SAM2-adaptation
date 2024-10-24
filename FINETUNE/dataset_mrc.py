# %%
import os
import sys

sys.path.append("..")
from torch.utils.data import Dataset, Subset, ConcatDataset
import torch
import numpy as np
from torchvision.transforms import ToTensor
import mrcfile
import torchvision.transforms.functional as TF


def create_multi_ds(main_folder, train_DS_IDs, val_DS_IDs, device="cuda"):
    train_datasets = []
    val_datasets = []
    for DS_ID in train_DS_IDs:
        ds = MRCDataset(main_folder=main_folder, DS_ID=DS_ID, device=device)
        train_datasets.append(ds)

    for DS_ID in val_DS_IDs:
        ds = MRCDataset(main_folder=main_folder, DS_ID=DS_ID, device=device)
        val_datasets.append(ds)

    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)

    print(f"Full train-DS length: {len(train_ds)}")

    val_indices = np.arange(len(val_ds))
    val_indices = val_indices[::8]

    val_ds = Subset(val_ds, val_indices)

    print(f"Train DS length: {len(train_ds)}")
    print(f"Val DS length: {len(val_ds)}")

    return train_ds, val_ds


import pdb


class MRCDataset(Dataset):
    # TODO: fix downsampling!
    def __init__(self, main_folder, DS_ID, s, p, device="cuda"):
        """
        main_folder: str (e.g., '/media/hdd1/oliver/EMPIAR_clean')
        DS_ID: str (e.g., 'TS_0001')
        s: int number of slices
        p: int position of the slices in the volume
        """
        self.device = device

        tomogram_file = os.path.join(main_folder, "tomograms", f"{DS_ID}.mrc")
        label_file = os.path.join(main_folder, "ribosomes", f"{DS_ID}.mrc")

        with mrcfile.open(tomogram_file, permissive=True) as mrc:
            input_volume = mrc.data.copy().astype(np.float32)
        # pdb.set_trace()
        assert len(input_volume) >= s, f"Slice size {s} is out of bounds"
        assert (
            len(input_volume) >= (p + 1) * s
        ), f"Slices {s*p} to {(p+1)*s} are out of bounds"

        with mrcfile.open(label_file, permissive=True) as mrc:
            label_volume = mrc.data.copy()

        # Gaussian normalize
        norm_input_volume = (input_volume - np.mean(input_volume)) / np.std(
            input_volume
        )

        if p == 0:
            self.input_volume = norm_input_volume[:s]
            self.label_volume = label_volume[:s]
        else:
            self.input_volume = norm_input_volume[p * s : (p + 1) * s]
            self.label_volume = label_volume[p * s : (p + 1) * s]

        assert (
            self.input_volume.shape == self.label_volume.shape
        ), "Input and label volumes must have the same shape"
        self.depth = len(self.input_volume)

        # save if not already saved
        deepict_save_folder = os.path.join(
            "/media", "hdd1", "oliver", "EMPIAR_DCR", DS_ID, str(s), str(p)
        )
        # Create all parent directories if they don't exist
        os.makedirs(deepict_save_folder, exist_ok=True)

        # Check if files don't exist
        if not os.path.exists(
            os.path.join(deepict_save_folder, "tomogram.mrc")
        ) and not os.path.exists(os.path.join(deepict_save_folder, "ribosome.mrc")):
            # Save tomogram
            with mrcfile.new(
                os.path.join(deepict_save_folder, "tomogram.mrc"), overwrite=True
            ) as mrc:
                mrc.set_data(self.input_volume)
            # Save ribosome
            with mrcfile.new(
                os.path.join(deepict_save_folder, "ribosome.mrc"), overwrite=True
            ) as mrc:
                mrc.set_data(self.label_volume)

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

        # Resize label to (1, 256, 256)
        label_slice_tensor = TF.resize(
            label_slice_tensor, [256, 256], interpolation=TF.InterpolationMode.NEAREST
        )

        # Repeat input_slice_tensor to 3 channels
        input_slice_tensor = input_slice_tensor.repeat(3, 1, 1)
        # Convert to float32 for precision
        input_slice_tensor = input_slice_tensor.float()

        # Move to device
        input_slice_tensor = input_slice_tensor.to(self.device)
        label_slice_tensor = label_slice_tensor.float().to(self.device)

        return input_slice_tensor, label_slice_tensor

    # def __init__(self, main_folder, DS_ID, s, p, device="cuda"):


ds = MRCDataset(
    main_folder="/media/hdd1/oliver/EMPIAR_clean",
    DS_ID="TS_0001",
    s=128,
    p=0,
    device="cpu",
)
# ds = MRCDataset(
#     main_folder='/media/hdd1/oliver/SHREC',
#     DS_ID='model_1',
#     device='cpu'
# )
# import matplotlib.pyplot as plt
# plt.imshow(ds[140][0][1].cpu())
# plt.axis('off')
# %%

# %%
import os
import mrcfile
import numpy as np


def get_sliced_volume(volume, slice_start, num_slices=64, target_height=116):
    """
    Extract a volume region:
    - 64 slices along first dimension based on slice_start
    - Full width in second dimension
    - 116 pixels centered in the last dimension

    Parameters:
    volume: 3D numpy array
    slice_start: starting slice index multiplier
    num_slices: number of slices to extract (default: 64)
    target_height: size along the last dimension (default: 116)
    """
    _, _, w = volume.shape
    center_w = w // 2

    w_start = center_w - target_height // 2
    w_end = w_start + target_height

    return volume[
        slice_start * num_slices : (slice_start + 1) * num_slices,
        :,  # take full width
        w_start:w_end,  # take centered box in last dimension
    ]


def save_mrc(volume, path):
    """
    Save volume as MRC file with proper data type conversion.

    Parameters:
    volume: numpy array to save
    path: output path for MRC file
    """
    # Convert to float32 before saving
    volume_float32 = volume.astype(np.float32)

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(volume_float32)
    print(f"Saved volume at {path}")


def process_tomo(position):
    tomogram_path = os.path.join(
        "/media/hdd1/oliver/EMPIAR_clean", "tomograms", "TS_0001.mrc"
    )
    ribosome_path = os.path.join(
        "/media/hdd1/oliver/EMPIAR_clean", "ribosomes", "TS_0001.mrc"
    )
    with mrcfile.open(tomogram_path, permissive=True) as mrc:
        tomogram = mrc.data
    with mrcfile.open(ribosome_path, permissive=True) as mrc:
        ribosome = mrc.data

    # Convert to float32 after normalization
    norm_tomo = ((tomogram - np.mean(tomogram)) / np.std(tomogram)).astype(np.float32)

    tomo_sliced = get_sliced_volume(norm_tomo, position)
    ribo_sliced = get_sliced_volume(ribosome, position)

    return tomo_sliced, ribo_sliced


if __name__ == "__main__":
    orig_slice = 8
    cache_dir = os.path.join(
        "/media", "hdd1", "oliver", "EMPIAR_DCR", "TS_0001", f"{orig_slice}-64"
    )

    os.makedirs(cache_dir, exist_ok=True)
    for position in [0, 1, 2]:
        print(f"Processing position {position}")
        final_dir = os.path.join(cache_dir, str(position))
        os.makedirs(final_dir, exist_ok=True)
        tomo, ribo = process_tomo(position=position)
        save_mrc(tomo, os.path.join(final_dir, "tomogram.mrc"))
        save_mrc(ribo, os.path.join(final_dir, "ribosome.mrc"))
# %%

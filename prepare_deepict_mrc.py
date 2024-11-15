# %%
import os
import mrcfile
import numpy as np


def get_sliced_volume(volume, slice_start, target_size, num_slices=64):
    """
    Extract a volume region centered in all dimensions except the first:
    - 64 slices along first dimension based on slice_start
    - Centered box of target_size pixels in second dimension
    - Centered box of target_size pixels in the last dimension

    Parameters:
    volume: 3D numpy array
    slice_start: starting slice index multiplier
    num_slices: number of slices to extract (default: 64)
    target_size: size along second and last dimensions (default: 116)
    """
    _, h, w = volume.shape
    center_h = h // 2
    center_w = w // 2

    h_start = center_h - target_size // 2
    h_end = h_start + target_size
    w_start = center_w - target_size // 2
    w_end = w_start + target_size

    return volume[
        slice_start * num_slices : (slice_start + 1) * num_slices,
        h_start:h_end,  # take centered box in second dimension
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


def process_tomo(position, target_size):
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

    tomo_sliced = get_sliced_volume(norm_tomo, position, target_size=target_size)
    ribo_sliced = get_sliced_volume(ribosome, position, target_size=target_size)

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
        tomo, ribo = process_tomo(position=position, target_size=328)
        save_mrc(tomo, os.path.join(final_dir, "tomogram.mrc"))
        save_mrc(ribo, os.path.join(final_dir, "ribosome.mrc"))
# %%

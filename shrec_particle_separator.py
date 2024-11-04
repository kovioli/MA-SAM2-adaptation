import os
import numpy as np
from mrcfile import new as new_mrc
import glob
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


def save_mrc(data, filename):
    """Save numpy array as MRC file."""
    with new_mrc(filename, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))


def process_dataset(input_folder, output_base_folder):
    """Process all models in the dataset."""
    # Get all model folders
    model_folders = sorted(glob.glob(os.path.join(input_folder, "model_*")))

    for model_folder in model_folders:
        model_name = os.path.basename(model_folder)
        print(f"Processing {model_name}...")

        # Create output directories
        output_folder = os.path.join(output_base_folder, model_name)
        os.makedirs(output_folder, exist_ok=True)

        # Read the class mask and reconstruction
        class_mask_path = os.path.join(model_folder, "class_mask.mrc")
        reconstruction_path = os.path.join(model_folder, "reconstruction.mrc")

        class_mask = read_mrc(class_mask_path)
        reconstruction = read_mrc(reconstruction_path)

        # Get unique class IDs (excluding 0 which is typically background)
        class_ids = np.unique(class_mask)
        class_ids = class_ids[class_ids != 0]  # Remove background class if present

        # Process each class
        for class_id in class_ids:
            # Create binary mask for this class
            binary_mask = (class_mask == class_id).astype(np.float32)

            # Save class mask
            mask_output_path = os.path.join(
                output_folder, f"class_mask_{int(class_id)}.mrc"
            )
            save_mrc(binary_mask, mask_output_path)

        # Save cropped reconstruction (slices 156:356)
        cropped_reconstruction = reconstruction[156:356].copy()
        reconstruction_output_path = os.path.join(
            output_folder, "reconstruction_cropped.mrc"
        )
        save_mrc(cropped_reconstruction, reconstruction_output_path)

        print(f"Completed processing {model_name}")


if __name__ == "__main__":
    input_folder = "/media/hdd1/oliver/shrec2020_full_dataset"
    output_folder = "/media/hdd1/oliver/DEEPICT_SHREC"

    print("Starting dataset processing...")
    process_dataset(input_folder, output_folder)
    print("Processing complete!")

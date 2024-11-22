# %%
# In your main pipeline, when you hit the point of interest:
import numpy as np
import torch
import matplotlib.pyplot as plt


def save_debug_state(
    input_slice, label_slice, pred, denoised_img, pred_sigmoid, pred_binary
):
    # Create a dictionary of all the tensors you want to save
    debug_data = {
        "input_slice": input_slice.cpu().numpy(),
        "label_slice": label_slice.cpu().numpy(),
        "pred": pred.cpu().numpy(),
        "denoised_img": denoised_img.cpu().numpy(),
        "pred_sigmoid": pred_sigmoid.cpu().numpy(),
        "pred_binary": pred_binary.cpu().numpy(),
    }

    # Save all tensors
    np.save("debug_data.npy", debug_data)


# Then in a separate debug.py file:
def load_debug_state():
    # Load the saved data
    debug_data = np.load("debug_data.npy", allow_pickle=True).item()

    # Convert back to torch tensors if needed
    debug_tensors = {k: torch.from_numpy(v) for k, v in debug_data.items()}

    return debug_tensors


# %%

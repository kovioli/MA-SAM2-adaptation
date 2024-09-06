# %%
import numpy as np
from _3D_HEAD.dataset import PNGDataset


# %%
ds = PNGDataset('/oliver/EMPIAR_png', 'TS_0005')

# List to store the number of label points in each slice
label_points = []

# Calculate the number of label points for each slice
for idx in range(len(ds)):
    _, label = ds[idx]  # Get the label image
    num_points = (label > 0.025).sum().item()  # Count non-zero points in the label
    label_points.append(num_points)

# Convert label points to numpy array for easier manipulation
label_points = np.array(label_points)

# Find the block of N consecutive slices with the most label points
block_size = 64
cumulative_sums = np.convolve(label_points, np.ones(block_size, dtype=int), 'valid')
best_start_idx = np.argmax(cumulative_sums)
best_end_idx = best_start_idx + block_size

# Output the result
print(f"Best block starts at slice {best_start_idx} and ends at slice {best_end_idx} with {cumulative_sums[best_start_idx]} label points.")

# %%

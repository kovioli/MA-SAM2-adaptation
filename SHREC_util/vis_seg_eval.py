# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Data
json_folder = os.path.join(
    '/oliver',
    'SAM2',
    'SHREC_segmentation_results'
)

TS = '19092024_10:43'
json_file = os.path.join(
    json_folder,
    f'segmentation_results_{TS}.json'
)
with open(json_file) as json_f:
    data = json.load(json_f)

# Preparing the data for the bar chart
models = [d['ds_name'] for d in data]
dice_scores = [d['dice'] for d in data]
iou_scores = [d['IoU'] for d in data]

x = np.arange(len(models))  # label locations
width = 0.35  # bar width

# Plotting the bars
fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, dice_scores, width, label='Dice')
bar2 = ax.bar(x + width/2, iou_scores, width, label='IoU')
plt.grid(True)

# Adding labels and title
ax.set_xlabel('DS Name')
ax.set_ylabel('Scores')
ax.set_title(f'Dice & IoU for TS {TS}')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%

filtered_new_data = data[1:]  # Exclude model_0

# Calculate the averages
average_dice_new = np.mean([d['dice'] for d in filtered_new_data])
average_iou_new = np.mean([d['IoU'] for d in filtered_new_data])

print(f"{average_dice_new:.3f} / {average_iou_new:.3f}")

# %%

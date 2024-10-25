# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample data as string (replace with actual data reading mechanism)

DS_ID = "TS_0001"
NR_SLICES = 128

with open(f"log_s{NR_SLICES}.csv", "r") as file:
    csv_data = file.read()

# Convert to DataFrame, filtering relevant columns and parsing iou/dice values
df = pd.DataFrame(
    [x.split(",") for x in csv_data.splitlines()],
    columns=["timestamp", "size", "series", "s32", "position", "run", "iou", "dice"],
)

# Extract numeric values from iou and dice columns
df["iou"] = df["iou"].str.split("=").str[1].astype(float)
df["dice"] = df["dice"].str.split("=").str[1].astype(float)

# Calculate mean and standard deviation for each position
stats = (
    df.groupby("position")
    .agg({"iou": ["mean", "std"], "dice": ["mean", "std"]})
    .reset_index()
)

# Flatten column names
stats.columns = ["position", "iou_mean", "iou_std", "dice_mean", "dice_std"]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot error bars for IOU
plt.errorbar(
    range(len(stats)),
    stats["iou_mean"],
    yerr=stats["iou_std"],
    fmt="o",
    label="IOU",
    capsize=5,
    capthick=1.5,
    color="blue",
    ecolor="lightblue",
    markersize=8,
)

# Plot error bars for Dice
plt.errorbar(
    range(len(stats)),
    stats["dice_mean"],
    yerr=stats["dice_std"],
    fmt="o",
    label="Dice",
    capsize=5,
    capthick=1.5,
    color="red",
    ecolor="lightcoral",
    markersize=8,
)

# Customize the plot
plt.xlabel("Position", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title(
    f"Mean IOU and Dice Scores with Standard Deviation ({DS_ID}, s{NR_SLICES})",
    fontsize=14,
)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(range(len(stats)), stats["position"], rotation=45)

# Add legend
plt.legend(fontsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print numerical results
print("\nNumerical Results:")
print(stats.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

# %%
# Group by position and get the index of max dice score for each group
best_idx = df.groupby("position")["dice"].idxmax()

# Create new dataframe with only the best runs
best_runs = df.loc[best_idx]

# %%

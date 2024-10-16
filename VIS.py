# %%
import numpy as np
import matplotlib.pyplot as plt

data = [
    {
        "particle_id": 1,
        "particle_name": "3cf3",
        "ts": "08102024_06:26",
        "dice": 0.6254814271095828,
        "iou": 0.4865257271591262,
    },
    {
        "particle_id": 2,
        "particle_name": "1s3x",
        "ts": "08102024_07:51",
        "dice": 0.2760966913611159,
        "iou": 0.18414315628348818,
    },
    {
        "particle_id": 3,
        "particle_name": "1u6g",
        "ts": "08102024_08:58",
        "dice": 0.4412708629230347,
        "iou": 0.3033391277427163,
    },
    {
        "particle_id": 4,
        "particle_name": "4cr2",
        "ts": "08102024_10:08",
        "dice": 0.593712338270639,
        "iou": 0.47123517470995047,
    },
    {
        "particle_id": 5,
        "particle_name": "1qvr",
        "ts": "08102024_11:45",
        "dice": 0.6645302120364743,
        "iou": 0.5235090248800356,
    },
    {
        "particle_id": 6,
        "particle_name": "3h84",
        "ts": "08102024_13:25",
        "dice": 0.48810508114933543,
        "iou": 0.3480685469293552,
    },
    {
        "particle_id": 7,
        "particle_name": "2cg9",
        "ts": "08102024_14:49",
        "dice": 0.4803865978909073,
        "iou": 0.3406617296715615,
    },
    {
        "particle_id": 8,
        "particle_name": "3qm1",
        "ts": "08102024_15:56",
        "dice": 0.3772355495068198,
        "iou": 0.25482831582901533,
    },
    {
        "particle_id": 9,
        "particle_name": "3gl1",
        "ts": "08102024_16:58",
        "dice": 0.3983848592149745,
        "iou": 0.2769999571842792,
    },
    {
        "particle_id": 10,
        "particle_name": "3d2f",
        "ts": "08102024_18:24",
        "dice": 0.42057448906166733,
        "iou": 0.2849288853005904,
    },
    {
        "particle_id": 11,
        "particle_name": "4d8q",
        "ts": "08102024_19:47",
        "dice": 0.720537208751731,
        "iou": 0.5999007421591809,
    },
    {
        "particle_id": 12,
        "particle_name": "1bxn",
        "ts": "08102024_20:55",
        "dice": 0.7254158765246135,
        "iou": 0.5848732810636026,
    },
]

# Extracting values
names = [d["particle_name"] for d in data]
dice_values = [d["dice"] for d in data]
iou_values = [d["iou"] for d in data]

x = np.arange(len(names))  # label locations
width = 0.35  # bar width

# Creating subplots
fig, ax = plt.subplots()
bars1 = ax.bar(x - width / 2, dice_values, width, label="Dice", color="blue")
bars2 = ax.bar(x + width / 2, iou_values, width, label="IOU", color="orange")

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel("Scores")
ax.set_title("Particle Exploration (8DS)")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()


# Adding value labels above each bar
def add_labels(bars, decimals=3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


# Apply the labels
add_labels(bars1)
add_labels(bars2)

# Show the plot
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
classes = [
    "3cf3",
    "1s3x",
    "1u6g",
    "4cr2",
    "1qvr",
    "3h84",
    "2cg9",
    "3qm1",
    "3gl1",
    "3d2f",
    "4d8q",
    "1bxn",
]
f1_scores = [
    0.97216,
    0.45863,
    0.97424,
    0.98462,
    0.97738,
    0.86651,
    0.91494,
    0.62663,
    0.7551,
    0.77273,
    0.9852,
    0.99177,
]

# Sort data by F1 score in descending order
sorted_data = sorted(zip(classes, f1_scores), key=lambda x: x[1], reverse=True)
classes, f1_scores = zip(*sorted_data)

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 1.5))  # Reduced height

# Define a more vivid color palette
vivid_colors = [
    "#FF1493",
    "#00FF00",
    "#1E90FF",
    "#FF4500",
    "#FFD700",
    "#FF00FF",
    "#00FFFF",
    "#32CD32",
    "#FF69B4",
    "#4169E1",
    "#8A2BE2",
    "#00FF7F",
]

# Plot stacked bars
left = 0
for i, (f1_score, color) in enumerate(zip(f1_scores, vivid_colors)):
    ax.barh(y=0, width=f1_score, left=left, height=0.5, color=color, label=classes[i])
    left += f1_score

# Customize the plot
ax.set_ylim(-0.5, 0.5)  # Center the bar
ax.set_xlabel("Cumulative F1 Score")
ax.set_yticks([])  # Remove y-axis ticks
ax.set_title(
    f"Cumulative Classification F1 Scores of Methods (sum: {sum(f1_scores):.2f})",
    pad=20,
)  # Add some padding to the title

# Add a legend below the chart
ax.legend(
    title="Classes",
    bbox_to_anchor=(0.5, -0.2),
    loc="upper center",
    ncol=6,
    fontsize="x-small",
)

# Remove the frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Adjust layout and display
plt.tight_layout()
plt.show()

# %%

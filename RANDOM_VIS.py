# %%
import json
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np


def extract_p_number(filename):
    """Extract p number from filename like TS_0001_s64_p1_24102024_23:43:23.json or TS_0001_s64_p0_4_24102024_23:43:23.json"""
    match = re.search(r"_p(\d+(?:_\d+)?)_", filename)
    if match:
        p_value = match.group(1).replace("_", ".")
        return f"p{p_value}"
    return "unknown"


def plot_f1_scores(json_files, slice_nr):
    """
    Plot F1 scores from multiple JSON files
    Args:
        json_files: List of tuples (filename, file_content)
    """
    plt.figure(figsize=(12, 6))

    # Set style
    plt.style.use("default")
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
    ]  # Different markers for different files

    # Keep track of min and max F1 scores
    min_f1 = float("inf")
    max_f1 = float("-inf")

    # Create plot
    for (filename, data, l), marker in zip(json_files, markers):
        # Extract x and y values
        ds_names = [item["ds_name"] for item in data]
        f1_scores = [item["F1"] for item in data]

        # Update min and max F1
        min_f1 = min(min_f1, min(f1_scores))
        max_f1 = max(max_f1, max(f1_scores))

        # Get label from filename
        # label = extract_p_number(filename)
        label = l

        # Plot lines connecting points
        plt.plot(ds_names, f1_scores, "-", alpha=0.6, label=None)

        # Plot scatter points
        plt.scatter(ds_names, f1_scores, label=label, marker=marker, s=100, alpha=1)

    # Customize plot
    plt.xlabel("Evaluation dataset", fontsize=12)
    plt.ylabel("F1 Score (picking)", fontsize=12)
    plt.title(f"F1 Scores (Slice number {slice_nr})", fontsize=14, pad=20)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Positions", loc="lower left")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Set y-axis limits with some padding
    plt.ylim(min_f1 - 0.05, max_f1 + 0.05)

    return plt


# Example usage:
if __name__ == "__main__":
    import os

    # Sample data structure - replace with actual file reading
    # json_file_names = [
    #     "TS_0001_s64_p0_24102024_23:24:34.json",
    #     "TS_0001_s64_p1_24102024_23:43:23.json",
    #     "TS_0001_s64_p2_24102024_23:53:13.json",
    # ]
    json_file_names = [
        ("TS_0001_s32_p0_24102024_23:02:12.json", 0),
        ("TS_0001_s32_p1_24102024_23:18:50.json", 1),
        ("TS_0001_s32_p2_24102024_23:46:51.json", 2),
        ("TS_0001_s32_p3_25102024_00:09:23.json", 3),
        ("TS_0001_s32_p4_25102024_00:33:05.json", 4),
        ("TS_0001_s32_p5_25102024_00:51:19.json", 5),
    ]
    SLICE_NR = "8-64"
    # json_file_names = [
    #     ("TS_0001_s128_p0_r1.json", 0),
    #     ("TS_0001_s128_p0_4_r1.json", 0.4),
    # ]
    json_file_names = [
        ("TS_0001_s16_p10_r1.json", 10),
        ("TS_0001_s16_p4_r4.json", 4),
        ("TS_0001_s16_p6_r2.json", 6),
        ("TS_0001_s16_p8_r3.json", 8),
    ]
    json_file_names = [
        ("TS_0001_s8-64_p0_r2.json", 0),
        ("TS_0001_s8-64_p1_r3.json", 1),
        ("TS_0001_s8-64_p2_r1.json", 2),
    ]
    json_file_names = [
        ("TS_0001_s16-64_p0_r4.json", 0),
        ("TS_0001_s16-64_p1_r4.json", 1),
        ("TS_0001_s16-64_p2_r2.json", 2),
    ]
    json_file_names = [
        ("TS_0001_s8-64_p0_r0.json", 0),
        ("TS_0001_s8-64_p1_r4.json", 1),
        ("TS_0001_s8-64_p2_r4.json", 2),
    ]
    json_files = []
    for file_name in json_file_names:
        with open(os.path.join("dc_eval", "DeePiCt", file_name[0]), "r") as file:
            data = json.load(file)
            json_files.append((file_name, data, file_name[1]))

    # Create the plot
    plt = plot_f1_scores(json_files, SLICE_NR)

    # Show or save the plot
    # plt.savefig("f1_scores_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%

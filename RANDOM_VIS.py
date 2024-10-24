# %%
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the three files
all_runs = {
    "Run 1": [
        {"ds_name": "TS_0002", "F1": 0.6222394420767143},
        {"ds_name": "TS_0003", "F1": 0.5876923076923077},
        {"ds_name": "TS_0004", "F1": 0.562754632671836},
        {"ds_name": "TS_0005", "F1": 0.6499665998663994},
        {"ds_name": "TS_0006", "F1": 0.6695517774343123},
        {"ds_name": "TS_0007", "F1": 0.46168312176774806},
        {"ds_name": "TS_0008", "F1": 0.615257504819609},
        {"ds_name": "TS_0009", "F1": 0.6564330911647527},
        {"ds_name": "TS_0010", "F1": 0.6103516765977334},
    ],
    "Run 2": [
        {"ds_name": "TS_0002", "F1": 0.6929667264395298},
        {"ds_name": "TS_0003", "F1": 0.5520817447003202},
        {"ds_name": "TS_0004", "F1": 0.7085597826086958},
        {"ds_name": "TS_0005", "F1": 0.669466806673667},
        {"ds_name": "TS_0006", "F1": 0.6141498216409037},
        {"ds_name": "TS_0007", "F1": 0.317780580075662},
        {"ds_name": "TS_0008", "F1": 0.7131120604666448},
        {"ds_name": "TS_0009", "F1": 0.6173410404624278},
        {"ds_name": "TS_0010", "F1": 0.5788946958745691},
    ],
    "Run 3": [
        {"ds_name": "TS_0002", "F1": 0.7881528374406511},
        {"ds_name": "TS_0003", "F1": 0.7408235513322154},
        {"ds_name": "TS_0004", "F1": 0.6694352159468439},
        {"ds_name": "TS_0005", "F1": 0.7913884725439433},
        {"ds_name": "TS_0006", "F1": 0.795698924731183},
        {"ds_name": "TS_0007", "F1": 0.5342394145321485},
        {"ds_name": "TS_0008", "F1": 0.6008151522416686},
        {"ds_name": "TS_0009", "F1": 0.7827748383303939},
        {"ds_name": "TS_0010", "F1": 0.7682100898410503},
    ],
    "Run 4": [
        {"ds_name": "TS_0002", "F1": 0.6812842599843383},
        {"ds_name": "TS_0003", "F1": 0.7253705318221446},
        {"ds_name": "TS_0004", "F1": 0.21939680662329986},
        {"ds_name": "TS_0005", "F1": 0.749093107617896},
        {"ds_name": "TS_0006", "F1": 0.7485477178423237},
        {"ds_name": "TS_0007", "F1": 0.5916197623514696},
        {"ds_name": "TS_0008", "F1": 0.22855333966445077},
        {"ds_name": "TS_0009", "F1": 0.7256870166200429},
        {"ds_name": "TS_0010", "F1": 0.7570818167431554},
    ],
    "Run 5": [
        {"ds_name": "TS_0002", "F1": 0.7793448589626932},
        {"ds_name": "TS_0003", "F1": 0.7382198952879582},
        {"ds_name": "TS_0004", "F1": 0.6113301514837103},
        {"ds_name": "TS_0005", "F1": 0.7821043910521955},
        {"ds_name": "TS_0006", "F1": 0.7702143663031216},
        {"ds_name": "TS_0007", "F1": 0.5485327313769751},
        {"ds_name": "TS_0008", "F1": 0.46907216494845355},
        {"ds_name": "TS_0009", "F1": 0.7394151025752946},
        {"ds_name": "TS_0010", "F1": 0.7567409998542487},
    ],
}

# Extracting dataset names and F1 scores for all runs
dataset_names = [item["ds_name"] for item in all_runs["Run 1"]]
f1_scores_per_run = {run: [item["F1"] for item in all_runs[run]] for run in all_runs}

# Creating the plot with 3 bars per dataset
bar_width = 0.1
index = np.arange(len(dataset_names))

plt.figure(figsize=(12, 8))

# Plotting the bars for each run
colors = ["red", "green", "orange", "darkviolet", "blue"]
for i, (run, f1_scores) in enumerate(f1_scores_per_run.items()):
    plt.bar(index + i * bar_width, f1_scores, bar_width, label=run, color=colors[i])

# Calculating and plotting the average F1 score for each run
for i, (run, f1_scores) in enumerate(f1_scores_per_run.items()):
    avg_f1 = np.mean(f1_scores)
    plt.axhline(
        y=avg_f1, color=colors[i], linestyle="--", label=f"{run} Average: {avg_f1:.2f}"
    )

# Adding labels and title
plt.xlabel("Dataset Names")
plt.ylabel("F1 Scores")
plt.title("F1 Scores for 5 Runs with Averages (Holdout Tomo)")
plt.xticks(index + bar_width, dataset_names, rotation=45)

# Adding legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# %%

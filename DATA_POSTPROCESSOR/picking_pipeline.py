# %%
# TODO: export PYTHONPATH="/oliver/MA:$PYTHONPATH"
import sys

sys.path.append("..")
import os
import matplotlib.pyplot as plt
import pdb
import numpy as np
from statistics import mean
import json
from DATA_POSTPROCESSOR.clustering_and_cleaning import cluster_and_clean
from DATA_POSTPROCESSOR.particle_picking_eval import eval_picking
from DATA_POSTPROCESSOR.predict_tomogram import predict
import pandas as pd

TRAIN_DS_ID = "TS_0001"
VAL_DS_ID = "TS_0010"
memory_used = False

pred_tomogram_info_list = [
    {"name": "TS_0001", "z_offset": 390, "target_shape": (210, 927, 927)},
    {"name": "TS_0002", "z_offset": 380, "target_shape": (240, 927, 927)},
    {"name": "TS_0003", "z_offset": 380, "target_shape": (250, 927, 927)},
    {"name": "TS_0004", "z_offset": 340, "target_shape": (300, 927, 927)},
    {"name": "TS_0005", "z_offset": 110, "target_shape": (280, 928, 928)},
    {"name": "TS_0006", "z_offset": 170, "target_shape": (140, 928, 928)},
    {"name": "TS_0007", "z_offset": 200, "target_shape": (150, 928, 928)},
    {"name": "TS_0008", "z_offset": 100, "target_shape": (400, 928, 928)},
    {"name": "TS_0009", "z_offset": 120, "target_shape": (250, 928, 928)},
    {"name": "TS_0010", "z_offset": 350, "target_shape": (290, 927, 927)},
]
training_folder = os.path.join("/media", "hdd1", "oliver", "SAM2_EMPIAR_DCR")


# threshold_list = np.arange(0.6, 0.85, 0.025)
threshold_list = np.arange(0.65, 0.9, 0.025)
# threshold_list = [0.5]
# threshold_list = [0.00001]
if __name__ == "__main__":
    NR_SLICES = 8

    with open(f"/oliver/SAM2/log_s{NR_SLICES}.csv", "r") as file:
        csv_data = file.read()

    # Convert to DataFrame, filtering relevant columns and parsing iou/dice values
    df = pd.DataFrame(
        [x.split(",") for x in csv_data.splitlines()],
        columns=[
            "timestamp",
            "size",
            "series",
            "s32",
            "position",
            "run",
            "iou",
            "dice",
        ],
    )
    best_idx = df.groupby("position")["dice"].idxmax()
    # Create new dataframe with only the best runs
    best_runs = df.loc[best_idx]

    for idx, row in best_runs.iterrows():
        timestamp = row["timestamp"]
        position = row["position"]
        F1_results = []
        PRED_FOLDER = os.path.join(training_folder, "PREDICT", f"{timestamp}")
        print(
            f"EVALUATING SLICE {NR_SLICES}, POSITION {position} WITH TIMESTAMP {timestamp}"
        )

        for tomo_info in pred_tomogram_info_list:
            # Skip the training dataset
            tomo_id = tomo_info.get("name")
            if TRAIN_DS_ID == tomo_id or VAL_DS_ID == tomo_id:
                continue
            print(f"\n\n{50*'-'}\nPROCESSING DS: {tomo_id}\n{50*'-'}\n")
            # Find the relevant info
            gt_motl_path = os.path.join(
                "/oliver", "MA", "GT_particle_lists", f"{tomo_id}_cyto_ribosomes.csv"
            )

            best_F1_on_th = -1.0  # best F1 score depending on threshold

            for th in threshold_list:
                print(f"THRESHOLD: {th}")
                motl_file_path = cluster_and_clean(
                    threshold=th,
                    min_cluster_size=0,
                    max_cluster_size=None,
                    clustering_connectivity=1,
                    prediction_dir=PRED_FOLDER,
                    tomogram_name=tomo_id,
                )
                if motl_file_path is None:
                    continue

                _, max_F1, _ = eval_picking(
                    predicted_motl_path=motl_file_path,
                    gt_motl_path=gt_motl_path,
                    z_offset=tomo_info.get("z_offset"),
                    ratio=(1, 1),  # motl_ratio
                )
                if max_F1 > best_F1_on_th:
                    best_F1_on_th = max_F1
                else:  # if the F1 score is decreasing, break the loop
                    break
            F1_results.append({"ds_name": f"{tomo_id}", "F1": float(best_F1_on_th)})
        # TODO: change json save path
        with open(
            os.path.join(
                "/oliver",
                "SAM2",
                "dc_eval",
                "SAM2",
                f"{TRAIN_DS_ID}_s{NR_SLICES}_{position}_{timestamp}.json",
            ),
            "w",
        ) as f:
            json.dump(F1_results, f, indent=4)

        average_F1 = mean([result["F1"] for result in F1_results])
        print(f"Average F1 score: {average_F1}")

# %%

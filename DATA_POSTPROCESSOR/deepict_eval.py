# %%
import os
import json
import sys

sys.path.append("..")
from statistics import mean
from DATA_POSTPROCESSOR.particle_picking_eval import eval_picking

TRAIN_DS_ID = "0001"
VAL_DS_ID = "0010"
MODEL_NAME = "TS_0001_s8-64_p2_r4"
model_name = f"{MODEL_NAME}_best"
# TS_0001_s8-64_p0_r0
# TS_0001_s8-64_p1_r4
# TS_0001_s8-64_p2_r4

deepict_pred_folder = os.path.join(
    "/media", "hdd1", "oliver", "DeePiCt", "PREDICT", "predictions", model_name
)

TOMO_IDs = [
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
]

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

F1_results = []

for tomo_id in TOMO_IDs:
    # Skip the training dataset
    if tomo_id in [TRAIN_DS_ID, VAL_DS_ID]:
        continue

    # Find the relevant info
    pred_tomogram_info = [
        item for item in pred_tomogram_info_list if item.get("name") == f"TS_{tomo_id}"
    ][0]

    # Find the motl file
    motl_folder = os.path.join(deepict_pred_folder, f"TS_{tomo_id}", "ribo")
    motl_file_name = ""
    for file_name in os.listdir(motl_folder):
        if file_name.startswith("motl_") and file_name.endswith(".csv"):
            motl_file_name = file_name
            break
    if not motl_file_name:
        print(f"No motl file found for tomo_id {tomo_id}")
        continue

    # Evaluate the picking
    print(f"Processing tomo_id {tomo_id} -> {motl_file_name}")
    predicted_motl_path = os.path.join(motl_folder, motl_file_name)
    gt_motl_path = os.path.join(
        "/oliver", "SAM2", "GT_particle_lists", f"TS_{tomo_id}_cyto_ribosomes.csv"
    )
    z_offset = pred_tomogram_info.get("z_offset")

    _, max_F1, _ = eval_picking(
        predicted_motl_path=predicted_motl_path,
        gt_motl_path=gt_motl_path,
        z_offset=z_offset,
    )

    F1_results.append({"ds_name": f"TS_{tomo_id}", "F1": max_F1})

# with open(os.path.join("/oliver", "MA", "sample_complexity_eval", "DeePiCt", f"{TRAIN_DS_ID}.json"), 'w') as f:
#     json.dump(F1_results, f, indent=4)
with open(
    os.path.join(
        "/oliver",
        "SAM2",
        "dc_eval",
        "DeePiCt",
        f"{MODEL_NAME}.json",
    ),
    "w",
) as f:
    json.dump(F1_results, f, indent=4)

average_F1 = mean([result["F1"] for result in F1_results])
print(f"Average F1 score: {average_F1}")

# %%

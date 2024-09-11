
# %%
# TODO: export PYTHONPATH="/oliver/MA:$PYTHONPATH"
import sys
sys.path.append('..')
import os
import matplotlib.pyplot as plt
import pdb
import numpy as np
from statistics import mean
import json
from DATA_POSTPROCESSOR.clustering_and_cleaning import cluster_and_clean
from DATA_POSTPROCESSOR.particle_picking_eval import eval_picking
from DATA_POSTPROCESSOR.predict_tomogram import predict

TRAIN_DS_ID = '0001'
TRAIN_DS_TIMESTAMP = '11092024_08:33'
memory_used = False

TOMO_IDs = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']
TOMO_IDs = ['0001', '0002']
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

PREDICTION_DIR = os.path.join(
    '/oliver',
    'MA',
    'PREDICT',
    f"TS_{TRAIN_DS_TIMESTAMP}"
)
PREDICTION_DIR = os.path.join(
    '/media',
    'hdd1',
    'oliver',
    'SAM2',
    'PREDICT',
    f"TS_{TRAIN_DS_TIMESTAMP}"
)

# threshold_list = np.arange(0.6, 0.85, 0.025)
# threshold_list = np.arange(0.85, 0.9, 0.025)
threshold_list = [0.00001]
if __name__ == '__main__':
    F1_results = []

    for tomo_id in TOMO_IDs:
        # Skip the training dataset
        # if TRAIN_DS_ID == tomo_id:
        #     continue
        print(f"\n\n{50*'-'}\nPROCESSING DS: {tomo_id} - memory {'enabled' if memory_used else 'disabled'}\n{50*'-'}\n")
        # Find the relevant info
        pred_tomogram_info = [item for item in pred_tomogram_info_list if item.get("name") == f"TS_{tomo_id}"][0]
        gt_motl_path = os.path.join(
            '/oliver',
            'MA',
            'GT_particle_lists',
            f'TS_{tomo_id}_cyto_ribosomes.csv'
        )
        predict(
            PRED_ID=tomo_id,
            TIMESTAMP=TRAIN_DS_TIMESTAMP,
            PREDICTION_DIR=PREDICTION_DIR,
            target_shape=pred_tomogram_info.get("target_shape"),
            memory_used=memory_used
        )
        

        best_F1_on_th = -1.0 # best F1 score depending on threshold

        for th in threshold_list:
            print(f"THRESHOLD: {th}")
            motl_file_path = cluster_and_clean(
                threshold=th,
                min_cluster_size=0,
                max_cluster_size=None,
                clustering_connectivity=1,
                prediction_dir=PREDICTION_DIR,
                tomogram_name=pred_tomogram_info.get("name")
            )
            if motl_file_path is None:
                continue
            
            # motl_ratio = ( # x, y, z
            #     pred_tomogram_info.get("target_shape")[1] / 256, # x length of tomogram / output of SAM
            #     pred_tomogram_info.get("target_shape")[2] / 256, # y length of tomogram / output of SAM
            # )

            _, max_F1, _ = eval_picking(
                predicted_motl_path=motl_file_path,
                gt_motl_path=gt_motl_path,
                z_offset=pred_tomogram_info.get("z_offset"),
                ratio=(1,1) #motl_ratio
            )
            if max_F1 > best_F1_on_th:
                best_F1_on_th = max_F1
            else: # if the F1 score is decreasing, break the loop
                break
        F1_results.append({
            "ds_name": f"TS_{tomo_id}",
            "F1": float(best_F1_on_th)
        })
    # TODO: change json save path
    with open(os.path.join("/oliver", "MA", "sample_complexity_eval", "SAM2", f"{TRAIN_DS_ID}_{TRAIN_DS_TIMESTAMP}.json"), 'w') as f:
        json.dump(F1_results, f, indent=4)
    
    average_F1 = mean([result["F1"] for result in F1_results])
    print(f"Average F1 score: {average_F1}")
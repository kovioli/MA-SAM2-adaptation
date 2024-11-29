import sys

sys.path.append("..")
import os
import numpy as np
from statistics import mean
import json
from PICKING.clustering_and_cleaning import cluster_and_clean
from PICKING.particle_picking_eval import eval_picking
import argparse
from config import SAVE_DIR, EVAL_DS_LIST, DATA_DIR


def evaluate_particles(
    train_timestamp,
    threshold=0.8,
    min_cluster_size=0,
    max_cluster_size=None,
    save_prediction=True,
):
    """
    Evaluate particle predictions with configurable parameters

    Args:
        train_timestamp (str): Timestamp of the training run
        threshold (float): Threshold value for particle detection (default: 0.8)
        min_cluster_size (int): Minimum cluster size to consider (default: 0)
        max_cluster_size (Optional[int]): Maximum cluster size to consider (default: None)
        save_prediction (bool): Whether to save clustered predictions as MRC files (default: True)
    """
    F1_results = []
    PRED_FOLDER = os.path.join(SAVE_DIR, "PREDICT", train_timestamp)
    print(f"Evaluating predictions from timestamp: {train_timestamp}")
    print(
        f"Using parameters: threshold={threshold}, min_cluster_size={min_cluster_size}, "
        f"max_cluster_size={max_cluster_size}, save_prediction={save_prediction}"
    )

    for DS_INFO in EVAL_DS_LIST:
        tomo_id = DS_INFO["name"]
        print(f"\n{50*'-'}\nProcessing dataset: {tomo_id}\n{50*'-'}\n")

        # Get ground truth path
        gt_motl_path = os.path.join(
            DATA_DIR, "GT_particle_lists", f"{tomo_id}_cyto_ribosomes.csv"
        )

        motl_file_path = cluster_and_clean(
            threshold=threshold,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            clustering_connectivity=1,
            prediction_dir=PRED_FOLDER,
            tomogram_name=tomo_id,
            save_clustered=save_prediction,
        )

        if motl_file_path is not None:
            _, max_F1, _ = eval_picking(
                predicted_motl_path=motl_file_path,
                gt_motl_path=gt_motl_path,
                z_offset=DS_INFO.get("z_offset", 0),
                ratio=(1, 1),
            )
            F1_results.append({"ds_name": tomo_id, "F1": float(max_F1)})

    # Save results
    results_path = os.path.join(PRED_FOLDER, "particle_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(F1_results, f, indent=4)

    average_F1 = mean([result["F1"] for result in F1_results])
    print(f"Average F1 score: {average_F1}")
    print(f"Results saved to: {results_path}")

    return F1_results, average_F1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_timestamp",
        type=str,
        required=True,
        help="Timestamp of the training run (format: DDMMYYYY_HH:MM:SS)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold value for particle detection (default: 0.8)",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=0,
        help="Minimum cluster size to consider (default: 0)",
    )
    parser.add_argument(
        "--max_cluster_size",
        type=int,
        default=None,
        help="Maximum cluster size to consider (default: None)",
    )
    parser.add_argument(
        "--save_prediction",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Save clustered predictions as MRC files (default: True)",
    )

    args = parser.parse_args()

    evaluate_particles(
        args.train_timestamp,
        threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        save_prediction=args.save_prediction,
    )

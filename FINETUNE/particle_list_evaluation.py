import mrcfile as mrc
import numpy as np
import warnings
from pathlib import Path
from scipy.spatial import distance
from skimage.morphology import dilation
import argparse
import json

warnings.simplefilter("ignore")

TEMPLATE = {
    "particle_list": [
        {"name": "3cf3", "f1": 0, "volume": 1123, "mol_weight": 541.74},
        {"name": "1s3x", "f1": 0, "volume": 104.1, "mol_weight": 42.75},
        {"name": "1u6g", "f1": 0, "volume": 498.5, "mol_weight": 238.82},
        {"name": "4cr2", "f1": 0, "volume": 3085, "mol_weight": 1309.28},
        {"name": "1qvr", "f1": 0, "volume": 1255, "mol_weight": 593.36},
        {"name": "3h84", "f1": 0, "volume": 375.3, "mol_weight": 158.08},
        {"name": "2cg9", "f1": 0, "volume": 394.2, "mol_weight": 188.73},
        {"name": "3qm1", "f1": 0, "volume": 139.1, "mol_weight": 62.62},
        {"name": "3gl1", "f1": 0, "volume": 207, "mol_weight": 84.61},
        {"name": "3d2f", "f1": 0, "volume": 521.9, "mol_weight": 236.11},
        {"name": "4d8q", "f1": 0, "volume": 2152, "mol_weight": 1952.74},
        {"name": "1bxn", "f1": 0, "volume": 978.9, "mol_weight": 559.96},
    ]
}


def evaluate_particle(predicted_particles, gt_particles, occupancy_map, particle_type):
    gt_filtered = [p for p in gt_particles if p[0] == particle_type]
    pred_filtered = [p for p in predicted_particles if p[0] == particle_type]

    n_gt_particles = len(gt_filtered)
    if n_gt_particles == 0:
        return 0.0

    n_predicted_particles = len(pred_filtered)
    if n_predicted_particles == 0:
        return 0.0

    found_particles = [[] for _ in range(n_gt_particles)]

    for p_pdb, *coordinates in pred_filtered:
        p_x, p_y, p_z = np.clip(coordinates, (0, 0, 0), (511, 511, 511))
        p_gt_id = int(occupancy_map[p_z, p_y, p_x])

        if p_gt_id == 0:
            continue

        p_gt_pdb, p_gt_x, p_gt_y, p_gt_z = gt_particles[p_gt_id]

        if p_gt_pdb != particle_type:
            continue

        gt_idx = gt_filtered.index((p_gt_pdb, p_gt_x, p_gt_y, p_gt_z))
        found_particles[gt_idx].append(p_pdb)

    n_unique_particles_found = sum([int(len(p) > 0) for p in found_particles])

    recall = n_unique_particles_found / n_gt_particles
    precision = (
        n_unique_particles_found / n_predicted_particles
        if n_predicted_particles > 0
        else 0
    )

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def main():
    parser = argparse.ArgumentParser(description="Particle-wise evaluation script")
    parser.add_argument("-s", "--submission", type=Path, required=True)
    parser.add_argument("-t", "--tomo", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    gt_particles = [("0", 0, 0, 0)]
    with open(args.tomo / "particle_locations.txt", "rU") as f:
        for line in f:
            pdb_id, x, y, z, *_ = line.rstrip("\n").split()
            gt_particles.append((pdb_id, int(x), int(y), int(z) + 156))

    with mrc.open(args.tomo / "occupancy_mask.mrc", permissive=True) as f:
        occupancy_map = dilation(f.data)
        occupancy_map = np.pad(
            occupancy_map,
            ((156, 156), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    predicted_particles = []
    with open(args.submission, "rU") as f:
        for line in f:
            pdb, x, y, z, *_ = line.rstrip("\n").split()
            predicted_particles.append(
                (pdb, int(float(x)), int(float(y)), int(float(z)))
            )

    if (
        min([p[3] for p in predicted_particles]) < 156
        or max([p[3] for p in predicted_particles]) > 356
    ):
        predicted_particles = [
            (p[0], p[1], p[2], p[3] + 156) for p in predicted_particles
        ]

    results = TEMPLATE.copy()
    for particle in results["particle_list"]:
        f1_score = evaluate_particle(
            predicted_particles, gt_particles, occupancy_map, particle["name"]
        )
        particle["f1"] = round(f1_score, 5)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

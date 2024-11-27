# %%
import os
import mrcfile
import numpy as np
from skimage import morphology as morph
from skimage.measure import regionprops_table
import pandas as pd
from skimage.transform import resize
from typing import Optional
import yaml
from scipy.special import expit


def build_tom_motive_list(
    particle_id: str,
    list_of_peak_coordinates: list,
    list_of_peak_scores=None,
    list_of_angles_in_degrees=None,
    list_of_classes=None,
    in_tom_format=False,
) -> pd.DataFrame:
    """
    This function builds a motive list of particles, according to the tom format
    standards:
        The following parameters are stored in the data frame motive_list of
    dimension (NPARTICLES, 20):
       column
          1         : Score Coefficient from localisation algorithm
          2         : x-coordinate in full tomogram
          3         : y-coordinate in full tomogram
          4         : peak number
          5         : running index of tilt series (optional)
          8         : x-coordinate in full tomogram
          9         : y-coordinate in full tomogram
          10        : z-coordinate in full tomogram
          14        : x-shift in subvolume (AFTER rotation of reference)
          15        : y-shift in subvolume
          16        : z-shift in subvolume
          17        : Phi
          18        : Psi
          19        : Theta
          20        : class number
    For more information check tom package documentation (e.g. tom_chooser.m).

    :param list_of_peak_coordinates: list of points in a dataset where a
    particle has been identified. The points in this list hold the format
    np.array([px, py, pz]) where px, py, pz are the indices of the dataset in
    the tom coordinate system.
    :param list_of_peak_scores: list of scores (e.g. cross correlation)
    associated to each point in list_of_peak_coordinates.
    :param list_of_angles_in_degrees: list of Euler angles in degrees,
    np.array([phi, psi, theta]), according to the convention in the
    tom_rotatec.h function: where the rotation is the composition
    rot_z(psi)*rot_x(theta)*rot_z(phi) (again, where xyz are the tom coordinate
    system). For more information on the Euler convention see help from
    datasets.transformations.rotate_ref.
    :param list_of_classes: list of integers of length n_particles representing
    the particle class associated to each particle coordinate.
    :return:
    """
    empty_cell_value = 0  # float('nan')
    if in_tom_format:
        xs, ys, zs = list(np.array(list_of_peak_coordinates, int).transpose())
    else:
        zs, ys, xs = list(np.array(list_of_peak_coordinates, int).transpose())

    n_particles = len(list_of_peak_coordinates)
    tom_indices = list(range(1, 1 + n_particles))
    create_const_list = lambda x: [x for _ in range(n_particles)]

    if list_of_peak_scores is None:
        list_of_peak_scores = create_const_list(empty_cell_value)
    if list_of_angles_in_degrees is None:
        phis, psis, thetas = (
            create_const_list(empty_cell_value),
            create_const_list(empty_cell_value),
            create_const_list(empty_cell_value),
        )
    else:
        phis, psis, thetas = list_of_angles_in_degrees

    if list_of_classes is None:
        list_of_classes = create_const_list(1)
    motive_list_df = pd.DataFrame({})
    # motive_list_df["score"] = list_of_peak_scores
    # motive_list_df["x_"] = xs
    # motive_list_df["y_"] = ys
    # motive_list_df["peak"] = tom_indices
    # motive_list_df["tilt_x"] = empty_cell_value
    # motive_list_df["tilt_y"] = empty_cell_value
    # motive_list_df["tilt_z"] = empty_cell_value
    motive_list_df["pdb"] = [particle_id] * n_particles
    motive_list_df["x"] = xs
    motive_list_df["y"] = ys
    motive_list_df["z"] = zs
    # motive_list_df["empty_1"] = empty_cell_value
    # motive_list_df["empty_2"] = empty_cell_value
    # motive_list_df["empty_3"] = empty_cell_value
    # motive_list_df["x-shift"] = empty_cell_value
    # motive_list_df["y-shift"] = empty_cell_value
    # motive_list_df["z-shift"] = empty_cell_value
    # motive_list_df["phi"] = phis
    # motive_list_df["psi"] = psis
    # motive_list_df["theta"] = thetas
    # motive_list_df["class"] = list_of_classes
    return motive_list_df


def write_mrc_dataset(mrc_path: str, array: np.array, dtype="float32"):
    array = np.array(array, dtype=dtype)
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(array)
    print("Dataset saved in", mrc_path)
    return


def write_tomogram(output_path: str, tomo_data: np.array) -> None:
    ext = os.path.splitext(output_path)[-1].lower()
    if ext == ".mrc":
        write_mrc_dataset(mrc_path=output_path, array=tomo_data)
    return


def read_mrc(path_to_mrc: str, dtype=None):
    with mrcfile.open(path_to_mrc, permissive=True) as f:
        return f.data


def get_clusters_within_size_range(
    dataset: np.array, min_cluster_size: int, max_cluster_size, connectivity=1
):
    if max_cluster_size is None:
        max_cluster_size = np.inf
    assert min_cluster_size <= max_cluster_size

    # find clusters and label them
    labeled_clusters, num = morph.label(
        dataset, background=0, return_num=True, connectivity=connectivity
    )
    labels_list, cluster_size = np.unique(labeled_clusters, return_counts=True)

    # Check if we have any non-background clusters
    if len(labels_list) <= 1:  # Only background cluster exists
        return labeled_clusters, [], []

    # excluding the background cluster
    labels_list, cluster_size = labels_list[1:], cluster_size[1:]

    # Create mask for clusters within size range
    size_mask = (cluster_size > min_cluster_size) & (cluster_size <= max_cluster_size)
    labels_list_within_range = labels_list[size_mask]
    cluster_size_within_range = list(cluster_size[size_mask])

    return labeled_clusters, labels_list_within_range, cluster_size_within_range


def get_cluster_centroids(
    dataset: np.array, min_cluster_size, max_cluster_size, connectivity=1
) -> tuple:
    labeled_clusters, labels_list_within_range, cluster_size_within_range = (
        get_clusters_within_size_range(
            dataset=dataset,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            connectivity=connectivity,
        )
    )

    # If no clusters found, return empty results
    if len(labels_list_within_range) == 0:
        return np.zeros_like(dataset), [], []

    # Create binary mask of the labels within range
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    clusters_map_in_range[np.isin(labeled_clusters, labels_list_within_range)] = 1

    # Find out the centroids of the labels within range
    filtered_labeled_clusters = (labeled_clusters * clusters_map_in_range).astype(
        np.int32
    )
    props = regionprops_table(
        filtered_labeled_clusters, properties=("label", "centroid")
    )

    centroids_list = [
        np.rint([x, y, z])
        for _, x, y, z in sorted(
            zip(
                props["label"].tolist(),
                props["centroid-0"].tolist(),
                props["centroid-1"].tolist(),
                props["centroid-2"].tolist(),
            )
        )
    ]
    return clusters_map_in_range, centroids_list, cluster_size_within_range


def save_cluster_config_yaml(
    save_dir: str,
    threshold: float,
    min_cluster_size: int,
    max_cluster_size: Optional[int],
    clustering_connectivity: int,
):
    config = {
        "threshold": threshold,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "clustering_connectivity": clustering_connectivity,
    }
    with open(os.path.join(save_dir, "cluster_config.yaml"), "w") as f:
        yaml.dump(config, f)


def sigmoid(x):
    return expit(x)


def cluster_and_clean(
    threshold: float,
    min_cluster_size: int,
    max_cluster_size: Optional[int],
    clustering_connectivity: int,
    prediction_dir: str,
    prediction_file: str,
    particle_id: str,
    output_file: str,
) -> str:
    """
    Process neural network predictions to detect particles in tomograms through thresholding
    and clustering. Generates a motive list compatible with the TOM format.

    Args:
        threshold: Float value for binary thresholding of predictions
        min_cluster_size: Minimum size of valid particle clusters
        max_cluster_size: Maximum size of valid particle clusters
        clustering_connectivity: Connectivity parameter for clustering
        prediction_dir: Directory containing prediction files
        prediction_file: Name of the prediction file to process
        particle_id: Particle ID from the dataset
        output_file: Path to the output motive list file

    Returns:
        str: Path to generated motive list file, or None if no particles found
    """
    # paths
    original_pred_path = os.path.join(prediction_dir, prediction_file)

    # Add debug prints
    print(f"Processing particle {particle_id}")
    print(f"Min cluster size: {min_cluster_size}")
    print(f"Reading prediction from: {original_pred_path}")

    # Check if file exists
    if not os.path.exists(original_pred_path):
        print(f"Warning: File not found - {original_pred_path}")
        return

    # load ds
    original_prediction_ds = read_mrc(original_pred_path)
    print(f"Loaded prediction shape: {original_prediction_ds.shape}")
    print(
        f"Prediction range: [{original_prediction_ds.min()}, {original_prediction_ds.max()}]"
    )

    # get & save clustered ds
    clusters_labeled_by_size, centroids_list, cluster_size_list = get_cluster_centroids(
        dataset=original_prediction_ds,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        connectivity=clustering_connectivity,
    )

    print(f"Found {len(centroids_list)} clusters")

    if len(centroids_list) > 0:
        motive_list_df = build_tom_motive_list(
            list_of_peak_coordinates=centroids_list,
            list_of_peak_scores=cluster_size_list,
            in_tom_format=False,
            particle_id=particle_id,
        )

        with open(output_file, "a") as f:
            for _, row in motive_list_df.iterrows():
                line = (
                    f"{particle_id} {int(row['x'])} {int(row['y'])} {int(row['z'])}\n"
                )
                f.write(line)
        print(
            f"Appended {len(centroids_list)} particles for {particle_id} to {output_file}"
        )
    else:
        print(f"No centroids found for {particle_id}")


# %%
particle_id_mapping = [
    {
        "particle_id": 1,
        "particle_name": "3cf3",
        "volume": 1123,
        "mol_weight": 541,
        "weight_group": "medium",
    },
    {
        "particle_id": 2,
        "particle_name": "1s3x",
        "volume": 104,
        "mol_weight": 42,
        "weight_group": "small",
    },
    {
        "particle_id": 3,
        "particle_name": "1u6g",
        "volume": 498,
        "mol_weight": 238,
        "weight_group": "medium",
    },
    {
        "particle_id": 4,
        "particle_name": "4cr2",
        "volume": 3085,
        "mol_weight": 1309,
        "weight_group": "large",
    },
    {
        "particle_id": 5,
        "particle_name": "1qvr",
        "volume": 1255,
        "mol_weight": 593,
        "weight_group": "medium",
    },
    {
        "particle_id": 6,
        "particle_name": "3h84",
        "volume": 375,
        "mol_weight": 158,
        "weight_group": "small",
    },
    {
        "particle_id": 7,
        "particle_name": "2cg9",
        "volume": 394,
        "mol_weight": 188,
        "weight_group": "small",
    },
    {
        "particle_id": 8,
        "particle_name": "3qm1",
        "volume": 139,
        "mol_weight": 62,
        "weight_group": "small",
    },
    {
        "particle_id": 9,
        "particle_name": "3gl1",
        "volume": 207,
        "mol_weight": 84,
        "weight_group": "small",
    },
    {
        "particle_id": 10,
        "particle_name": "3d2f",
        "volume": 521,
        "mol_weight": 236,
        "weight_group": "medium",
    },
    {
        "particle_id": 11,
        "particle_name": "4d8q",
        "volume": 2152,
        "mol_weight": 1952,
        "weight_group": "large",
    },
    {
        "particle_id": 12,
        "particle_name": "1bxn",
        "volume": 978,
        "mol_weight": 559,
        "weight_group": "medium",
    },
]

# %%

mol_weight_ratio = 0.55
volume_ratio = 0.2
ds_name = "grandmodel_1ds_large"
output_file = f"/.../SAM2/PARTICLE_COORDS/{ds_name}.txt"

for p_map in particle_id_mapping:
    cluster_and_clean(
        threshold=0.5,
        min_cluster_size=int(...),
        max_cluster_size=None,
        clustering_connectivity=1,
        prediction_dir="...",
        prediction_file="...",
        particle_id=p_map["particle_name"],
        output_file=output_file,
    )

print("All particle data has been written to", "...")

# %%
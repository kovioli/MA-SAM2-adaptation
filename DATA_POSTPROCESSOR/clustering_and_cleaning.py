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
    motive_list_df["score"] = list_of_peak_scores
    motive_list_df["x_"] = xs
    motive_list_df["y_"] = ys
    motive_list_df["peak"] = tom_indices
    motive_list_df["tilt_x"] = empty_cell_value
    motive_list_df["tilt_y"] = empty_cell_value
    motive_list_df["tilt_z"] = empty_cell_value
    motive_list_df["x"] = xs
    motive_list_df["y"] = ys
    motive_list_df["z"] = zs
    motive_list_df["empty_1"] = empty_cell_value
    motive_list_df["empty_2"] = empty_cell_value
    motive_list_df["empty_3"] = empty_cell_value
    motive_list_df["x-shift"] = empty_cell_value
    motive_list_df["y-shift"] = empty_cell_value
    motive_list_df["z-shift"] = empty_cell_value
    motive_list_df["phi"] = phis
    motive_list_df["psi"] = psis
    motive_list_df["theta"] = thetas
    motive_list_df["class"] = list_of_classes
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

    # find clusters and label them. Each cluster is assigned a unique integer from 0 to num_clusters-1
    # for example: [... 0   0   0   0   0   0   0   0   0 592 592 592 592 592   0   0   0   0 ...]
    labeled_clusters, num = morph.label(
        dataset, background=0, return_num=True, connectivity=connectivity
    )
    labels_list, cluster_size = np.unique(labeled_clusters, return_counts=True)
    # excluding the background cluster: (e.g. where labels_list is zero)
    labels_list, cluster_size = labels_list[1:], cluster_size[1:]
    maximum = np.max(cluster_size)
    print("number of clusters before size filtering = ", len(labels_list))
    # print("size range before size filtering: ", np.min(cluster_size), "to",
    #       maximum)
    labels_list_within_range = labels_list[
        (cluster_size > min_cluster_size) & (cluster_size <= max_cluster_size)
    ]
    cluster_size_within_range = list(
        cluster_size[
            (cluster_size > min_cluster_size) & (cluster_size <= max_cluster_size)
        ]
    )
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
    tomogram_name: str,
    epoch: int = None,
) -> str:
    # paths
    if epoch is not None:
        original_prediction_path = os.path.join(
            prediction_dir, f"{tomogram_name}_epoch_{epoch:03d}.mrc"
        )
    else:
        original_prediction_path = os.path.join(prediction_dir, f"{tomogram_name}.mrc")

    clustered_prediction_path = original_prediction_path.replace(".mrc", "_cluster.mrc")

    # save cluster config
    save_cluster_config_yaml(
        save_dir=prediction_dir,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        clustering_connectivity=clustering_connectivity,
    )

    # load ds
    original_prediction_ds = read_mrc(original_prediction_path)

    # normalize
    norm_prediction_ds = (original_prediction_ds - original_prediction_ds.min()) / (
        original_prediction_ds.max() - original_prediction_ds.min()
    )
    # norm_prediction_ds = sigmoid(original_prediction_ds)
    # threshold
    prediction_ds_thr = 1 * (norm_prediction_ds > threshold)

    # get & save clustered ds
    clusters_labeled_by_size, centroids_list, cluster_size_list = get_cluster_centroids(
        dataset=prediction_ds_thr,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        connectivity=clustering_connectivity,
    )
    # clustered_prediction = 1 * (clusters_labeled_by_size > 0)
    # write_tomogram(output_path=clustered_prediction_path, tomo_data=clustered_prediction)

    # create motl list
    motl_name = f"motl_{str(len(centroids_list))}.csv"
    motl_file_path = os.path.join(prediction_dir, motl_name)

    if len(centroids_list) > 0:
        motive_list_df = build_tom_motive_list(
            list_of_peak_coordinates=centroids_list,
            list_of_peak_scores=cluster_size_list,
            in_tom_format=False,
        )
        motive_list_df.to_csv(motl_file_path, index=False, header=False)
        print(f"Saved motl to {motl_file_path}")
        return motl_file_path
    else:
        print("No centroids found in the dataset")
        return None

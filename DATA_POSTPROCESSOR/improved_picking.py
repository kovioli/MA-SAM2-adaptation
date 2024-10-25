# %%
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN
from scipy.special import expit  # numerically stable sigmoid
from skimage.morphology import local_maxima


def preprocess_image(original_prediction_ds, threshold):
    # Normalize and threshold
    norm_prediction_ds = expit(original_prediction_ds)
    binary_image = norm_prediction_ds > threshold
    return binary_image


# def preprocess_image(original_prediction_ds, threshold):
#     # Normalize and threshold
#     norm_prediction_ds = (original_prediction_ds - original_prediction_ds.min()) / (
#         original_prediction_ds.max() - original_prediction_ds.min()
#     )
#     binary_image = norm_prediction_ds > threshold
#     return binary_image


def apply_morphology(binary_image):
    from scipy.ndimage import binary_closing

    # Apply morphological closing
    closed_image = binary_closing(binary_image, structure=np.ones((3, 3, 3)))
    return closed_image


def separate_particles(binary_image):
    # Compute distance transform
    distance = distance_transform_edt(binary_image)

    # Find local maxima
    local_maxi = local_maxima(distance)

    # Markers for watershed
    markers, _ = label(local_maxi)

    # Apply watershed
    labels = watershed(-distance, markers=markers, mask=binary_image)
    return labels


def cluster_fragments(labels):
    # Extract coordinates of labeled regions
    coords = np.column_stack(np.nonzero(labels))
    # Apply DBSCAN
    clustering = DBSCAN(eps=2, min_samples=5).fit(coords)
    # Create new labels
    clustered_labels = np.zeros_like(labels)
    clustered_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = (
        clustering.labels_ + 1
    )  # +1 to avoid zero label
    return clustered_labels


def get_centroids(clustered_labels):
    from skimage.measure import regionprops

    props = regionprops(clustered_labels)
    centroids = [prop.centroid for prop in props if prop.label != 0]
    return centroids


# %%

# Usage
binary_image = preprocess_image(original_prediction_ds, threshold)
closed_image = apply_morphology(binary_image)
separated_labels = separate_particles(closed_image)
clustered_labels = cluster_fragments(separated_labels)
centroids = get_centroids(clustered_labels)

# %%

import os
import pandas as pd
import numpy as np

def read_motl_from_csv(path_to_csv_motl: str):
    """
    Output: array whose first entries are the rows of the motif list
    Usage example:
    motl = read_motl_from_csv(path_to_csv_motl)
    list_of_max=[row[0] for row in motl]
    """
    if os.stat(path_to_csv_motl).st_size == 0:
        motl_df = pd.DataFrame({"x": [], "y": [], "z": [], "score": []})
    else:
        motl_df = pd.read_csv(path_to_csv_motl)
        if motl_df.shape[1] == 20:
            motl_df = pd.read_csv(path_to_csv_motl,
                                  names=['score', 'x_', 'y_', 'peak', 'tilt_x', 'tilt_y', 'tilt_z',
                                         'x', 'y', 'z', 'empty_1', 'empty_2', 'empty_3', 'x-shift',
                                         'y-shift', 'z-shift', 'phi', 'psi', 'theta', 'class'])
        elif motl_df.shape[1] == 3:
            motl_df = pd.read_csv(path_to_csv_motl, names=["x", "y", "z"])
            motl_df["score"] = np.nan
    return motl_df

def extract_motl_coordinates_and_score_values(
        motl: list,
        z_offset: int,
        ratio: tuple
    ) -> tuple:
    coordinates = list(motl[["x", "y", "z"]].values)
    for coord in coordinates:
        coord[2] = coord[2] + z_offset
        coord[0] = coord[0] * ratio[0]
        coord[1] = coord[1] * ratio[1]
    score_values = list(motl["score"].values)
    return score_values, coordinates

def read_motl_coordinates_and_values(
        path_to_motl: str,
        z_offset: int,
        ratio: tuple
    ) -> tuple:
    motl = read_motl_from_csv(path_to_motl)
    motl_values, motl_coords = extract_motl_coordinates_and_score_values(motl, z_offset, ratio)
    motl_coords = np.array(motl_coords, dtype=int)
    return motl_values, motl_coords

def get_clean_points_close2point(point, clean, radius):
    close_to_point = []
    distances = []
    for clean_p in clean:
        dist = np.linalg.norm(clean_p - point)
        # print(dist) TODO: distance ALWAYS greater radius -> likely false calculation
        # IT WILL NEVER WORK ON CLEANED DATA! -> has to predict on EMPIAR instead of EMPAER_clean!
        if dist <= radius:
            close_to_point.append(clean_p)
            distances.append(dist)
    close_to_point = [tuple(p) for p in close_to_point]
    return close_to_point, distances

def precision_recall_calculator(predicted_coordinates,
                                value_predicted: list,
                                true_coordinates,
                                radius: float):
    true_coordinates = list(true_coordinates)
    predicted_coordinates = list(predicted_coordinates)
    detected_true = list()
    predicted_true_positives = list()
    predicted_redundant = list()
    value_predicted_true_positives = list()
    value_predicted_redundant = list()
    precision = list()
    recall = list()
    total_true_points = len(true_coordinates)
    assert total_true_points > 0, "one empty list here!"
    if len(predicted_coordinates) == 0:
        print("No predicted points")
        precision = []
        recall = []
        detected_true = []
        predicted_true_positives = []
        predicted_false_positives = []
        value_predicted_true_positives = []
        value_predicted_false_positives = []
        predicted_redundant = []
        value_predicted_redundant = []
        false_negatives = total_true_points
    else:
        predicted_false_positives = list()
        value_predicted_false_positives = list()
        for value, point in zip(value_predicted, predicted_coordinates):
            close_to_point, distances = get_clean_points_close2point(
                point,
                true_coordinates,
                radius
            )
            if len(close_to_point) > 0:
                flag = "true_positive_candidate"
                flag_tmp = "not_redundant_yet"
                for dist, clean_p in sorted(zip(distances, close_to_point)):
                    if flag == "true_positive_candidate":
                        if tuple(clean_p) not in detected_true:
                            detected_true.append(tuple(clean_p))
                            flag = "true_positive"
                        else:
                            flag_tmp = "redundant_candidate"
                    # else:
                    # print(point, "is already flagged as true positive")
                if flag == "true_positive":
                    predicted_true_positives.append(tuple(point))
                    value_predicted_true_positives.append(value)
                elif flag == "true_positive_candidate" and \
                        flag_tmp == "redundant_candidate":
                    predicted_redundant.append(tuple(point))
                    value_predicted_redundant.append(value)
                else:
                    print("This should never happen!")
            else:
                predicted_false_positives.append(tuple(point))
                value_predicted_false_positives.append(value)
            true_positives_total = len(predicted_true_positives)
            false_positives_total = len(predicted_false_positives)
            total_current_predicted_points = true_positives_total + \
                                             false_positives_total
            precision.append(true_positives_total / total_current_predicted_points)
            recall.append(true_positives_total)
        false_negatives = [point for point in true_coordinates if tuple(point) not in detected_true]
        N_inv = 1 / total_true_points
        recall = np.array(recall) * N_inv
        recall = list(recall)
    return precision, recall, detected_true, predicted_true_positives, \
           predicted_false_positives, value_predicted_true_positives, \
           value_predicted_false_positives, false_negatives, predicted_redundant, \
           value_predicted_redundant

def f1_score_calculator(precision: list, recall: list):
    f1_score = []
    if len(precision) == 0:
        print("No precision and recall")
        f1_score = [0]
    else:
        for p, r in zip(precision, recall):
            if p + r != 0:
                f1_score.append(2 * p * r / float(p + r))
            else:
                f1_score.append(0)
    return f1_score

def get_max_F1(F1_score: list):
    if len(F1_score) > 0:
        max_F1 = np.max(F1_score)
        optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
    else:
        max_F1 = 0
        optimal_peak_number = np.nan
    return max_F1, optimal_peak_number

def quadrature_calculator(x_points: list, y_points: list) -> float:
    """
    This function computes an approximate value of the integral of a real
    function f in an interval, using the trapezoidal rule.

    Input:
    x_points: is a list of points in the x axis (not necessarily ordered)
    y_points: is a list of points, such that y_points[n] = f(x_points[n]) for
    each n.
    """
    # sorted_y = [p for _, p in sorted(zip(x_points, y_points))]
    sorted_y = [p for _, p in
                sorted(list(zip(x_points, y_points)), key=lambda x: x[0])]
    n = len(y_points)
    sorted_x = sorted(x_points)

    trapezoidal_rule = [
        0.5 * (sorted_x[n + 1] - sorted_x[n]) * (sorted_y[n + 1] + sorted_y[n])
        for n in range(n - 1)]

    return float(np.sum(trapezoidal_rule))

def pr_auc_score(precision: list, recall: list) -> float:
    """
    This function computes an approximate value to the area
    under the precision-recall (PR) curve.
    """
    return quadrature_calculator(recall, precision)


def eval_picking(
        predicted_motl_path: str,
        gt_motl_path: str,
        z_offset: int,
        pr_tolerance_radius: int = 10,
        ratio: tuple = (1, 1)
    ):
    # path_to_motl_ground_truth = os.path.join('/oliver', 'MA', 'PREDICT', f'{tomogram_name}_cyto_ribosomes.csv')
    predicted_values, predicted_coordinates = read_motl_coordinates_and_values(
        path_to_motl=predicted_motl_path,
        z_offset=z_offset,
        ratio=ratio # pass ratio to match original coordinate range
    )
    true_values, true_coordinates = read_motl_coordinates_and_values(
        path_to_motl=gt_motl_path,
        z_offset=0,
        ratio=(1, 1) # no need to change
    )
    unique_peaks_number = len(predicted_values)
    predicted_coordinates = np.array(predicted_coordinates)

    prec, recall, tp_true, tp_pred, fp_pred, tp_pred_scores, fp_pred_scores, fn, *_ = \
        precision_recall_calculator(
            predicted_coordinates=predicted_coordinates,
            value_predicted=predicted_values,
            true_coordinates=true_coordinates,
            radius=pr_tolerance_radius
    )

    F1_score = f1_score_calculator(prec, recall)
    auPRC = pr_auc_score(precision=prec, recall=recall)
    max_F1, optimal_peak_number = get_max_F1(F1_score)
    print(f"MOTL PATH: {predicted_motl_path}\n\tauPRC = {auPRC}\n\tmax_F1 = {max_F1}\n\tfinal_F1 = {F1_score[-1]}")
    return (auPRC, max_F1, F1_score[-1])

# if __name__ == "__main__":
#     motl_path = os.path.join(
#         "/oliver",
#         "DeePiCt",
#         "PREDICT",
#         "predictions",
#         "ribo_model_best",
#         "EMPIAR_0009",
#         "ribo",
#         "motl_2933.csv"
#     )
#     auprc, max_F1, last_F1 = eval_picking(
#         predicted_motl_path=motl_path,
#         z_offset=120,
#         tomogram_name="TS_0009"
#     )
    
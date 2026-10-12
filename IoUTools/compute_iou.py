#!/usr/bin/python3
import math
from scipy.optimize import linear_sum_assignment

# (ax1, ay1, ax2, ay2)
# (bx1, by1, bx2, by2)
# les coordonnées du rectangle d’intersection sont :
# ix1 = max(ax1,bx1)
# iy1 = max(ay1,by1)
# ix2 = min(ax2,bx2)
# iy2 = min(ay2,by2)
# Puis :
# iw = max(0,ix2-ix1)
# ih = max(0,iy2-iy1)

# box = (x1, y1, x2, y2)
# Enumerated type?
IDX_DET_CLASS = 0
IDX_DET_CONFIDENCE = 1
IDX_DET_BBOX = 2

X1 = 0
X2 = 2
Y1 = 1
Y2 = 3
def compute_iou(box_a, box_b):
#    ix1 = max (box_a.x1, box_b.x1)
#    iy1 = max (box_a.y1, box_b.y1)
#    ix2 = min (box_a.x2, box_b.x2)
#    iy2 = min (box_a.y2, box_b.y2)
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    if ((ax1 >= ax2) or (ay1 >= ay2)):
            raise ValueError("Invalid box 1")
    if ((bx1 >= bx2) or (by1 >= by2)):
            raise ValueError("Invalid box 2")
    ix1 = max (ax1, bx1)
    iy1 = max (ay1, by1)
    ix2 = min (ax2, bx2)
    iy2 = min (ay2, by2)
    iw = max(0,ix2-ix1)
    ih = max(0,iy2-iy1)
    area_inter = iw * ih
    if (0 == area_inter):
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - area_inter
    if (0 == union_area):
        raise RuntimeError("division by 0")
    return area_inter / union_area

def compute_iou_matrix(reference_boxes, detected_boxes):
    retVal = []
    for ref_box in reference_boxes:
        row = []
        for detect_box in detected_boxes:
            row.append(compute_iou(ref_box, detect_box))
        retVal.append(row)
    return retVal

def greedy_match_candidates(candidates, threshold=0.5):
    """
    candidates: iterable of (score, ref_idx, det_idx)
    """

    ordered_candidates = sorted(
        candidates,
        key=lambda x: (-x[0], x[1], x[2])
    )

    matched_refs = set()
    matched_dets = set()
    matches = []

    for score, ref_idx, det_idx in ordered_candidates:
        if ref_idx in matched_refs or det_idx in matched_dets:
            continue

        matches.append((score, ref_idx, det_idx))
        matched_refs.add(ref_idx)
        matched_dets.add(det_idx)
    return matches

def match_from_score_matrix(matrix, threshold=0.5):
    candidates = [
        (score, ref_idx, det_idx)
        for ref_idx, row in enumerate(matrix)
        for det_idx, score in enumerate(row)
        if score >= threshold
    ]
#    print("##### DEBUG #######")
#    print("DEBUG match_from_score_matrix - candidates:",candidates)
    return greedy_match_candidates(candidates, threshold)

def match_boxes(reference_boxes, detected_boxes, threshold=0.5):
    candidates = []

    for ref_idx, ref in enumerate(reference_boxes):
        for det_idx, det in enumerate(detected_boxes):
            score = compute_iou(ref, det)

            if score >= threshold:
                candidates.append((score, ref_idx, det_idx))

    return greedy_match_candidates(candidates, threshold)


def maximum_cardinality_match_from_score_matrix(matrix, threshold=0.5):
    number_of_references = len(matrix)
    if (0 == number_of_references):
        return 0, []
    # Assume: Matrix is rectangular
    number_of_dets = len(matrix[0])
    det_to_ref = [-1] * number_of_dets

    def new_match(ref_idx, used_detections):
        for det_idx in range(number_of_dets):
            if matrix[ref_idx][det_idx] < threshold or det_idx in used_detections:
                continue

            used_detections.add(det_idx)
            used_ref_for_det = det_to_ref[det_idx]

            if (-1 == used_ref_for_det) or new_match(used_ref_for_det, used_detections):
                det_to_ref[det_idx] = ref_idx
                return True
        return False

    taille_couplage = 0
    for idx_ref in range(number_of_references):
        used_detections = set()
        if new_match(idx_ref, used_detections):
            taille_couplage += 1
    return taille_couplage, det_to_ref

def build_reward_matrix(matrix, threshold):
    R = len(matrix)
    if R == 0:
        return []
    D = len(matrix[0])
    if any(len(row) != D for row in matrix):
        raise ValueError("Score matrix must be rectangular")
    B = min(R, D) + 1

    reward_matrix = []
    for idx_ref in range(len(matrix)):
        row = []
        for idx_det in range(len(matrix[idx_ref])):
            score = matrix[idx_ref][idx_det]
            if score >= threshold:
                row.append(B + score)
            else:
                row.append(0.0)
        row.extend([0.0] * R)
        reward_matrix.append(row)
    return reward_matrix

def optimal_match_from_score_matrix(matrix, threshold):
    """
    Compute an optimal one-to-one assignment for scores in [0, 1].

    Valid associations are given a cardinality bonus:
        reward = B + score
    where:
        B = min(R, D) + 1

    This ensures that maximizing the total reward first maximizes
    the number of valid matches, and then, among assignments with
    the same cardinality, maximizes the total score.

    Associations with scores below the threshold are not retained.
    """
    if len(matrix) == 0:
        return []

    reward_matrix = build_reward_matrix(matrix, threshold)
    D = len(matrix[0])

    row_indices, col_indices = linear_sum_assignment(
        reward_matrix,
        maximize=True
    )

    matches = []

    for row_idx, col_idx in zip(row_indices, col_indices):
        if col_idx >= D:
            continue

        score = matrix[row_idx][col_idx]

        if score >= threshold:
            matches.append((score, row_idx, col_idx))

    return matches

def compute_detection_metrics(
    number_of_references,
    number_of_detections,
    matches
):
    """
    Compute detection metrics
    """
    R = number_of_references
    D = number_of_detections
    M = len(matches)
    if R < 0 or D < 0:
        raise ValueError("Number of references and detections must be non-negative")
    if M > min(R, D):
        raise ValueError("Number of matches cannot exceed min(references, detections)")

    TP = M
    FP = D - M
    FN = R - M
    no_detection = (D == 0)
    no_reference = (R == 0)
    no_reference_no_det = no_detection and no_reference

    precision = math.nan if no_detection else TP / D
    recall = math.nan if no_reference else TP / R
    f1 = math.nan if no_reference_no_det else 2*TP / (2*TP+FP+FN)

    return {
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

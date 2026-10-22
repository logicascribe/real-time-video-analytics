#!/usr/bin/python3
import math
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from typing import TypeAlias

BBox: TypeAlias = tuple[float, float, float, float]
BoxesArray: TypeAlias = NDArray[np.float32]
# Bounding-box convention:
#   box = (x1, y1, x2, y2)
#   x1 < x2, y1 < y2
#   coordinates are floating-point image coordinates
#
# Scalar representation:
#   BBox = tuple[float, float, float, float]
#
# Batch representation:
#   np.ndarray, shape (N, 4), dtype float32
#
# Empty batch:
#   shape (0, 4)

"""Array of bounding boxes with shape (N, 4).

Each row contains (x1, y1, x2, y2).
The empty representation has shape (0, 4).
"""

"""Pairwise IoU matrix with shape (R, D).

R = number of reference boxes.
D = number of detection boxes.
"""
IoUMatrix: TypeAlias = NDArray[np.float32]
RewardMatrix: TypeAlias = NDArray[np.float32]
ScoreMatrix: TypeAlias = NDArray[np.float32]

from typing import Tuple
BBox = tuple[float, float, float, float]
box : BBox = (10.0, 20.0, 30.0, 50.0)

@dataclass(frozen=True)
class Detection:
    bbox: BBox
    class_name: str
    confidence: float

@dataclass
class Track:
    id: int
    bbox: BBox
    class_name: str
    confidence: float
    # temporal state
    # history: deque[bool]
    # status: TrackingStatus
    # trajectory: list[Point]
    # last_detection: Detection | None = None

# def compute_iou(box_a: BBox, box_b: BBox) -> float:

RECT_LENGTH = 4


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



#if np.any(arr[:, 0] >= arr[:, 2]):
#    ...
#if np.any(arr[:, 1] >= arr[:, 3]):
#    ...
def _as_boxes_array(
    boxes: list[BBox] | BoxesArray,
    ) -> BoxesArray:
    raw = np.asarray(boxes)

    #boxAr : BoxesArray
    #raise ValueError("DBG_ValueError")
    #return boxAr

    # Python empty sequence: normalize to canonical empty BoxesArray.
    if raw.shape == (0,):
        return np.empty((0, RECT_LENGTH), dtype=np.float32)

    # Already normalized empty array: valid.
    if raw.shape == (0, RECT_LENGTH):
        return raw.astype(np.float32, copy=False)

    if raw.ndim != 2 or raw.shape[1] != RECT_LENGTH:
        raise ValueError(
            f"Boxes must have shape (N, {RECT_LENGTH}), "
            f"got {raw.shape}"
        )

    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(
            f"Box coordinates must be numeric, got dtype {raw.dtype}"
        )

    box_array = raw.astype(np.float32, copy=False)

    if not np.all(np.isfinite(box_array)):
        raise ValueError("Box coordinates must be finite")

    invalid = (
        (box_array[:, 0] >= box_array[:, 2])
        | (box_array[:, 1] >= box_array[:, 3])
    )

    if np.any(invalid):
        idx = np.flatnonzero(invalid)[0]
        raise ValueError(
            f"Invalid box at index {idx}: {box_array[idx]}"
        )

    return box_array

def compute_iou_matrix(
    reference_boxes: BoxesArray,
    detected_boxes: BoxesArray,
) -> IoUMatrix:

    references = _as_boxes_array(reference_boxes)
    detections = _as_boxes_array(detected_boxes)

    # Ref shape: (M, 4) -> converti en (M, 1, 4)
    # Det shape: (N, 4) -> converti en (1, N, 4)
    ref_expanded = references[:, np.newaxis, :]
    det_expanded = detections[np.newaxis, :, :]

    # Extraction des coordonnées des coins (x1, y1, x2, y2)
    # Le broadcasting applique la comparaison (M, 1) vs (1, N) -> (M, N)
    inter_x1 = np.maximum(ref_expanded[..., 0], det_expanded[..., 0])
    inter_y1 = np.maximum(ref_expanded[..., 1], det_expanded[..., 1])
    inter_x2 = np.minimum(ref_expanded[..., 2], det_expanded[..., 2])
    inter_y2 = np.minimum(ref_expanded[..., 3], det_expanded[..., 3])

    # Largeur et hauteur de la zone d'intersection
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Calcul des surfaces (aires) individuelles
    area1 = (
        (references[:, 2] - references[:, 0])
        * (references[:, 3] - references[:, 1])
        )

    area2 = (
        (detections[:, 2] - detections[:, 0])
        * (detections[:, 3] - detections[:, 1])
        )

    # Alignement des shapes pour l'union via newaxis
    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area

    # Calcul final de l'Intersection over Union (IoU)
    # Évite la division par zéro si l'union est nulle
    return np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area != 0)


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


def _old_optimal_match_from_score_matrix(matrix, threshold):
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

################################################
# "optimal_match_from_score_matrix" from Numpy #
################################################
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
    scores = _as_score_matrix(matrix)

    R, D = scores.shape

    if R == 0 or D == 0:
        return []

    reward_matrix = build_reward_matrix(scores, threshold)

    row_indices, col_indices = linear_sum_assignment(
        reward_matrix,
        maximize=True,
    )

    # Keep only assignments to real detection columns.
    real_mask = col_indices < D

    row_indices = row_indices[real_mask]
    col_indices = col_indices[real_mask]

    matched_scores = scores[row_indices, col_indices]

    # Defensive filtering according to the public contract.
    valid_mask = matched_scores >= threshold

    row_indices = row_indices[valid_mask]
    col_indices = col_indices[valid_mask]
    matched_scores = matched_scores[valid_mask]

    matches = [
        (float(score), int(r), int(c))
        for score, r, c in zip(
            matched_scores,
            row_indices,
            col_indices,
        )
    ]

    return matches


######################################
# "build_reward_matrix()" from Numpy #
######################################
SCORE_MAT_NDIM = 2
def _as_score_matrix(matrix) -> IoUMatrix:
    raw = np.asarray(matrix)

    # Python empty sequence: normalize to canonical empty BoxesArray.
    if raw.shape == (0,):
        return np.empty((0, 0), dtype=np.float32)

    if SCORE_MAT_NDIM != raw.ndim:
        raise ValueError(
            f"Score matrix must have shape (N, M), "
            f"got {raw.shape}"
        )

    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(
            f"Score matrix coordinates must be numeric, got dtype {raw.dtype}"
        )

    score_array = raw.astype(np.float32, copy=False)

    if not np.all(np.isfinite(score_array)):
        raise ValueError("Score matrix values must be finite")

    if np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ValueError("Score matrix values must be in [0, 1]")

    return score_array

def build_reward_matrix(matrix, threshold):
    scores = _as_score_matrix(matrix)
    bool_mask = scores >= threshold
    R, D = scores.shape
    B = min(R, D) + 1
    reward_scores = np.where(
        bool_mask,
        B + scores,
        0.0,
        )
    dummy = np.zeros((R, R), dtype=np.float32)
    reward_matrix = np.concatenate(
        (reward_scores, dummy),
        axis=1,
        )
    return reward_matrix

############################################
# "build_reward_matrix()" from Numpy - End #
############################################


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

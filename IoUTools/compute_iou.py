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
    """
    build a reward matrix from matrix
    """
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
    return the optimal match from score matrix
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
        raise ValueError(...)
    if M > min(R, D):
        raise ValueError(...)

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


if __name__ == "__main__":
    print ("###########################")
    print ("# Testing IoU computation #")
    print ("###########################")

    box_a = (10, 10, 30, 30)
    box_b = (20, 20, 40, 40)
    IoU_1 = compute_iou(box_a, box_b)
    print ("IoU_1 = ", IoU_1)

    box_a1 = (0, 0, 100, 100)
    box_b1 = (50, 0, 150, 100)
    IoU_2 = compute_iou(box_a1, box_b1)
    print ("IoU_2 = ", IoU_2)

    donnees = [
    # Identical
    ((10, 10, 30, 30),
    (10, 10, 30, 30),
    1.0),

    # No overlap
    ((10, 10, 20, 20),
    (30, 30, 40, 40),
    0.0),

    # Partial overlap
    ((10, 10, 30, 30),
    (20, 20, 40, 40),
    100 / 700),

    # One inside another
    ((100, 100, 200, 300),
    (110, 120, 190, 280),
    0.64)
    ]

    for A, B, expected in donnees:
        IoU = compute_iou(A, B)
        print ("A", A)
        print ("B", B)
        print ("expected", expected)
        print ("IoU", IoU)
        print ()
        assert math.isclose(
        IoU,
        expected,
        rel_tol=1e-9)
        assert math.isclose(
        IoU,
        compute_iou(B, A),
        rel_tol=1e-9)

    # Erreur x2 < x1
    try:
        compute_iou(
            (100, 200, 90, 300),
            (110, 120, 190, 280)
            )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Erreur width = 0
    try:
        compute_iou(
            (100, 200, 100, 300),
            (110, 120, 190, 280)
            )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    try:
        compute_iou(
            (100, 200, 110, 300),
            (110, 120, 90, 280)
            )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Erreur heigth = 0
    try:
        compute_iou(
            (100, 200, 110, 200),
            (110, 120, 190, 280)
            )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    try:
        compute_iou(
            (100, 200, 110, 210),
            (110, 120, 190, 120))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


    print()
    print ("All IoU tests PASSED")

    print()
    print ("########################")
    print ("# Testing IoU matching #")
    print ("########################")
# Ground truth:
# GT0 = (100, 80, 180, 220)
# GT1 = (300, 90, 390, 240)
#
# Predictions:
# P0 = (105, 85, 176, 218)
# P1 = (295, 95, 395, 245)
# P2 = (500, 100, 540, 160)
    matrix = compute_iou_matrix([(100, 80, 180, 220), (300, 90, 390, 240)], [(105, 85, 176, 218), (295, 95, 395, 245), (500, 100, 540, 160)])
    print("matrix = ", matrix)
    match_list = match_from_score_matrix(matrix, 0.5)
    print("match_list = ", match_list)


#           D0      D1      D2
# R0       .84     .00     .00
# R1       .00     .84     .00
    matrix = [[.84, .00, .00], [.00, .84, .00]]
    match_list = match_from_score_matrix(matrix, 0.5)
    match_list = match_from_score_matrix(matrix, 0.5)
    print("matrix=", matrix)
    print("match_list = ", match_list)

#           D0      D1
# R0       .70     .65
# R1       .68     .10
    matrix = [[.70, .65], [.68, .10]]
    match_list = match_from_score_matrix(matrix, 0.5)
    print("matrix=", matrix)
    print("match_list = ", match_list)


    print()
    print ("#########################")
    print ("# Testing opt. matching #")
    print ("#########################")
    matrix = compute_iou_matrix([(100, 80, 180, 220), (300, 90, 390, 240)], [(105, 85, 176, 218), (295, 95, 395, 245), (500, 100, 540, 160)])
    maximum_cardinality_match_from_score_matrix(matrix, 0.5)
    print ("Done")


    matrix = [[0.7, 0.75], [0.68, 0.10]]
    print ("matrix = ", matrix)
    ret = maximum_cardinality_match_from_score_matrix(matrix, threshold=0.5)
    print ("maximum_cardinality_match_from_score_matrix = ", ret)
    ref_for_det = ret[1]
    for ret_idx in ref_for_det:
        ref_idx = ref_for_det[ret_idx]
        print (ref_idx, " - ",ret_idx, " = " , matrix[ref_for_det[ref_idx]][ref_idx])


    print()
    print ("#########################")
    print ("# Testing reward matrix #")
    print ("#########################")
    #reward_matrix = build_reward_matrix(matrix, 0.5)
    #print ("reward_matrix = ", reward_matrix)
    assert build_reward_matrix(
        [
            [0.70, 0.65],
            [0.68, 0.10]
            ],
        0.5
        ) == [
            [3.70, 3.65, 0.0, 0.0],
            [3.68, 0.0,  0.0, 0.0]
            ]



    assert build_reward_matrix(
        [
            [0.20, 0.30],
            [0.10, 0.49]
            ],
        0.5
        ) == [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0,  0.0, 0.0]
            ]
    print ("All reward matrix tests PASSED")


    print()
    print ("#################################")
    print ("# Testing linear_sum_assignment #")
    print ("#################################")
    matrix = [[0.7, 0.75], [0.68, 0.10]]
    print ("matrix = ", matrix)
    listScores = optimal_match_from_score_matrix(matrix, 0.5)
    print ("listScores = ", listScores)

    print ()
    # 1. Cas où greedy échoue
    matrix = [
        [0.70, 0.65],
        [0.68, 0.10]
    ]
    print ("matrix = ", matrix)
    # attendu :
    # R0-D1 (.65), R1-D0 (.68)
    listScores = optimal_match_from_score_matrix(matrix, 0.5)
    print ("listScores = ", listScores)
    assert (listScores == [.65, 0, 1], [.68, 1, 0])

    print ()
    # 2. Matching évident + détection sans correspondant
    matrix = [
        [0.84, 0.00, 0.00],
        [0.00, 0.84, 0.00]
    ]
    print ("matrix = ", matrix)
    listScores = optimal_match_from_score_matrix(matrix, 0.5)
    print ("listScores = ", listScores)
    assert (len(listScores) == 2)
    # attendu : 2 matches

    print ()
    # 3. Aucun match valide
    matrix = [
        [0.20, 0.30],
        [0.10, 0.49]
        ]
    print ("matrix = ", matrix)
    listScores = optimal_match_from_score_matrix(matrix, 0.5)
    print ("listScores = ", listScores)
    assert (len(listScores) == 0)
    # attendu : []

    print ()
    # 4. Une référence ne peut pas être associée
    matrix = [
        [0.80],
        [0.20]
        ]
    print ("matrix = ", matrix)
    listScores = optimal_match_from_score_matrix(matrix, 0.5)
    print ("listScores = ", listScores)
    assert (listScores == [(0.80, 0, 0)])
    # attendu : [(0.80, 0, 0)]

    print ("All linear_sum_assignment tests PASSED")


    print()
    print ("#############################")
    print ("# Testing detection metrics #")
    print ("#############################")
    match_0 = []
    match_2 = [(0.1, 2, 1),(0.1, 2, 2)]
#def compute_detection_metrics(
#    number_of_references,
#    number_of_detections,
#    matches
#)
#return (TP, FP, FN, precision, recall, f1)
    # 3 refs, 4 detections, 2 matches
    # TP=2 FP=2 FN=1
    resultat = compute_detection_metrics(
        3,
        4,
        match_2)
    print("Resultat: ", resultat)
    assert ([resultat["tp"], resultat["fp"], resultat["fn"]] == [2, 2, 1])

    # 2 refs, 2 detections, 2 matches
    # TP=2 FP=0 FN=0
    print()
    resultat = compute_detection_metrics(
        2,
        2,
        match_2)
    print("Resultat: ", resultat)
    assert ([resultat["tp"], resultat["fp"], resultat["fn"]] == [2, 0, 0])

    # 2 refs, 0 detection
    # TP=0 FP=0 FN=2
    print()
    resultat = compute_detection_metrics(
        2,
        0,
        match_0)
    print("Resultat: ", resultat)
    assert ([resultat["tp"], resultat["fp"], resultat["fn"]] == [0, 0, 2])

    # 0 ref, 3 detections
    # TP=0 FP=3 FN=0
    print()
    resultat = compute_detection_metrics(
        0,
        3,
        match_0)
    print("Resultat: ", resultat)
    assert ([resultat["tp"], resultat["fp"], resultat["fn"]] == [0, 3, 0])

    # 0 ref, 0 detection
    print()
    resultat = compute_detection_metrics(
        0,
        0,
        match_0)
    print("Resultat: ", resultat)
    assert ([resultat["tp"], resultat["fp"], resultat["fn"]] == [0, 0, 0])

    print ("All detection metrics tests PASSED")

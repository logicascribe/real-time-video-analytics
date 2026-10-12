import pytest
from pytest import approx
from IoUTools.compute_iou import compute_iou, optimal_match_from_score_matrix, build_reward_matrix, compute_detection_metrics

###############
# Compute IoU #
###############

@pytest.mark.parametrize(
    "box_a, box_b, expected",
    [
    # Identical
    pytest.param(
        (10, 10, 30, 30),
        (10, 10, 30, 30),
        1.0, id="Identical",
    ),

    # No overlap
    pytest.param(
        (10, 10, 20, 20),
        (30, 30, 40, 40),
        0.0, id="No-overlap",
    ),

    # Partial overlap
#    ((10, 10, 30, 30),
#    (20, 20, 40, 40),
#    100 / 700),
    pytest.param(
        (10, 10, 30, 30),
        (20, 20, 40, 40),
        100 / 700, id="partial-overlap",
    ),
    # One inside another
    pytest.param(
        (100, 100, 200, 300),
        (110, 120, 190, 280),
        0.64, id="One-inside-another",
    ),

    # Touching edge: zero intersection area
    pytest.param(
        (10, 10, 20, 20),
        (20, 10, 30, 20),
        0.0, id="Touching-edge:-zero-intersection area",
    ),
    ]
)
def test_compute_iou(box_a, box_b, expected):
    assert compute_iou(box_a, box_b) == pytest.approx(expected)

# Cas invalides
INVALID_BOXES = [
    pytest.param((100, 200, 90, 300), id="x2_inf_x1"), # x2 < x1
    pytest.param((100, 200, 100, 300), id="x2_eq_x1"), # x2 == x1
    pytest.param((100, 300, 110, 200), id="y2_inf_y1"), # y2 < y1
    pytest.param((100, 200, 110, 200), id="y2_eq_y1"),  # y2 == y1
    ]

@pytest.mark.parametrize("invalid_box", INVALID_BOXES,)
def test_compute_iou_rejects_invalid_boxes_first(invalid_box):
    valid_box = (10, 10, 20, 20)

    with pytest.raises(ValueError):
        compute_iou(invalid_box, valid_box)

@pytest.mark.parametrize("invalid_box", INVALID_BOXES,)
def test_compute_iou_rejects_invalid_boxes_second(invalid_box):
    valid_box = (10, 10, 20, 20)

    with pytest.raises(ValueError):
        compute_iou(valid_box, invalid_box)

def test_compute_iou_is_symmetric():
    box_a = (10, 10, 30, 30)
    box_b = (20, 20, 40, 40)
    assert compute_iou(box_a, box_b) == pytest.approx(compute_iou(box_b, box_a))


#########################
# Testing reward matrix #
#########################
REWARD_MATRIX = [
    pytest.param(
        [
            [0.70, 0.65],
            [0.68, 0.10]
            ], 0.5,
        [
            [3.70, 3.65, 0.0, 0.0],
            [3.68, 0.0,  0.0, 0.0]
            ], id="valid_matrix"),
    pytest.param(
        [
            [0.20, 0.30],
            [0.10, 0.49]
            ], 0.5,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0,  0.0, 0.0]
            ], id="invalid_matrix")
        ]

## Testing reward matric build ##
@pytest.mark.parametrize("score_matrix, threshold, expected_matrix", REWARD_MATRIX,)
def test_reward_matrix(score_matrix, threshold, expected_matrix):
    reward_matrix = build_reward_matrix(score_matrix, threshold)
    assert(checkMatrixEqualityApprox(reward_matrix, expected_matrix))


########################################
# Matching using linear_sum_assignment #
########################################
# Ground truth:
# GT0 = (100, 80, 180, 220)
# GT1 = (300, 90, 390, 240)
#
# Predictions:
# P0 = (105, 85, 176, 218)
# P1 = (295, 95, 395, 245)
# P2 = (500, 100, 540, 160)

## Patterns for linear_sum_assignment ##
MATCHING_PATTERNS = [
    # 1. Cas où greedy échoue
    pytest.param([
        [0.70, 0.65],
        [0.68, 0.10]
        ], 0.5, [ (.65, 0, 1), (.68, 1, 0) ]
    , id="greedy_fails",
    ),
    # 2. Aucun match valide
    pytest.param([
        [0.20, 0.30],
        [0.10, 0.49]
    ], 0.5, [  ]
    , id="no_valid_match",
    ),
    # 3. Une référence ne peut pas être associée
    pytest.param([
        [0.80],
        [0.20]
    ], 0.5, [ (0.80, 0, 0) ]
    , id="one_reference_is_impossible",
    ),
    ]

def isMatrix(python_list_of_lists):
    # Maps len() over every sublist and checks if there is exactly 1 unique length
    return 0 == len(python_list_of_lists) or len(set(len(row) for row in python_list_of_lists)) == 1


def checkMatrixDimEquality(matA, matB):
    if (len(matA) != len(matB)):
        return False
    nbRow = len(matA)
    if (0 == nbRow):
        return True
    return isMatrix(matA) and isMatrix(matB) and (len(matA[0]) == len(matB[0]))

def checkMatrixEqualityApprox(matA, matB):
    if not checkMatrixDimEquality(matA, matB):
        return False
    nbRow = len(matA)
    if (0 == nbRow):
        return True
    nbCol = len(matA[0])

    for itRow in range(nbRow):
            if not (matA[itRow] == approx(matB[itRow])):
                return False
    return True

TEST_MATRIX_VALID = [
    pytest.param([], [],
                 [True, True, True],
                 id = "2_empty_matrix"),
    pytest.param([], [ [1] ],
                 [True, False, False],
                 id = "1_empty__&_single_elem_matrix"),
    pytest.param([[1]], [[1]],
                 [True, True, True],
                 id = "2_single_value_matrix"),
    pytest.param([[1, 2],  [3, 4]],
                 [[1, 2],  [3, 4]],
                 [True, True, True],
                 id = "2_dummy_2*2_equal_matrix"),
    pytest.param([[1, 2],  [3, 4]],
                 [[1, 3],  [2, 4]],
                 [True, True, False],
                 id = "2_dummy_2*2_diff_matrix"),
    ]

@pytest.mark.parametrize("matA, matB, expected_results", TEST_MATRIX_VALID,)
def test_valid_matrix_checking(matA, matB, expected_results):
    resMat = []
    resMat.append(isMatrix(matA))
    resMat.append(checkMatrixDimEquality(matA, matB))
    resMat.append(checkMatrixEqualityApprox(matA, matB))
    assert(resMat == expected_results)



## Testing linear_sum_assignment ##
@pytest.mark.parametrize("score_matrix, threshold, expected_matrix", MATCHING_PATTERNS,)
def test_iou_matching(score_matrix, threshold, expected_matrix):
    resMat = optimal_match_from_score_matrix(score_matrix, threshold)
    assert(checkMatrixEqualityApprox(resMat, expected_matrix))

#           D0      D1      D2
# R0       .84     .00     .00
# R1       .00     .84     .00
    # 4. Matching évident + détection sans correspondant
UNMATCHED_DETECTION_PARAMS = [
    pytest.param(
        [
            [0.84, 0.00, 0.00],
            [0.00, 0.84, 0.00],
            ],
        0.5,
        [
            (0.84, 0, 0),
            (0.84, 1, 1),
            ], id="unmatched_detection",
    )]
@pytest.mark.parametrize("score_matrix, threshold, expected_matrix", UNMATCHED_DETECTION_PARAMS,)
def test_iou_matching_with_unmatched_detection(score_matrix, threshold, expected_matrix):
    resMat = optimal_match_from_score_matrix(score_matrix, threshold)
    assert checkMatrixEqualityApprox(resMat, expected_matrix)


#############################
# Testing detection metrics #
#############################
match_0 = []
match_2 = [
    (0.8, 0, 0),
    (0.7, 1, 1),
]
#def compute_detection_metrics(
#    number_of_references,
#    number_of_detections,
#    matches
#)
#return (TP, FP, FN, precision, recall, f1)
    # 3 refs, 4 detections, 2 matches
    # TP=2 FP=2 FN=1
DET_METRICS_PATTERNS = [
    pytest.param(
        [
            3, 4,
            match_2],
        [2, 2, 1],
        id="2_refs__2_detections__2_matches"),
    pytest.param(
        [
            2, 2,
            match_2],
        [2, 0, 0],
        id="2_refs__2_detection__2_matches"),
    pytest.param(
        [
            2, 0,
            match_0],
        [0, 0, 2],
        id="2_refs__0_detection"),
    pytest.param(
        [
            0, 3,
            match_0],
        [0, 3, 0],
        id="0_ref__3_detections"),
    pytest.param(
        [
            0, 0,
            match_0],
        [0, 0, 0],
        id="0_ref_0_detection"),
    ]

@pytest.mark.parametrize("metrics_matrix, expected_matrix", DET_METRICS_PATTERNS,)
def test_detection_metrics(metrics_matrix, expected_matrix):
    resultat = compute_detection_metrics(metrics_matrix[0], metrics_matrix[1], metrics_matrix[2])
    assert ([resultat["tp"], resultat["fp"], resultat["fn"]] == expected_matrix)

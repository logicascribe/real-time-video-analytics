import pytest
from pytest import approx
from IoUTools.compute_iou import optimal_match_from_score_matrix, build_reward_matrix
from .matrix_utility import (
    isMatrix,
    checkMatrixDimEquality,
    checkMatrixEqualityApprox,
)

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

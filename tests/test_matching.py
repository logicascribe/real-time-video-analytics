import numpy as np
import pytest
from pytest import approx
from IoUTools.compute_iou import build_reward_matrix, optimal_match_from_score_matrix
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
            ], id="valid_score_matrix"),
    pytest.param(
        [
            [0.20, 0.30],
            [0.10, 0.49]
            ], 0.5,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0,  0.0, 0.0]
            ], id="all_scores_below_threshold"),
    pytest.param(
        [
            [0.20, 0.30],
            [0.10, 0.5]
            ], 0.5,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.5,  0.0, 0.0]
            ], id="score_matrix_with_threshold_value"),


    pytest.param(
        np.empty((0, 3), dtype=np.float32),
        0.5,
        np.empty((0, 3), dtype=np.float32),
        id="score_matrix_0_3",
        ),
    pytest.param(
        np.empty((2, 0), dtype=np.float32),
        0.5,
        np.zeros((2, 2), dtype=np.float32),
        id="score_matrix_2_0",
        ),
    ]

@pytest.mark.parametrize("score_matrix, threshold, expected_matrix", REWARD_MATRIX,)
def test_reward_matrix(score_matrix, threshold, expected_matrix):
#    reward_matrix = build_reward_matrix(score_matrix, threshold)
    reward_matrix_NUMPY = build_reward_matrix(score_matrix, threshold)
#    np.testing.assert_allclose(reward_matrix, reward_matrix_NUMPY)
    np.testing.assert_allclose(expected_matrix, reward_matrix_NUMPY)

## Testing reward matrix build with empty matrices ##
REWARD_MATRIX_EMPTY_SCORE = [
    pytest.param(
        [], 0.5,
        (0, 0),
        id="empty_score_matrix")
    ]
@pytest.mark.parametrize("score_matrix, threshold, expected_shape", REWARD_MATRIX_EMPTY_SCORE,)
def test_reward_matrix_empty_score(score_matrix, threshold, expected_shape):
    reward_matrix = build_reward_matrix(score_matrix, threshold)
    assert reward_matrix.shape == expected_shape
    assert reward_matrix.dtype == np.float32

INVALID_SCORE_MATRICES = [
    pytest.param(
        [
            [0.20, 1.1],
            [0.10, 0.5],
        ],
        id="score_above_one",
    ),
    pytest.param(
        [
            [0.20, -0.1],
            [0.10, 0.5],
        ],
        id="negative_score",
    ),
]


@pytest.mark.parametrize("score_matrix", INVALID_SCORE_MATRICES)
def test_reward_matrix_rejects_invalid_scores(score_matrix):
    with pytest.raises(ValueError):
        build_reward_matrix(score_matrix, 0.5)

#############################################
# Optimal match using linear_sum_assignment #
#############################################
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

    # 4. Empty matrix (0, 3)
    pytest.param(
        np.empty((0, 3), dtype=np.float32),
        0.5, [],
        id="empty matrix_0_3"
    ),
    # 5. Empty matrix (2, 0)
    pytest.param(
        np.empty((2, 0), dtype=np.float32),
        0.5, [],
        id="empty matrix_2_0"
    ),
    # 6. Empty matrix (0, 0)
    pytest.param(
        np.empty((0, 0), dtype=np.float32),
        0.5,
        [],
        id="empty_matrix_0_0",
        ),

    pytest.param(
        [[0.5]],
        0.5,
        [(0.5, 0, 0)],
        id="score_equal_to_threshold",
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

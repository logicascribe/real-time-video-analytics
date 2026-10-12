import pytest
from pytest import approx
from IoUTools.compute_iou import compute_iou, optimal_match_from_score_matrix, build_reward_matrix, compute_detection_metrics

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

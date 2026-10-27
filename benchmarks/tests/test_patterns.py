import numpy as np
import pytest

from IoUTools.compute_iou import compute_iou_matrix

from ..patterns import (
    BoxPattern,
    horizontal_shift_for_iou,
    generate_isolated_pairs,
    box_from_center,
    generate_ambiguous_chain,
)

## Test pattern generation ##
def test_iou_simple():
    assert horizontal_shift_for_iou(100.0, 0.5) \
           == pytest.approx(100.0 / 3.0)

def test_iou_0():
    assert horizontal_shift_for_iou(100.0, 0.0) \
           == pytest.approx(100.0)

def test_iou_1():
    assert horizontal_shift_for_iou(100.0, 1.0) \
           == pytest.approx(0.0)

def test_iou_negative():
    with pytest.raises(ValueError):
        horizontal_shift_for_iou(100.0, -0.5)

def test_iou_greater_than_1():
    with pytest.raises(ValueError):
        horizontal_shift_for_iou(100.0, 1.5)

def test_iou_non_numeric():
    with pytest.raises(TypeError):
        horizontal_shift_for_iou(100.0, "a test text")

def test_iou_nan():
    with pytest.raises(ValueError):
        horizontal_shift_for_iou(
            100.0,
            np.nan,
        )

def test_iou__width_eq_0():
    with pytest.raises(ValueError):
        horizontal_shift_for_iou(0.0, 0.5)

def test_iou__width_less_than_0():
    with pytest.raises(ValueError):
        horizontal_shift_for_iou(-100.0, 0.5)

def test_iou__width_non_numeric():
    with pytest.raises(TypeError):
        horizontal_shift_for_iou("a test text", 0.6)

def test_width_nan():
    with pytest.raises(ValueError):
        horizontal_shift_for_iou(
            np.nan,
            0.5,
        )

@pytest.mark.parametrize(
    "iou",
    [np.inf, -np.inf],
)
def test_iou_infinite(iou):
    with pytest.raises(ValueError):
        horizontal_shift_for_iou(100.0, iou)

## Test "compute_iou_matrix" ##
@pytest.mark.parametrize(
    "target_iou",
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ],
)
def test_horizontal_shift_produces_target_iou(target_iou):
    width = 100.0
    height = 50.0

    shift = horizontal_shift_for_iou(
        width,
        target_iou,
    )

    reference_boxes = np.array(
        [
            [0.0, 0.0, width, height],
        ],
        dtype=np.float32,
    )

    detection_boxes = np.array(
        [
            [
                shift,
                0.0,
                shift + width,
                height,
            ],
        ],
        dtype=np.float32,
    )

    iou_matrix = compute_iou_matrix(
        reference_boxes,
        detection_boxes,
        )

    assert iou_matrix.shape == (1, 1)
    assert iou_matrix[0, 0] == pytest.approx(target_iou, abs=1e-6)


def test_iou_box_generation():
    pattern = generate_isolated_pairs(
        count=3,
        width=100.0,
        height=50.0,
        target_iou=0.5,
        spacing=200.0,
        )

    iou_matrix = compute_iou_matrix(
        pattern.reference_boxes,
        pattern.detection_boxes,
        )

    expected = np.eye(3, dtype=np.float32) * 0.5

    np.testing.assert_allclose(
        iou_matrix,
        expected,
        atol=1e-6,
        )

    assert iou_matrix.shape == (3, 3)


def test_isolated_pairs_produce_identity_adjacency():
    pattern = generate_isolated_pairs(
        count=3,
        width=100.0,
        height=50.0,
        target_iou=0.6,
        spacing=200.0,
    )

    iou_matrix = compute_iou_matrix(
        pattern.reference_boxes,
        pattern.detection_boxes,
    )
    assert iou_matrix.shape == (3, 3)

    threshold = 0.5

    observed_adjacency = iou_matrix >= threshold

    # assertion à écrire
    np.testing.assert_array_equal(
        observed_adjacency,
        pattern.target_adjacency,
        )

def test_box_pattern_rejects_invalid_reference_shape():
    detection_boxes = np.array(
        [
            [0.0, 0.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"reference_boxes must have shape \(R, 4\)\.",
    ):
        BoxPattern(
            reference_boxes=np.array(
                [
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            detection_boxes=detection_boxes,
            target_adjacency=np.eye(
                1,
                dtype=np.bool_,
            ),
        )

def test_box_pattern_rejects_invalid_detection_shape():
    reference_boxes = np.array(
        [
            [0.0, 0.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"detection_boxes must have shape \(D, 4\)\.",
    ):
        BoxPattern(
            reference_boxes=reference_boxes,
            detection_boxes=np.array(
                [
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            target_adjacency=np.eye(
                1,
                dtype=np.bool_,
            ),
        )


def test_box_pattern_rejects_inconsistent_adjacency_shape():
    reference_boxes = np.array(
        [
            [0.0, 0.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"target_adjacency must have shape \(R, D\), ",
    ):
        BoxPattern(
            reference_boxes=reference_boxes,
            detection_boxes=np.array(
                [
                    [0.0, 0.00, 100.0, 100.0],
                ],
                dtype=np.float32,
            ),
            target_adjacency=np.eye(
                2,
                dtype=np.bool_,
            ),
        )


def test_box_pattern_rejects_non_boolean_adjacency():
    reference_boxes = np.array(
        [
            [0.0, 0.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        TypeError,
        match=r"target_adjacency must have dtype np\.bool_\.",
    ):
        BoxPattern(
            reference_boxes=reference_boxes,
            detection_boxes=np.array(
                [
                    [0.0, 0.0, 100.0, 100.0],
                ],
                dtype=np.float32,
            ),
            target_adjacency=np.eye(
                1,
                dtype=np.int_,
            ),
        )

## Test patterns ambigus ##

def test_ambiguous_patterns():
    width = 100.0
    height = 50.0
    reference_centers = [100.0, 160.0]
    detection_centers = [80.0, 130.0, 180.0]

    reference_boxes = np.array(
        [box_from_center(cx, width, height) for cx in reference_centers],
        dtype=np.float32,
        )

    detection_boxes = np.array(
        [box_from_center(cx, width, height) for cx in detection_centers],
        dtype=np.float32,
        )

    assert reference_boxes.shape == (2, 4)
    assert detection_boxes.shape == (3, 4)

    iou_matrix = compute_iou_matrix(
        reference_boxes,
        detection_boxes,
        )

    assert iou_matrix.shape == (2, 3)

    observed_adjacency = iou_matrix >= 0.5

    expected_adjacency = np.array(
        [
            [True,  True,  False],
            [False, True,  True],
        ],
        dtype=np.bool_,
        )

    np.testing.assert_array_equal(
        observed_adjacency,
        expected_adjacency,
        )

def test_ambiguous_chain_with_valid_margin():
    threshold = 0.5
    iou_margin = 0.1
    edge_iou = threshold + iou_margin

    box_pattern = generate_ambiguous_chain(
        reference_count = 3,
        width = 100.0,
        height = 50.0,
        threshold = threshold,
        iou_margin = iou_margin,
        )

    iou_matrix = compute_iou_matrix(
        box_pattern.reference_boxes,
        box_pattern.detection_boxes,
        )

    observed_adjacency = iou_matrix >= threshold

    edge_scores = iou_matrix[
        box_pattern.target_adjacency
        ]

    non_edge_scores = iou_matrix[
        ~box_pattern.target_adjacency
        ]

    assert np.all(edge_scores > threshold)
    assert np.all(non_edge_scores < threshold)

    np.testing.assert_array_equal(
        observed_adjacency,
        box_pattern.target_adjacency,
        )

    np.testing.assert_allclose(
        edge_scores,
        edge_iou,
        atol=1e-6,
        )


def test_ambiguous_chain_rejects_zero_margin():
    threshold = 0.5
    iou_margin = 0.0

    with pytest.raises(
        ValueError,
        match=r"iou_margin must be > 0, got",
    ):
        generate_ambiguous_chain(
            reference_count = 3,
            width = 100.0,
            height = 50.0,
            threshold = threshold,
            iou_margin = iou_margin,
        )

def test_ambiguous_chain_rejects_margin_causing_spurious_edges():
    threshold = 0.5
    iou_margin = (1 - threshold*threshold)/(2 + threshold) + 0.1

    with pytest.raises(
        ValueError,
        match=r"iou_margin is too large for the requested threshold",
    ):
        generate_ambiguous_chain(
            reference_count = 3,
            width = 100.0,
            height = 50.0,
            threshold = threshold,
            iou_margin = iou_margin,
        )

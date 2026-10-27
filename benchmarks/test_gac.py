import numpy as np
import pytest

from IoUTools.compute_iou import compute_iou, compute_iou_matrix, _as_boxes_array
from .generate_patterns import (
    GAC,
    DegreeSequences,
    GACBuilder,
    GACRandomizer,
    horizontal_shift_for_iou,
    generate_isolated_pairs,
    BoxPattern,
    box_from_center,
    generate_ambiguous_chain,
    make_isolated_scenario,
    make_ambiguous_chain_scenario,
)

common_matrix = np.array(
    [
        [True,  False, False],
        [True,  True,  False],
        [False, True,  True],
    ])

def test_gac_properties():
    matrix = np.array(
        [
            [True,  False, False],
            [True,  True,  False],
            [False, True,  True],
        ],
        dtype=np.bool_,
    )

    gac = GAC(matrix)

    assert gac.size_R == 3
    assert gac.size_D == 3
    assert gac.edge_count == 5

    np.testing.assert_array_equal(
        gac.reference_degrees,
        np.array([1, 2, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        gac.detection_degrees,
        np.array([2, 2, 1], dtype=np.int64),
    )

    assert gac.reference_degrees.sum() == gac.edge_count
    assert gac.detection_degrees.sum() == gac.edge_count

def test_gac_density():
    matrix = np.array(
        [
            [True, False, False],
            [True, True, False],
            [False, True, True],
        ],
        dtype=np.bool_,
    )

    gac = GAC(matrix)

    assert gac.density == pytest.approx(5 / 9)


def test_gac_rejects_non_2d_matrix():
    matrix = np.array(
        [True, False, True],
        dtype=np.bool_,
    )

    with pytest.raises(
        ValueError,
        match="must be a 2-D matrix",
    ):
        GAC(matrix)


def test_gac_rejects_non_boolean_matrix():
    matrix = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.int64,
    )

    with pytest.raises(
        TypeError,
        match="dtype np.bool_",
    ):
        GAC(matrix)


def test_empty_gac_has_zero_density():
    gac = GAC(
        np.empty((0, 0), dtype=np.bool_)
    )

    assert gac.size_R == 0
    assert gac.size_D == 0
    assert gac.edge_count == 0
    assert gac.density == 0.0


def test_gac_builder_preserves_degree_sequences():
    degree_sequences = DegreeSequences(
        reference_degrees=np.array(
            [2, 2, 1, 1],
            dtype=np.int64,
        ),
        detection_degrees=np.array(
            [2, 2, 2],
            dtype=np.int64,
        ),
    )

    builder = GACBuilder()

    gac = builder(degree_sequences)

    np.testing.assert_array_equal(
        gac.reference_degrees,
        degree_sequences.reference_degrees,
    )

    np.testing.assert_array_equal(
        gac.detection_degrees,
        degree_sequences.detection_degrees,
    )

    assert gac.edge_count == degree_sequences.edge_count

## Randomization

def test_gac_randomizer_performs_switch():
    original = GAC(
        np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=np.bool_,
        )
    )

    randomizer = GACRandomizer(seed=42)

    randomized = randomizer(
        original,
        num_switches=1,
    )

    expected = np.array(
        [
            [False, True],
            [True, False],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(
        randomized.target_adjacency,
        expected,
    )

def test_gac_randomizer_num_switches_zero():
    original = GAC(
        np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=np.bool_,
        )
    )

    randomizer = GACRandomizer(seed=42)

    randomized = randomizer(
        original,
        num_switches=0,
    )

    expected = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(
        randomized.target_adjacency,
        expected,
    )

def test_gac_randomizer_less_than_2_edges():
    original = GAC(
        np.array(
            [
                [True, False],
            ],
            dtype=np.bool_,
        )
    )

    randomizer = GACRandomizer(seed=42)

    with pytest.raises(
        RuntimeError,
        match="Could perform only ",
        ):
        randomizer(
            original,
            num_switches=1,
            )

def test_gac_randomizer_on_full_graph():
    gac = GAC(
        np.array(
            [
                [True, True, True],
                [True, True, True],
            ],
            dtype=np.bool_,
        )
    )

    with pytest.raises(RuntimeError):
        GACRandomizer(seed=42)(
            gac,
            num_switches=1,
            )

def test_gac_randomizer_rejects_negative_num_switches():
    gac = GAC(
        np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=np.bool_,
        )
    )

    with pytest.raises(ValueError):
        GACRandomizer(seed=42)(
            gac,
            num_switches=-1,
            )

def test_gac_randomizer_rejects_non_integer_num_switches():
    gac = GAC(
        np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=np.bool_,
        )
    )

    with pytest.raises(TypeError):
        GACRandomizer(seed=42)(
            gac,
            num_switches=1.5,
            )


def test_gac_randomizer_preserves_degrees():
    reference_degrees = np.array(
        [1, 2, 2],
        dtype=np.int64,
    )
    detection_degrees = np.array(
        [2, 2, 1],
        dtype=np.int64,
    )

    degree_sequences = DegreeSequences(
        reference_degrees,
        detection_degrees,
    )

    builder = GACBuilder()
    original = builder(degree_sequences)

    randomizer = GACRandomizer(42)
    randomized = randomizer(original, 3)

    np.testing.assert_array_equal(
        randomized.reference_degrees,
        original.reference_degrees,
        )

    np.testing.assert_array_equal(
        randomized.detection_degrees,
        original.detection_degrees,
        )

    assert randomized.edge_count == original.edge_count

def test_gac_randomizer_is_reproducible_with_same_seed():
    reference_degrees = np.array(
        [1, 2, 2],
        dtype=np.int64,
    )
    detection_degrees = np.array(
        [2, 2, 1],
        dtype=np.int64,
    )

    degree_sequences = DegreeSequences(
        reference_degrees,
        detection_degrees,
    )

    builder = GACBuilder()
    original = builder(degree_sequences)

    randomizer1 = GACRandomizer(42)
    randomized1 = randomizer1(original, 3)

    randomizer2 = GACRandomizer(42)
    randomized2 = randomizer2(original, 3)

    np.testing.assert_array_equal(
        randomized1.target_adjacency,
        randomized2.target_adjacency,
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

def test_isolated_scenario_20():
    scenario = make_isolated_scenario(20, "isolated_20")

    assert scenario.reference_count == 20
    assert scenario.detection_count == 20
    assert scenario.pattern.reference_boxes.shape == (20, 4)
    assert scenario.pattern.detection_boxes.shape == (20, 4)
    assert scenario.name == "isolated_20"
    assert scenario.threshold == pytest.approx(0.5)

    observed = (
        compute_iou_matrix(
            scenario.pattern.reference_boxes,
            scenario.pattern.detection_boxes,
            )
        >= scenario.threshold
        )

    np.testing.assert_array_equal(
        observed,
        scenario.pattern.target_adjacency,
        )

def test_ambiguous_chain_scenario_20():
    scenario_name = "ambiguous_20"
    scenario = make_ambiguous_chain_scenario(20, scenario_name)

    assert scenario.reference_count == 20
    assert scenario.detection_count == 21
    assert scenario.pattern.reference_boxes.shape == (20, 4)
    assert scenario.pattern.detection_boxes.shape == (21, 4)
    assert scenario.pattern.target_adjacency.shape == (20, 21)
    assert scenario.name == scenario_name
    assert scenario.threshold == pytest.approx(0.5)

    observed = (
        compute_iou_matrix(
            scenario.pattern.reference_boxes,
            scenario.pattern.detection_boxes,
            )
        >= scenario.threshold
        )

    np.testing.assert_array_equal(
        observed,
        scenario.pattern.target_adjacency,
        )

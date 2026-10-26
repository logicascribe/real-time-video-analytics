import numpy as np
import pytest

from .generate_patterns import (
    GAC,
    DegreeSequences,
    GACBuilder,
    GACRandomizer,
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


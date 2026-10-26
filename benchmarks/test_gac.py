import numpy as np
import pytest

from .generate_patterns import (
    GAC,
    DegreeSequences,
    GACBuilder,
)



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

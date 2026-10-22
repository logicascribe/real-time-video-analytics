import numpy as np
import pytest
from .generate_patterns import (
    BipartiteDegreeGenerator,
    ProblemSize
)


def test_generate_compatible_degree_sequences():
    generator = BipartiteDegreeGenerator()

    degrees = generator(
        reference_distribution={1: 0.5, 2: 0.5},
        detection_distribution={2: 1.0},
        size=ProblemSize(references=4, detections=3),
    )

    np.testing.assert_array_equal(
        degrees.reference_degrees,
        np.array([1, 1, 2, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        degrees.detection_degrees,
        np.array([2, 2, 2], dtype=np.int64),
    )

    assert degrees.edge_count == 6
    assert (
        degrees.reference_degrees.sum()
        == degrees.detection_degrees.sum()
        == degrees.edge_count
        )


def test_largest_remainder_discretization():
    """
    R = 7

    Distribution:
        degree 0: 0.2 -> expected count 1.4
        degree 1: 0.3 -> expected count 2.1
        degree 2: 0.5 -> expected count 3.5

    Integer parts:
        [1, 2, 3]

    One element remains.
    Largest remainder is 0.5, associated with degree 2.

    Final counts:
        degree 0 -> 1
        degree 1 -> 2
        degree 2 -> 4

    Degree sequence:
        [0, 1, 1, 2, 2, 2, 2]

    Total number of edges:
        0 + 1 + 1 + 2 + 2 + 2 + 2 = 10
    """
    generator = BipartiteDegreeGenerator()

    degrees = generator(
        reference_distribution={
            0: 0.2,
            1: 0.3,
            2: 0.5,
        },
        detection_distribution={
            2: 1.0,
        },
        size=ProblemSize(
            references=7,
            detections=5,
        ),
    )

    np.testing.assert_array_equal(
        degrees.reference_degrees,
        np.array(
            [0, 1, 1, 2, 2, 2, 2],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        degrees.detection_degrees,
        np.array(
            [2, 2, 2, 2, 2],
            dtype=np.int64,
        ),
    )

    assert degrees.edge_count == 10


def test_incompatible_edge_counts_raise():
    generator = BipartiteDegreeGenerator()

    # Reference side:
    # [1, 1, 2, 2] -> 6 edges
    #
    # Detection side:
    # [1, 1, 1] -> 3 edges
    with pytest.raises(
        ValueError,
        match="Incompatible degree distributions",
    ):
        generator(
            reference_distribution={
                1: 0.5,
                2: 0.5,
            },
            detection_distribution={
                1: 1.0,
            },
            size=ProblemSize(
                references=4,
                detections=3,
            ),
        )


def test_reference_degree_above_detection_count_raises():
    generator = BipartiteDegreeGenerator()

    # There are only 3 detection vertices.
    # Therefore a reference vertex cannot have degree 4.
    with pytest.raises(
        ValueError,
        match="reference degree cannot exceed 3",
    ):
        generator(
            reference_distribution={
                4: 1.0,
            },
            detection_distribution={
                1: 1.0,
            },
            size=ProblemSize(
                references=4,
                detections=3,
            ),
        )


def test_detection_degree_above_reference_count_raises():
    generator = BipartiteDegreeGenerator()

    # There are only 3 reference vertices.
    # Therefore a detection vertex cannot have degree 4.
    with pytest.raises(
        ValueError,
        match="detection degree cannot exceed 3",
    ):
        generator(
            reference_distribution={
                1: 1.0,
            },
            detection_distribution={
                4: 1.0,
            },
            size=ProblemSize(
                references=3,
                detections=4,
            ),
        )

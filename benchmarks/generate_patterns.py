#!/usr/bin/python3

import numpy as np
from enum import auto, Enum
from dataclasses import dataclass
from typing import Mapping, TypeAlias
from numpy.typing import NDArray

####################
## Generation GAC ##
####################

DegreeArray: TypeAlias = NDArray[np.int64]
DegreeDistribution: TypeAlias = Mapping[int, float]


@dataclass(frozen=True)
class ProblemSize:
    references: int
    detections: int


@dataclass
class DegreeSequences:
    reference_degrees: DegreeArray
    detection_degrees: DegreeArray

    @property
    def edge_count(self) -> int:
        return int(self.reference_degrees.sum())

class BipartiteDegreeGenerator:
    def __call__(
        self,
        reference_distribution: DegreeDistribution,
        detection_distribution: DegreeDistribution,
        size: ProblemSize,
    ) -> DegreeSequences:

        self._validate_problem_size(size)

        reference_degrees = self._discretize_distribution(
            distribution=reference_distribution,
            n=size.references,
        )

        detection_degrees = self._discretize_distribution(
            distribution=detection_distribution,
            n=size.detections,
        )

        self._validate_degree_bounds(
            reference_degrees,
            max_degree=size.detections,
            name="reference",
        )

        self._validate_degree_bounds(
            detection_degrees,
            max_degree=size.references,
            name="detection",
        )

        reference_edges = int(reference_degrees.sum())
        detection_edges = int(detection_degrees.sum())

        if reference_edges != detection_edges:
            raise ValueError(
                "Incompatible degree distributions: "
                f"reference side requires {reference_edges} edges, "
                f"detection side requires {detection_edges} edges."
            )

        return DegreeSequences(
            reference_degrees=reference_degrees,
            detection_degrees=detection_degrees,
        )

    @staticmethod
    def _discretize_distribution(
        distribution: DegreeDistribution,
        n: int,
        ) -> DegreeArray:

        BipartiteDegreeGenerator._validate_distribution(distribution)

        degrees = np.asarray(
            sorted(distribution.keys()),
            dtype=np.int64,
            )

        probabilities = np.asarray(
            [distribution[int(degree)] for degree in degrees],
            dtype=np.float64,
            )

        expected_counts = probabilities * n

        counts = np.floor(expected_counts).astype(np.int64)
        remainders = expected_counts - counts

        remaining = n - int(counts.sum())

        if remaining > 0:
            # Stable sort gives deterministic tie-breaking:
            # smaller degree first when remainders are equal.
            priority = np.argsort(-remainders, kind="stable")
            counts[priority[:remaining]] += 1

        degree_sequence = np.repeat(degrees, counts)

        if degree_sequence.shape != (n,):
            raise RuntimeError(
                "Internal error while discretizing degree distribution."
                )

        return degree_sequence.astype(np.int64, copy=False)

    @staticmethod
    def _validate_distribution(
        distribution: DegreeDistribution,
        ) -> None:

        if not distribution:
            raise ValueError("Degree distribution must not be empty.")

        for degree, probability in distribution.items():
            if (
                not isinstance(degree, (int, np.integer))
                or isinstance(degree, (bool, np.bool_))
                ):
                raise TypeError(
                    f"Degree must be an integer, got {degree!r}."
                    )

            if degree < 0:
                raise ValueError(
                    f"Degree must be >= 0, got {degree}."
                    )

            if not isinstance(
                probability,
                (int, float, np.integer, np.floating),
                ):
                raise TypeError(
                    f"Probability must be numeric, got {probability!r}."
                    )

            if not np.isfinite(probability):
                raise ValueError("Probability must be finite.")

            if probability < 0:
                raise ValueError("Probability must be >= 0.")


            total = float(sum(distribution.values()))

            if not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"Probabilities must sum to 1.0, got {total}."
                    )

    @staticmethod
    def _validate_problem_size(size: ProblemSize) -> None:
        for name, value in (
            ("references", size.references),
            ("detections", size.detections),
            ):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                ):
                raise TypeError(
                    f"{name} must be an integer, got {value!r}."
                    )

            if value < 0:
                raise ValueError(
                    f"{name} must be >= 0, got {value}."
                    )

    @staticmethod
    def _validate_degree_bounds(
        degrees: DegreeArray,
        max_degree: int,
        name: str,
    ) -> None:
        if np.any(degrees > max_degree):
            raise ValueError(
                f"{name} degree cannot exceed {max_degree}."
            )



AdjacencyMatrix: TypeAlias = NDArray[np.bool_]

@dataclass
class GAC:
    """
    Graphe d'Adjacence Cible biparti.

    Rows    : references R
    Columns : detections D

    target_adjacency[i, j] is True iff detection j is a candidate
    for reference i.
    """

    target_adjacency: AdjacencyMatrix

    def __post_init__(self) -> None:
        if not isinstance(self.target_adjacency, np.ndarray):
            raise TypeError(
                "target_adjacency must be a NumPy ndarray."
            )

        if self.target_adjacency.ndim != 2:
            raise ValueError(
                "target_adjacency must be a 2-D matrix, "
                f"got shape {self.target_adjacency.shape}."
            )

        if self.target_adjacency.dtype != np.bool_:
            raise TypeError(
                "target_adjacency must have dtype np.bool_, "
                f"got {self.target_adjacency.dtype}."
            )

    @property
    def size_R(self) -> int:
        """Number of reference vertices."""
        return self.target_adjacency.shape[0]

    @property
    def size_D(self) -> int:
        """Number of detection vertices."""
        return self.target_adjacency.shape[1]

    @property
    def reference_degrees(self) -> DegreeArray:
        """
        Degree of each reference vertex.

        reference_degrees[i] =
            number of detections connected to reference i.
        """
        return self.target_adjacency.sum(
            axis=1,
            dtype=np.int64,
        )

    @property
    def detection_degrees(self) -> DegreeArray:
        """
        Degree of each detection vertex.

        detection_degrees[j] =
            number of references connected to detection j.
        """
        return self.target_adjacency.sum(
            axis=0,
            dtype=np.int64,
        )

    @property
    def edge_count(self) -> int:
        """Total number of edges in the bipartite graph."""
        return int(
            self.target_adjacency.sum(dtype=np.int64)
        )

    @property
    def density(self) -> float:
        """
        Ratio of existing edges to all possible R*D edges.
        """
        possible_edges = self.size_R * self.size_D

        if possible_edges == 0:
            return 0.0

        return self.edge_count / possible_edges

########################
## Fin Generation GAC ##
########################

import numpy as np

from dataclasses import dataclass
from typing import TypeAlias

from numpy.typing import NDArray

from .degree_sequences import (
    DegreeArray,
    DegreeSequences,
    is_bipartite_graphical,
)

####################
## Generation GAC ##
####################

AdjacencyMatrix: TypeAlias = NDArray[np.bool_]

@dataclass
class GAC:
    """
    Graphe d'Adjacence Cible biparti.

    Rows    : references R
    Columns : detections D

    target_adjacency[i, j] is True if detection j is a candidate
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



class GACBuilder:
    def __call__(
        self,
        degree_sequences: DegreeSequences,
    ) -> GAC:

        reference_degrees = degree_sequences.reference_degrees
        detection_degrees = degree_sequences.detection_degrees

        if not is_bipartite_graphical(
            reference_degrees,
            detection_degrees,
        ):
            raise ValueError(
                "Degree sequences are not bipartite graphical."
            )

        R = reference_degrees.size
        D = detection_degrees.size

        adjacency = np.zeros(
            (R, D),
            dtype=np.bool_,
        )

        reference_remaining = reference_degrees.copy()
        detection_remaining = detection_degrees.copy()

        while np.any(reference_remaining > 0):
            # Reference having the largest remaining degree.
            ref_idx = int(np.argmax(reference_remaining))
            degree = int(reference_remaining[ref_idx])

            # Detections having the largest remaining degrees.
            det_indices = np.argsort(
                -detection_remaining,
                kind="stable",
                )[:degree]

            if np.any(detection_remaining[det_indices] <= 0):
                raise RuntimeError(
                    "GAC construction failed despite graphical degree sequences."
                    )

            adjacency[ref_idx, det_indices] = True

            detection_remaining[det_indices] -= 1
            reference_remaining[ref_idx] = 0

        gac = GAC(adjacency)

        return gac


class GACRandomizer:
    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        gac: GAC,
        num_switches: int,
    ) -> GAC:
        if not isinstance(num_switches, (int, np.integer)):
            raise TypeError("num_switches must be an integer.")
        if num_switches < 0:
            raise ValueError("num_switches must be >= 0.")
        adjacency = gac.target_adjacency.copy()
        successful_switches = 0
        attempts = 0
        max_attempts = 10 * num_switches

        while (
            attempts < max_attempts
            and successful_switches < num_switches
            ):
            attempts += 1

            edges = np.argwhere(adjacency)
            if len(edges) < 2:
                break
            edge_indices = self._rng.choice(
                len(edges),
                size=2,
                replace=False,
                )
            r1, d1 = edges[edge_indices[0]]
            r2, d2 = edges[edge_indices[1]]

            if (
                r1 != r2
                and d1 != d2
                and not adjacency[r1, d2]
                and not adjacency[r2, d1]
                ):
                adjacency[r1, d1] = False
                adjacency[r2, d2] = False

                adjacency[r1, d2] = True
                adjacency[r2, d1] = True

                successful_switches += 1

        if successful_switches < num_switches:
            raise RuntimeError(
                f"Could perform only {successful_switches} "
                f"of {num_switches} requested switches."
                )
        return GAC(adjacency)

#!/usr/bin/python3

import numpy as np
import time
from enum import auto, Enum
from dataclasses import dataclass
from typing import Mapping, TypeAlias
from numpy.typing import NDArray

from IoUTools.compute_iou import BoxesArray, compute_iou_matrix


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

"""
1. vérifier que les deux entrées sont des vecteurs 1-D

2. vérifier les degrés :
      r_i >= 0
      d_j >= 0

3. vérifier les bornes :
      r_i <= D
      d_j <= R

4. vérifier :
      sum(r) == sum(d)

5. trier r et d décroissants

6. pour k = 1 ... R :
      left  = sum(r[:k])
      right = sum(min(k, d_j))

      si left > right :
          return False

7. return True
"""
def is_bipartite_graphical(
    reference_degrees: DegreeArray,
    detection_degrees: DegreeArray,
    ) -> bool:
    """
    Return True if the two degree sequences can be realized
    by a simple bipartite graph.

    Uses the Gale-Ryser theorem.

    Raises:
    TypeError: if inputs are not NumPy arrays or do not contain integers.
    ValueError: if inputs are not 1-D or contain negative degrees.
    """

# 1. Validate input types
    if not isinstance(reference_degrees, np.ndarray):
        raise TypeError(
            "reference_degrees must be a NumPy ndarray."
            )

    if not isinstance(detection_degrees, np.ndarray):
        raise TypeError(
            "detection_degrees must be a NumPy ndarray."
            )

    if not np.issubdtype(reference_degrees.dtype, np.integer):
        raise TypeError(
            "reference_degrees must contain integers."
            )

    if not np.issubdtype(detection_degrees.dtype, np.integer):
        raise TypeError(
            "detection_degrees must contain integers."
            )

    # 2. Degree sequences must be 1-D
    if reference_degrees.ndim != 1:
        raise ValueError(
            "reference_degrees must be a 1-D array."
            )

    if detection_degrees.ndim != 1:
        raise ValueError(
            "detection_degrees must be a 1-D array."
            )

    R_size = reference_degrees.size
    D_size = detection_degrees.size

    # 3. Degrees must be non-negative
    if (
        np.any(reference_degrees < 0)
        or np.any(detection_degrees < 0)
        ):
        raise ValueError(
            "Degrees must be >= 0."
            )

    # 4. Individual degree bounds
    if np.any(reference_degrees > D_size):
        return False

    if np.any(detection_degrees > R_size):
        return False

    # 5. Both partitions must describe the same number of edges
    reference_edge_count = int(reference_degrees.sum())
    detection_edge_count = int(detection_degrees.sum())

    if reference_edge_count != detection_edge_count:
        return False

    # 6. Gale-Ryser requires non-increasing sequences
    sorted_r = np.sort(reference_degrees)[::-1]
    sorted_d = np.sort(detection_degrees)[::-1]

    # 7. Gale-Ryser inequalities
    #
    # For every k in {1, ..., R}:
    #
    #   sum(r[0:k]) <= sum_j min(k, d_j)
    #
    left_cumsum = np.cumsum(sorted_r)
    for k in range(1, R_size + 1):
        left = left_cumsum[k - 1]
        right = np.minimum(k, sorted_d).sum()

        if left > right:
            return False

    return True

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


#######################
## Generate patterns ##
#######################
def horizontal_shift_for_iou(
    width: float,
    iou: float,
) -> float:
    if not np.isfinite(width) or width <= 0:
        raise ValueError(
            f"width must be finite and > 0, got {width}."
        )

    if not np.isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError(
            f"iou must be finite and in [0, 1], got {iou}."
        )

    return width * (1.0 - iou) / (1.0 + iou)


@dataclass
class BoxPattern:
    reference_boxes: BoxesArray
    detection_boxes: BoxesArray
    target_adjacency: AdjacencyMatrix

    def __post_init__(self) -> None:
        if not isinstance(self.reference_boxes, np.ndarray):
            raise TypeError(
                "reference_boxes must be a NumPy ndarray."
            )

        if self.reference_boxes.ndim != 2:
            raise ValueError(
                "reference_boxes must be a 2-D array."
            )

        if self.reference_boxes.shape[1] != 4:
            raise ValueError(
                "reference_boxes must have shape (R, 4)."
            )

        if not isinstance(self.detection_boxes, np.ndarray):
            raise TypeError(
                "reference_boxes must be a NumPy ndarray."
            )

        if self.detection_boxes.ndim != 2:
            raise ValueError(
                "detection_boxes must be a 2-D array."
            )

        if self.detection_boxes.shape[1] != 4:
            raise ValueError(
                "detection_boxes must have shape (D, 4)."
            )

        R = self.reference_boxes.shape[0]
        D = self.detection_boxes.shape[0]

        if self.target_adjacency.shape != (R, D):
            raise ValueError(
                "target_adjacency must have shape (R, D), "
                f"expected {(R, D)}, "
                f"got {self.target_adjacency.shape}."
            )

        if self.target_adjacency.dtype != np.bool_:
            raise TypeError(
                "target_adjacency must have dtype np.bool_."
            )


def generate_isolated_pairs(
    count: int,
    width: float,
    height: float,
    target_iou: float,
    spacing: float,
) -> BoxPattern:

    reference_boxes = np.empty(
        (count, 4),
        dtype=np.float32,
    )

    detection_boxes = np.empty(
        (count, 4),
        dtype=np.float32,
    )

    dx = horizontal_shift_for_iou(width, target_iou)

    for idx in range(count):
        x1 = idx * spacing
        x2 = x1 + width
        reference_boxes[idx] = (x1, 0.0, x2, height)
        detection_x1 = x1 + dx
        detection_x2 = x2 + dx
        detection_boxes[idx] = (detection_x1, 0, detection_x2, height)
    return BoxPattern(
        reference_boxes=reference_boxes,
        detection_boxes=detection_boxes,
        target_adjacency=np.eye(
            count,
            dtype=np.bool_,
            ),
        )


def box_from_center(cx, width, height):
    x1 = cx - width / 2
    x2 = cx + width / 2
    return (x1, 0.0, x2, height)


def generate_ambiguous_chain(
    reference_count: int,
    width: float,
    height: float,
    threshold: float,
    iou_margin: float,
) -> BoxPattern:
    edge_iou = threshold + iou_margin

    max_edge_iou = (
        (1.0 + 2.0 * threshold)
        / (2.0 + threshold)
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}."
            )

    if iou_margin <= 0.0:
        raise ValueError(
            f"iou_margin must be > 0, got {iou_margin}."
            )

    if edge_iou >= max_edge_iou:
        raise ValueError(
            "iou_margin is too large for the requested threshold: "
            "non-neighboring boxes would become candidate edges."
            )

    detection_count = reference_count + 1
    shift = horizontal_shift_for_iou(
        width,
        edge_iou,
        )

    spacing = 2.0 * shift
    detection_centers = [
        idx * spacing
        for idx in range(detection_count)
        ]

    reference_centers = [
        (idx + 0.5) * spacing
        for idx in range(reference_count)
        ]

    target_adjacency = np.zeros(
        (reference_count, detection_count),
        dtype=np.bool_,
        )

    for idx in range(reference_count):
        target_adjacency[idx, idx] = True
        target_adjacency[idx, idx + 1] = True

    reference_boxes = np.array(
        [
            box_from_center(cx, width, height)
            for cx in reference_centers
        ],
        dtype=np.float32,
    )

    detection_boxes = np.array(
        [
            box_from_center(cx, width, height)
            for cx in detection_centers
        ],
        dtype=np.float32,
    )

    return BoxPattern(
        reference_boxes=reference_boxes,
        detection_boxes=detection_boxes,
        target_adjacency=target_adjacency,
        )


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    threshold: float
    pattern: BoxPattern

    @property
    def reference_count(self) -> int:
        return self.pattern.reference_boxes.shape[0]

    @property
    def detection_count(self) -> int:
        return self.pattern.detection_boxes.shape[0]


# Début avc 4 scenarios
# isolated_5
# isolated_50
# ambiguous_chain_5
# ambiguous_chain_50

DEFAULT_BOX_WIDTH = 100.0
DEFAULT_BOX_HEIGHT = 50.0
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MARGIN_RATIO = 0.5

def make_isolated_scenario(
    count: int,
    name: str
) -> BenchmarkScenario:
    width = DEFAULT_BOX_WIDTH
    height = DEFAULT_BOX_HEIGHT
    threshold = DEFAULT_IOU_THRESHOLD
    target_iou = 0.6
    spacing = 200.0
    box_pattern = generate_isolated_pairs(
        count = count,
        width = width,
        height = height,
        target_iou = target_iou,
        spacing = spacing,
        )
    benchmark_scenario = BenchmarkScenario(
        name = name,
        threshold = threshold,
        pattern = box_pattern
        )
    return benchmark_scenario

def max_ambiguous_chain_iou_margin(
    threshold: float,
) -> float:
    return (
        (1.0 - threshold * threshold)
        / (2.0 + threshold)
    )


def max_ambiguous_chain_iou_margin(
    threshold: float,
) -> float:
    return (
        (1.0 - threshold * threshold)
        / (2.0 + threshold)
    )


def make_ambiguous_chain_scenario(
    count: int,
    name: str,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
) -> BenchmarkScenario:
    width = DEFAULT_BOX_WIDTH
    height = DEFAULT_BOX_HEIGHT
    threshold = DEFAULT_IOU_THRESHOLD

    if not 0.0 < margin_ratio < 1.0:
        raise ValueError(
            "margin_ratio must be in (0, 1), "
            f"got {margin_ratio}."
        )

    max_iou_margin = max_ambiguous_chain_iou_margin(
        threshold
    )

    iou_margin = margin_ratio * max_iou_margin

    box_pattern = generate_ambiguous_chain(
        reference_count=count,
        width=width,
        height=height,
        threshold=threshold,
        iou_margin=iou_margin,
    )

    return BenchmarkScenario(
        name=name,
        threshold=threshold,
        pattern=box_pattern,
    )


#############################
## Runner de benchmark IoU ##
#############################

@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iterations: int
    measured_iterations: int

@dataclass(frozen=True)
class BenchmarkResult:
    scenario_name: str
    reference_count: int
    detection_count: int
    mean_ns: float
    median_ns: float
    std_ns: float
    p95_ns: float
    p99_ns: float
    min_ns: int
    max_ns: int
    durations_ns: np.ndarray

def measure_perf(scenario: BenchmarkScenario):
    start = time.perf_counter_ns()

    compute_iou_matrix(
        scenario.pattern.reference_boxes,
        scenario.pattern.detection_boxes,
        )
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns

def benchmark_iou(
    scenario: BenchmarkScenario,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    for _ in range(config.warmup_iterations):
        compute_iou_matrix(
            scenario.pattern.reference_boxes,
            scenario.pattern.detection_boxes,
        )

    durations_ns = np.empty(
        config.measured_iterations,
        dtype=np.int64,
    )

    for idx in range(config.measured_iterations):
        durations_ns[idx] = measure_perf(
            scenario
        )

    return BenchmarkResult(
        scenario_name=scenario.name,
        reference_count=scenario.reference_count,
        detection_count=scenario.detection_count,
        mean_ns=float(np.mean(durations_ns)),
        median_ns=float(np.median(durations_ns)),
        std_ns=float(np.std(durations_ns)),
        p95_ns=float(np.percentile(durations_ns, 95)),
        p99_ns=float(np.percentile(durations_ns, 99)),
        min_ns=int(np.min(durations_ns)),
        max_ns=int(np.max(durations_ns)),
        durations_ns=durations_ns,
        )

if __name__ == "__main__":
    benchmark_scenario = make_ambiguous_chain_scenario(
        count=5,
        name="ambiguous_count_5",
    )

    benchmark_config = BenchmarkConfig(
        warmup_iterations=10,
        measured_iterations=100,
    )

    perf = benchmark_iou(
        benchmark_scenario,
        benchmark_config,
    )

    print(f"perf = {perf}")



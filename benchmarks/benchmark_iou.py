import time

import numpy as np

from dataclasses import dataclass

from IoUTools.compute_iou import compute_iou_matrix

from .scenarios import (
    BenchmarkScenario,
    make_ambiguous_chain_scenario,
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

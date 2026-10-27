from dataclasses import dataclass

from .patterns import (
    BoxPattern,
    generate_isolated_pairs,
    generate_ambiguous_chain,
    max_ambiguous_chain_iou_margin,
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

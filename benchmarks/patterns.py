import numpy as np

from dataclasses import dataclass

from IoUTools.compute_iou import BoxesArray
from .gac import AdjacencyMatrix

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

import pytest

from IoUTools.compute_iou import compute_iou


@pytest.mark.parametrize(
    "box_a, box_b, expected",
    [
    # Identical
    pytest.param(
        (10, 10, 30, 30),
        (10, 10, 30, 30),
        1.0,
        id="Identical",
    ),

    # No overlap
    pytest.param(
        (10, 10, 20, 20),
        (30, 30, 40, 40),
        0.0,
        id="No-overlap",
    ),

    # Partial overlap
#    ((10, 10, 30, 30),
#    (20, 20, 40, 40),
#    100 / 700),
    pytest.param(
        (10, 10, 30, 30),
        (20, 20, 40, 40),
        100 / 700,
        id="partial-overlap",
    ),
    # One inside another
    pytest.param(
        (100, 100, 200, 300),
        (110, 120, 190, 280),
        0.64,
        id="One-inside-another",
    ),

    # Touching edge: zero intersection area
    pytest.param(
        (10, 10, 20, 20),
        (20, 10, 30, 20),
        0.0,
        id="Touching-edge:-zero-intersection area",
    ),
    ]
)
def test_compute_iou(box_a, box_b, expected):
    assert compute_iou(box_a, box_b) == pytest.approx(expected)

# Cas invalides
INVALID_BOXES = [
    # À compléter
    pytest.param((100, 200, 90, 300), id="x2_inf_x1)"), # x2 < x1
    pytest.param((100, 200, 100, 300), id="x2_eq_x1)"), # x2 == x1
    pytest.param((100, 300, 110, 200), id="y2_inf_y1)"), # y2 < y1
    pytest.param((100, 200, 110, 200), id="y2_eq_y1)"),  # y2 == y1
    ]

@pytest.mark.parametrize("invalid_box", INVALID_BOXES,)
def test_compute_iou_rejects_invalid_boxes_first(invalid_box):
    valid_box = (10, 10, 20, 20)

    with pytest.raises(ValueError):
        compute_iou(invalid_box, valid_box)

@pytest.mark.parametrize("invalid_box", INVALID_BOXES,)
def test_compute_iou_rejects_invalid_boxes_second(invalid_box):
    valid_box = (10, 10, 20, 20)

    with pytest.raises(ValueError):
        compute_iou(valid_box, invalid_box)

def test_compute_iou_is_symmetric():
    box_a = (10, 10, 30, 30)
    box_b = (20, 20, 40, 40)
    assert compute_iou(box_a, box_b) == pytest.approx(compute_iou(box_b, box_a))

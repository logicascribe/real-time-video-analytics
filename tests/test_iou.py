from enum import Enum, auto
import numpy as np
import pytest
from pytest import approx
from IoUTools.compute_iou import compute_iou, compute_iou_matrix, _as_boxes_array


###############
# Compute IoU #
###############

@pytest.mark.parametrize(
    "box_a, box_b, expected",
    [
    # Identical
    pytest.param(
        (10, 10, 30, 30),
        (10, 10, 30, 30),
        1.0, id = "Identical",
    ),

    # No overlap
    pytest.param(
        (10, 10, 20, 20),
        (30, 30, 40, 40),
        0.0, id = "No-overlap",
    ),

    # Partial overlap
#    ((10, 10, 30, 30),
#    (20, 20, 40, 40),
#    100 / 700),
    pytest.param(
        (10, 10, 30, 30),
        (20, 20, 40, 40),
        100 / 700, id = "partial-overlap",
    ),
    # One inside another
    pytest.param(
        (100, 100, 200, 300),
        (110, 120, 190, 280),
        0.64, id = "One-inside-another",
    ),

    # Touching edge: zero intersection area
    pytest.param(
        (10, 10, 20, 20),
        (20, 10, 30, 20),
        0.0, id = "Touching-edge:-zero-intersection area",
    ),
    ]
)
def test_compute_iou(box_a, box_b, expected):
    assert compute_iou(box_a, box_b) == pytest.approx(expected)

# Cas invalides
INVALID_BOXES = [
    pytest.param((100, 200, 90, 300), id = "x2_inf_x1"), # x2 < x1
    pytest.param((100, 200, 100, 300), id = "x2_eq_x1"), # x2 == x1
    pytest.param((100, 300, 110, 200), id = "y2_inf_y1"), # y2 < y1
    pytest.param((100, 200, 110, 200), id = "y2_eq_y1"),  # y2 == y1
    ]


##################
# Validity check #
##################

class ValidationResult(Enum):
    VALID = auto()
    TYPE_ERROR = auto()
    VALUE_ERROR = auto()

VALID = ValidationResult.VALID
TYPE_ERROR = ValidationResult.TYPE_ERROR
VALUE_ERROR = ValidationResult.VALUE_ERROR
BOX_VALIDITY_CHECK = [
    pytest.param(
        [
            (10, 10, 20, 20),
            (30, 30, 40, 40)
            ],
        VALID,
        id = "integer_boxes"),
    pytest.param(
        [
            (10.0, 10.1, 20.5, 20.6),
            (30.2, 30.3, 40.4, 40.5)
            ],
        VALID,
        id = "float_boxes"),
    pytest.param(
        [
            ()
            ],
        VALUE_ERROR,
        id="collection_with_empty_rectangle_with_brackets"),
    pytest.param(
            (),
        VALID,
        id = "collection_with_empty_rectangle"),
    pytest.param(
        [
            (10, 10, 20, 20)
            ],
        VALID,
        id = "one_boxes"),
    pytest.param(
        [
            (10, 10, 20, 20),
            (30, 30, 40, 40),
            (35, 35, 45, 45),
            (55, 55, 65, 65)],
        VALID,
        id = "several_boxes"),
    pytest.param(
        (10, 10, 20, 20),
        VALUE_ERROR,
        id = "shape(4,)"),
    pytest.param(
        [
            (10, 10, 20)],
        VALUE_ERROR,
        id = "wrong_dimension"),
    pytest.param(
        [
            ("A", 10, 20, 20)],
        TYPE_ERROR,
        id = "non_numeric"),
    pytest.param(
        [
            (np.nan, 10, 20, 20)],
        VALUE_ERROR,
        id = "nan_box"),
    pytest.param(
        [
            (np.inf, 10, 20, 20)
            ],
        VALUE_ERROR,
        id = "inf_box"),
    pytest.param(
        [
            (20, 10, 20, 20)
            ],
        VALUE_ERROR,
        id = "x1==x2_box"),
    pytest.param(
        [
            (25, 10, 20, 20)],
        VALUE_ERROR,
        id = "x1>x2_box"),
    pytest.param(
        [
            (15, 20, 20, 20)
            ],
        VALUE_ERROR,
        id = "y1==y2_box"),
    pytest.param(
        [
            (10, 25, 20, 20)],
        VALUE_ERROR,
        id = "y1>y2_box"),
        ]

def is_valid_boxes(boxes) -> ValidationResult:
    try:
        _as_boxes_array(boxes)
    except ValueError:
        return VALUE_ERROR
    except TypeError:
        return TYPE_ERROR
    return VALID

@pytest.mark.parametrize("boxes, expected_validity", BOX_VALIDITY_CHECK)
def test_validity_check(boxes, expected_validity):
    validity = is_valid_boxes(boxes)
    assert(expected_validity == validity)

@pytest.mark.parametrize("boxes, expected_validity", BOX_VALIDITY_CHECK)
def test_list_nparray_conversion(boxes, expected_validity):
    if not (expected_validity == VALID) or (0 == len(boxes)):
        pytest.skip("Conversion checked only for valid non-empty boxes")
    retVal = _as_boxes_array(boxes)
    raw = np.asarray(boxes)
    box_array = raw.astype(np.float32, copy=False)
    assert np.array_equal(retVal, box_array)


@pytest.mark.parametrize(
    "boxes",
    [
        pytest.param(
            (),
            id="empty_tuple",
        ),
        pytest.param(
            [],
            id="empty_list",
        ),
        pytest.param(
            np.empty((0, 4), dtype=np.float32),
            id="empty_numpy_array",
        ),
    ],
)
def test_empty_boxes_conversion(boxes):
    boxes_in_array = _as_boxes_array(boxes)

    assert boxes_in_array.shape == (0, 4)
    assert boxes_in_array.dtype == np.float32


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



def _compute_iou_matrix_reference(reference_boxes, detected_boxes):
    retVal = []
    for ref_box in reference_boxes:
        row = []
        for detect_box in detected_boxes:
            row.append(compute_iou(ref_box, detect_box))
        retVal.append(row)
    return retVal


###################
# IoU matrix case #
###################

BOX_IOU_MATRIX_CASES = [
    pytest.param(
        [
            (10, 10, 20, 20),
            (30, 30, 40, 40)
            ],
        [
            (15, 15, 25, 25),
            (35, 35, 45, 45),
            (55, 55, 65, 65)
            ],
        id = "ref2_det3"),
    pytest.param(
        [
            # ref =
            (10, 10, 20, 20),
            (30, 30, 40, 40),
            ],
        [
            # det =
            (15, 15, 25, 25),
            (37, 32, 48, 45),
            #        (50, 62, 67, 75),
            ],
        id = "ref2_det2"),
    pytest.param(
        [
            (0, 0, 20, 20),
            (10, 10, 30, 30),
            ],
        [
            (5, 5, 15, 15),
            (10, 10, 20, 20),
            (25, 25, 35, 35),
            ],
        id="multiple_overlaps",
        )
        ]
@pytest.mark.parametrize(
    "ref_boxes, det_boxes",
    BOX_IOU_MATRIX_CASES,)
def test_iou_matrix_matches_reference(ref_boxes, det_boxes):
    matRef = _compute_iou_matrix_reference(ref_boxes, det_boxes)
    matUT = compute_iou_matrix(ref_boxes, det_boxes)
    np.testing.assert_allclose(matRef, matUT)


@pytest.mark.parametrize(
    "ref_boxes, det_boxes, expected_shape",
    [
        pytest.param(
            np.empty((0, 4), dtype=np.float32),
            [
                (15, 15, 25, 25),
                (35, 35, 45, 45),
                (55, 55, 65, 65),
            ],
            (0, 3),
            id="ref0_det3",
        ),
        pytest.param(
            [
                (15, 15, 25, 25),
                (55, 55, 65, 65),
            ],
            np.empty((0, 4), dtype=np.float32),
            (2, 0),
            id="ref2_det0",
        ),
        pytest.param(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            (0, 0),
            id="ref0_det0",
        ),
    ],
)
def test_iou_matrix_empty_shapes(
    ref_boxes,
    det_boxes,
    expected_shape,
):
    matrix = compute_iou_matrix(ref_boxes, det_boxes)

    assert matrix.shape == expected_shape
    assert matrix.dtype == np.float32

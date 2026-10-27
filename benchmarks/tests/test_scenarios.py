import numpy as np
import pytest

from IoUTools.compute_iou import compute_iou_matrix

from ..scenarios import (
    make_isolated_scenario,
    make_ambiguous_chain_scenario,
)

def test_isolated_scenario_20():
    scenario = make_isolated_scenario(20, "isolated_20")

    assert scenario.reference_count == 20
    assert scenario.detection_count == 20
    assert scenario.pattern.reference_boxes.shape == (20, 4)
    assert scenario.pattern.detection_boxes.shape == (20, 4)
    assert scenario.name == "isolated_20"
    assert scenario.threshold == pytest.approx(0.5)

    observed = (
        compute_iou_matrix(
            scenario.pattern.reference_boxes,
            scenario.pattern.detection_boxes,
            )
        >= scenario.threshold
        )

    np.testing.assert_array_equal(
        observed,
        scenario.pattern.target_adjacency,
        )

def test_ambiguous_chain_scenario_20():
    scenario_name = "ambiguous_20"
    scenario = make_ambiguous_chain_scenario(20, scenario_name)

    assert scenario.reference_count == 20
    assert scenario.detection_count == 21
    assert scenario.pattern.reference_boxes.shape == (20, 4)
    assert scenario.pattern.detection_boxes.shape == (21, 4)
    assert scenario.pattern.target_adjacency.shape == (20, 21)
    assert scenario.name == scenario_name
    assert scenario.threshold == pytest.approx(0.5)

    observed = (
        compute_iou_matrix(
            scenario.pattern.reference_boxes,
            scenario.pattern.detection_boxes,
            )
        >= scenario.threshold
        )

    np.testing.assert_array_equal(
        observed,
        scenario.pattern.target_adjacency,
        )

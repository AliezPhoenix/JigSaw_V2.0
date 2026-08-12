"""SizeDetector detect_direction (outward / inward) unit tests."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.size_detector import SizeDetector


@pytest.mark.parametrize(
    "direction,expected",
    [
        ("outward", {"top": True, "left": True, "bottom": False, "right": False}),
        (None, {"top": True, "left": True, "bottom": False, "right": False}),
        ("foo", {"top": True, "left": True, "bottom": False, "right": False}),
        ("inward", {"top": False, "left": False, "bottom": True, "right": True}),
    ],
)
def test_is_reverse_for_side_matrix(direction, expected):
    for side, want in expected.items():
        assert SizeDetector._is_reverse_for_side(side, direction) is want


def test_normalize_detect_direction():
    assert SizeDetector.normalize_detect_direction("inward") == "inward"
    assert SizeDetector.normalize_detect_direction("outward") == "outward"
    assert SizeDetector.normalize_detect_direction("foo") == "outward"
    assert SizeDetector.normalize_detect_direction(None) == "outward"


def test_update_params_normalizes_illegal_direction():
    det = SizeDetector()
    assert det.update_params({"detect_direction": "bogus"}) is True
    assert det.params["detect_direction"] == "outward"
    assert det.update_params({"detect_direction": "inward"}) is True
    assert det.params["detect_direction"] == "inward"


def test_detect_outward_with_default_rois_valid_box():
    """合成白矩形黑底：outward 能给出有效 box_points。"""
    h, w = 200, 300
    image = np.zeros((h, w), dtype=np.uint8)
    image[40:160, 50:250] = 255

    det = SizeDetector()
    det.update_params(
        {
            "min_threshold": 128,
            "max_threshold": 255,
            "rois": SizeDetector.default_rois(w, h, strip=40),
            "std_size": (0.0, 0.0),
            "pixel_size": 0.01,
            "detect_direction": "outward",
        }
    )
    result = det.detect(image)
    assert result.error_code == 0
    assert result.box_points is not None
    assert len(result.box_points) == 4
    x, y, bw, bh = result.box_points
    assert bw > 0 and bh > 0
    # 大致落在白矩形附近（亚像素/条带投影容差）
    assert 30 < x < 80
    assert 20 < y < 70
    assert 140 < bw < 230
    assert 70 < bh < 150

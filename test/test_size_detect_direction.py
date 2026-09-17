"""SizeDetector detect_direction (outward / inward) unit tests."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.size_detector import (
    SizeDetector,
    compensate_box_toward_std,
    compensation_delta_px,
    detect_boundary_subpixel,
)


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


def test_boundary_rising_vs_falling_polarity():
    """黑→白（rising）与白→黑（falling）应分别落在上升/下降沿附近。"""
    # 索引 0..4 黑，5..8 白，9 黑
    curve = np.array([0, 0, 0, 0, 0, 255, 255, 255, 255, 0], dtype=np.float64)
    falling = detect_boundary_subpixel(curve, edge_polarity="falling", smooth_window=1)
    rising = detect_boundary_subpixel(curve, edge_polarity="rising", smooth_window=1)
    assert falling is not None and rising is not None
    # falling 找白→黑，应靠近末端下降
    assert falling > 7
    # rising 找黑→白，应靠近前段上升
    assert 3 < rising < 6


def _run_detect(direction):
    h, w = 200, 300
    image = np.zeros((h, w), dtype=np.uint8)
    # 白产品需伸入默认四边条带 ROI，否则左右 ROI 全黑无边
    image[25:175, 30:270] = 255
    det = SizeDetector()
    det.update_params(
        {
            "min_threshold": 128,
            "max_threshold": 255,
            "rois": SizeDetector.default_rois(w, h, strip=80),
            "std_size": (0.0, 0.0),
            "pixel_size": 0.01,
            "detect_direction": direction,
        }
    )
    return det.detect(image)


def test_detect_outward_with_default_rois_valid_box():
    """合成白矩形黑底：outward 能给出有效 box_points。"""
    result = _run_detect("outward")
    assert result.error_code == 0
    assert result.box_points is not None
    assert len(result.box_points) == 4
    x, y, bw, bh = result.box_points
    assert bw > 0 and bh > 0
    assert 15 < x < 50
    assert 10 < y < 45
    assert 200 < bw < 260
    assert 120 < bh < 170


@pytest.mark.parametrize(
    "delta,want",
        [
            (0.0, 0.0),
            (0.99, 0.0),
            (1.0, 0.0),
            (2.0, -0.5),
            (3.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
            (-2.0, 0.5),
            (-5.0, 0.0),
        ],
)
def test_compensation_delta_px_deadband_frac(delta, want):
    assert compensation_delta_px(delta) == pytest.approx(want)


def test_compensate_box_keeps_center_when_too_large():
    x, y, w, h = compensate_box_toward_std(
        10.0, 20.0, 102.0, 80.0, std_w_mm=1.0, std_h_mm=0.8, scale_x=0.01, scale_y=0.01
    )
    # 宽 102px vs 100px → Δ=2 → 减 0.5px，中心不变；高 Δ=0 不补
    assert w == pytest.approx(101.5)
    assert x == pytest.approx(10.25)
    assert h == pytest.approx(80.0)
    assert y == pytest.approx(20.0)
    assert (x + w / 2) == pytest.approx(10.0 + 102.0 / 2)


def test_compensate_box_grows_when_too_small():
    x, y, w, h = compensate_box_toward_std(
        10.0, 20.0, 98.0, 78.0, std_w_mm=1.0, std_h_mm=0.8, scale_x=0.01, scale_y=0.01
    )
    # Δw=Δh=-2 → 各加 0.5px
    assert w == pytest.approx(98.5)
    assert x == pytest.approx(9.75)
    assert h == pytest.approx(78.5)
    assert y == pytest.approx(19.75)


def test_compensate_box_skips_without_std():
    x, y, w, h = compensate_box_toward_std(
        1, 2, 10, 20, std_w_mm=0.0, std_h_mm=0.0, scale_x=0.01, scale_y=0.01
    )
    assert (x, y, w, h) == (1.0, 2.0, 10.0, 20.0)


def test_detect_inward_with_default_rois_valid_box():
    """inward：反转搜索方向 + 黑→白极性，仍能给出有效尺寸框。"""
    result = _run_detect("inward")
    assert result.error_code == 0
    x, y, bw, bh = result.box_points
    assert bw > 0 and bh > 0
    assert 15 < x < 50
    assert 10 < y < 45
    assert 200 < bw < 260
    assert 120 < bh < 170


def test_legacy_algorithm_still_detects_white_rectangle():
    """Transfer 旧投影路径：algorithm=legacy 仍可用。"""
    h, w = 200, 300
    image = np.zeros((h, w), dtype=np.uint8)
    image[25:175, 30:270] = 255
    det = SizeDetector()
    det.update_params(
        {
            "algorithm": "legacy",
            "min_threshold": 128,
            "max_threshold": 255,
            "rois": SizeDetector.default_rois(w, h, strip=80),
            "std_size": (0.0, 0.0),
            "pixel_size": 0.01,
            "detect_direction": "outward",
        }
    )
    result = det.detect(image)
    assert result.error_code == 0
    x, y, bw, bh = result.box_points
    assert bw > 0 and bh > 0
    assert 15 < x < 50
    assert 10 < y < 45


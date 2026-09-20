"""点击绿格显示：尺寸算法失败仍落盘，并按结果重绘叠加。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.support.data_structure import Product, Size_Result, Shift_Result
from src.support.support_funs import (
    execute_product_detection,
    resolve_product_overlay_patch,
    size_result_has_measurement,
    shift_result_has_measurement,
    product_needs_display_refresh,
    refresh_product_results_for_display,
    draw_dry_product_metrics,
    paint_product_image_result,
    layout_dry_metrics_panel,
)


class _DummyDetector:
    def __init__(self, result):
        self.result = result
        self.params = {"expected_ball_count": 0}

    def detect(self, *args, **kwargs):
        return self.result


def _blank_image():
    return np.zeros((80, 80), dtype=np.uint8)


def _size_only_params():
    return {
        "mark_check_enable": False,
        "size_check_enable": True,
        "ball_check_enable": False,
        "shift_check_enable": False,
        "scratch_check_enable": False,
    }


def test_size_algorithm_failure_is_size_ng_not_ok():
    failed = Size_Result(error_code=2, error_msg="top边拟合残差过大", is_valid=False)
    success, msg, product = execute_product_detection(
        image=_blank_image(),
        detectors={"size_detector": _DummyDetector(failed)},
        params=_size_only_params(),
        early_return_on_ng=True,
    )
    assert success is True
    assert product.size_result.error_code == 2
    assert product.size_result.error_msg == "top边拟合残差过大"
    assert "OK" not in product.defect_type
    assert "Size" in product.defect_type


def test_size_success_still_ok_when_valid():
    ok = Size_Result(width=10.1, height=15.2, box_points=[5, 6, 40, 50], is_valid=True)
    success, _, product = execute_product_detection(
        image=_blank_image(),
        detectors={"size_detector": _DummyDetector(ok)},
        params=_size_only_params(),
        early_return_on_ng=True,
    )
    assert success is True
    assert product.defect_type == ["OK"]
    assert product.size_result.width == 10.1


def test_overlay_redraws_size_box_even_if_result_image_is_raw():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    product = Product()
    product.product_image = image.copy()
    product.product_image_result = image.copy()
    product.size_result = Size_Result(
        width=10.0, height=12.0, box_points=[10, 10, 30, 30], is_valid=True
    )
    patch = resolve_product_overlay_patch(product)
    assert patch is not None
    assert not np.array_equal(patch, image)
    assert tuple(patch[10, 20]) == (0, 255, 0)


def test_empty_size_result_needs_refresh():
    product = Product()
    product.product_image = _blank_image()
    assert size_result_has_measurement(product.size_result) is False
    assert product_needs_display_refresh(product) is True


def test_refresh_copies_size_without_changing_defect_type():
    product = Product(defect_type=["OK"])
    product.product_image = np.zeros((80, 80, 3), dtype=np.uint8)
    measured = Size_Result(
        width=9.5, height=11.5, box_points=[8, 8, 20, 20], is_valid=True
    )
    refreshed = refresh_product_results_for_display(
        product,
        detectors={"size_detector": _DummyDetector(measured)},
        params=_size_only_params(),
    )
    assert refreshed is True
    assert product.defect_type == ["OK"]
    assert product.size_result.width == 9.5
    assert product.product_image_result is not None
    assert tuple(product.product_image_result[8, 18]) == (0, 255, 0)


def test_hud_skips_zero_placeholder_and_draws_real_size():
    canvas = np.zeros((200, 400, 3), dtype=np.uint8)
    empty = Product()
    before = canvas.copy()
    draw_dry_product_metrics(canvas, empty, 40, 40, 80, 80)
    assert not np.array_equal(canvas, before)
    assert canvas[0:15, 0:15].sum() == 0
    assert canvas[:, 120:].sum() > 0

    measured = Product()
    measured.size_result = Size_Result(
        width=10.123, height=15.456, box_points=[1, 1, 10, 10], is_valid=True
    )
    canvas2 = np.zeros((200, 400, 3), dtype=np.uint8)
    draw_dry_product_metrics(canvas2, measured, 40, 40, 80, 80)
    assert canvas2.sum() > 0
    assert canvas2[0:15, 0:15].sum() == 0
    assert shift_result_has_measurement(Shift_Result()) is False


def test_hud_layout_right_of_product_when_gap_exists():
    ox, oy, panel_w, panel_h, *_ = layout_dry_metrics_panel(240, 240, 80, 80, 80, 80, 3)
    assert ox >= 80 + 80
    assert abs(oy - int(80 + 80 / 10)) <= 2
    assert ox + panel_w <= 240
    assert oy + panel_h <= 240


def test_hud_layout_covers_product_right_when_no_gap():
    ox, oy, panel_w, panel_h, *_ = layout_dry_metrics_panel(240, 240, 160, 80, 80, 80, 3)
    assert ox + panel_w <= 240
    assert ox < 160 + 80
    assert ox >= 160
    assert not (ox <= 8 and oy <= 8)
    assert abs(oy - int(80 + 80 / 10)) <= 2


def test_paint_keeps_result_when_no_geometry():
    product = Product()
    product.product_image = np.full((20, 20, 3), 40, dtype=np.uint8)
    patch = paint_product_image_result(product)
    assert patch is not None
    assert patch.shape == (20, 20, 3)

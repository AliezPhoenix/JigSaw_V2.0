# -*- coding: utf-8 -*-
"""在整版定位结果上逐颗测量塑封外形。

每颗以定位中心裁出约一个节距的窗口，再交给 QfnSizeDetector。
合格：宽和高都在 2.95–3.05 mm。测不出边算不合格。
塑封亮区与邻颗连成一片、中间没有暗缝的，记为接壤并保持不合格。
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from src.detectors.qfn_size_detector import QfnSizeDetector

SIZE_LOW_MM = 2.95
SIZE_HIGH_MM = 3.05
# 整版上外轮廓比单颗模板的标注略偏外，向内收这一段后尺寸中心落在 3.00 mm。
STRIP_EDGE_BIAS = {
    "edge_bias_left": 2.2,
    "edge_bias_right": 1.6,
    "edge_bias_top": 2.4,
    "edge_bias_bottom": 1.6,
}


def measure_strip(image, located, params=None):
    """对 located['units'] 逐颗测尺寸。返回合格数、比例和带尺寸的单元列表。"""
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    pitch = min(float(located["pitch_x"]), float(located["pitch_y"]))
    half = max(16, int(round(pitch * 0.5)))
    detector = QfnSizeDetector({**STRIP_EDGE_BIAS, **(params or {})})
    touching = _touching_units(gray, located["units"])
    units = []
    passed = 0
    skipped = 0
    for unit in located["units"]:
        cx = int(round(unit["x"]))
        cy = int(round(unit["y"]))
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(width, cx + half)
        y1 = min(height, cy + half)
        result = detector.detect(gray[y0:y1, x0:x1])
        item = {
            "row": int(unit["row"]),
            "col": int(unit["col"]),
            "x": float(unit["x"]),
            "y": float(unit["y"]),
            "passed": False,
            "width_mm": None,
            "height_mm": None,
            "message": None,
            "box": None,
            "touching": False,
        }
        if result.error_code == 0:
            item["width_mm"] = round(float(result.width), 4)
            item["height_mm"] = round(float(result.height), 4)
            box = result.box_points
            item["box"] = [
                round(x0 + float(box[0]), 2),
                round(y0 + float(box[1]), 2),
                round(float(box[2]), 2),
                round(float(box[3]), 2),
            ]
            item["passed"] = (
                SIZE_LOW_MM <= result.width <= SIZE_HIGH_MM
                and SIZE_LOW_MM <= result.height <= SIZE_HIGH_MM
            )
            if not item["passed"]:
                item["message"] = "尺寸超出 2.95–3.05 mm"
        else:
            item["message"] = result.error_msg or "尺寸检测失败"
        if (item["row"], item["col"]) in touching:
            item["touching"] = True
            item["passed"] = False
            item["message"] = "与邻颗塑封亮区接壤，跳过"
            skipped += 1
        elif item["passed"]:
            passed += 1
        units.append(item)
    total = len(located["units"])
    rate = (passed / total) if total else 0.0
    return {
        "passed": passed,
        "skipped": skipped,
        "total": total,
        "rate": rate,
        "accepted": rate >= 0.80,
        "limit_low_mm": SIZE_LOW_MM,
        "limit_high_mm": SIZE_HIGH_MM,
        "units": units,
    }


def _touching_units(gray, units, level=160):
    """亮区阈值下与邻颗属于同一连通域，说明中间没有可分开的暗缝。"""
    by = {(int(unit["row"]), int(unit["col"])): unit for unit in units}
    touched = set()
    for unit in units:
        for step in ((0, 1), (1, 0)):
            other = by.get((int(unit["row"]) + step[0], int(unit["col"]) + step[1]))
            if other is None or not _bright_linked(gray, unit, other, level):
                continue
            touched.add((int(unit["row"]), int(unit["col"])))
            touched.add((int(other["row"]), int(other["col"])))
    return touched


def _bright_linked(gray, unit, other, level):
    ax, ay = int(round(unit["x"])), int(round(unit["y"]))
    bx, by = int(round(other["x"])), int(round(other["y"]))
    x0, x1 = sorted((ax, bx))
    y0, y1 = sorted((ay, by))
    pad = 8
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(gray.shape[1] - 1, x1 + pad)
    y1 = min(gray.shape[0] - 1, y1 + pad)
    crop = gray[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return False
    _count, labels = cv.connectedComponents((crop >= level).astype(np.uint8), 8)
    left = labels[ay - y0, ax - x0]
    right = labels[by - y0, bx - x0]
    return int(left) > 0 and int(left) == int(right)

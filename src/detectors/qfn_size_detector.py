# -*- coding: utf-8 -*-
"""单颗 QFN 塑封体外形尺寸。

亮塑封、暗背景，边缘与图像轴对齐。输入是一颗外包矩形 ROI；
未给 ROI 时把整幅图当作这一颗。四边测量窗由 ROI、搜索深度和角点排除推出。

塑封边是外轮廓的二值交点：沿扫描方向平滑后，外缘带里梯度最大的位置是上升沿。
背景只取这个峰前面几个像素，避开更外侧的邻颗亮区。再用背景到峰顶的一定比例
作为阈值，求每条扫描线从外向内第一次升过该阈值的亚像素位置。左右亮边用
较高比例，靠近饱和起点；上边薄脊用接近峰值的比例；下边弱脊用较低比例，
落在外肩。引脚缺口与这条边错开的样本当作离群点去掉。edge_bias_* 为正时
把边向塑封内侧移动。
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from src.support.data_structure import Size_Result
from src.support.support_funs import ensure_gray_u8

_SIDES = ("left", "right", "top", "bottom")


class QfnSizeDetector:
    def __init__(self, params: dict | None = None):
        self.image = None
        self.detection_result = None
        self.last_debug = {}
        self.params = {
            "roi": None,
            "band_width": 0,
            "corner_exclude": 0,
            "inner_margin": 18,
            "min_contrast": 20.0,
            "min_gradient": 6.0,
            "body_drop_ratio": 0.40,
            "min_body_span": 8,
            "smooth_sigma": 0.8,
            "outlier_px": 1.5,
            "min_inliers": 16,
            "binary_ratio_x": 0.93,
            "binary_ratio_top": 0.99,
            "binary_ratio_bottom": 0.37,
            "edge_bias_x": 0.0,
            "edge_bias_y": 0.0,
            "edge_bias_left": 1.4158,
            "edge_bias_right": 0.7117,
            "edge_bias_top": 1.4781,
            "edge_bias_bottom": -0.1615,
            "pixel_size": 0.014,
            "pixel_size_x": None,
            "std_size": (0.0, 0.0),
            "allow_tolerance_x": 0.0,
            "allow_tolerance_y": 0.0,
        }
        if params:
            self.update_params(params)

    def get_params(self):
        return self.params.copy()

    def update_params(self, params: dict, clear_result: bool = True):
        if not params:
            return True
        errors = _validate_params(params)
        if errors:
            print("参数验证失败:\n" + "\n".join(f"  - {item}" for item in errors))
            return False
        for key in self.params:
            if key in params:
                self.params[key] = params[key]
        if clear_result:
            self.detection_result = None
            self.last_debug = {}
        return True

    def detect(self, image):
        try:
            gray = ensure_gray_u8(image, copy=True)
        except (TypeError, ValueError, cv.error) as exc:
            return self._fail(str(exc))
        self.image = gray
        height, width = gray.shape
        roi, roi_error = _normalize_roi(self.params.get("roi"), width, height)
        if roi_error is not None:
            return self._fail(roi_error)

        windows, window_error = self._side_windows(gray, roi)
        if window_error is not None:
            return self._fail(window_error, {"roi": roi})

        measured = {}
        for side, window in windows.items():
            edge, info = _measure_side(gray, side, window, self.params)
            measured[side] = info
            if edge is None:
                return self._fail(
                    info.get("message", f"{side}边检测失败"),
                    {"roi": roi, "windows": windows, "sides": measured},
                )

        bias = {side: _inward_bias(self.params, side) for side in _SIDES}
        for side, inward in bias.items():
            _shift_edge(measured[side], side, inward)
        left = measured["left"]["edge"]
        right = measured["right"]["edge"]
        top = measured["top"]["edge"]
        bottom = measured["bottom"]["edge"]
        width_px = right - left
        height_px = bottom - top
        if width_px <= 0 or height_px <= 0:
            return self._fail("尺寸边界顺序无效", measured)

        pixel_size_y = float(self.params.get("pixel_size", 0.014))
        pixel_size_x = self.params.get("pixel_size_x")
        pixel_size_x = pixel_size_y if pixel_size_x is None else float(pixel_size_x)
        width_mm = width_px * pixel_size_x
        height_mm = height_px * pixel_size_y
        std_width, std_height = self.params.get("std_size", (0.0, 0.0))
        tolerance_x = float(self.params.get("allow_tolerance_x", 0.0))
        tolerance_y = float(self.params.get("allow_tolerance_y", 0.0))
        is_valid = True
        if float(std_width) > 0:
            is_valid = is_valid and abs(width_mm - float(std_width)) <= tolerance_x
        if float(std_height) > 0:
            is_valid = is_valid and abs(height_mm - float(std_height)) <= tolerance_y

        box_points = [float(left), float(top), float(width_px), float(height_px)]
        self.last_debug = {
            "roi": roi,
            "windows": windows,
            "sides": measured,
            "width_px": float(width_px),
            "height_px": float(height_px),
            "width_mm": float(width_mm),
            "height_mm": float(height_mm),
        }
        self.detection_result = Size_Result(
            width=float(width_mm),
            height=float(height_mm),
            box_points=box_points,
            is_valid=bool(is_valid),
        )
        return self.detection_result

    def _fail(self, message, debug=None):
        self.last_debug = debug or {}
        self.detection_result = Size_Result(
            error_code=2, error_msg=message, is_valid=False
        )
        return self.detection_result

    def _side_windows(self, gray, roi):
        x, y, w, h = roi
        band_width = int(self.params.get("band_width") or 0)
        view = gray[y:y + h, x:x + w]
        if band_width > 0:
            body = (0, 0, w - 1, h - 1)
        else:
            body = _rough_body_box(view, float(self.params.get("min_contrast", 20.0)))
            if body is None:
                return None, "ROI 内找不到亮塑封体"
        left, top, right, bottom = body
        corner = _corner_exclude(
            self.params.get("corner_exclude") or 0,
            right - left + 1,
            bottom - top + 1,
        )
        margin = max(int(self.params.get("inner_margin", 18)), int(self.params.get("min_body_span", 8)))
        lateral_y0 = top + corner
        lateral_y1 = bottom - corner
        lateral_x0 = left + corner
        lateral_x1 = right - corner
        if lateral_y1 - lateral_y0 < int(self.params.get("min_inliers", 16)):
            return None, "上下角点排除后，左右边的直线段太短"
        if lateral_x1 - lateral_x0 < int(self.params.get("min_inliers", 16)):
            return None, "左右角点排除后，上下边的直线段太短"

        if band_width > 0:
            depth = min(band_width, w // 2 - 1, h // 2 - 1)
            if depth < margin + 2:
                return None, "band_width 小于进入塑封体所需的深度"
            windows = {
                "left": (x, y + lateral_y0, x + depth, y + lateral_y1 + 1),
                "right": (x + w - 1 - depth, y + lateral_y0, x + w - 1, y + lateral_y1 + 1),
                "top": (x + lateral_x0, y, x + lateral_x1 + 1, y + depth),
                "bottom": (x + lateral_x0, y + h - 1 - depth, x + lateral_x1 + 1, y + h - 1),
            }
        else:
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            windows = {
                "left": (
                    x,
                    y + lateral_y0,
                    x + min(left + margin, center_x),
                    y + lateral_y1 + 1,
                ),
                "right": (
                    x + max(right - margin, center_x),
                    y + lateral_y0,
                    x + w - 1,
                    y + lateral_y1 + 1,
                ),
                "top": (
                    x + lateral_x0,
                    y,
                    x + lateral_x1 + 1,
                    y + min(top + margin, center_y),
                ),
                "bottom": (
                    x + lateral_x0,
                    y + max(bottom - margin, center_y),
                    x + lateral_x1 + 1,
                    y + h - 1,
                ),
            }
        return windows, None


def _validate_params(params):
    errors = []
    numeric_ranges = {
        "band_width": (0, 5000),
        "corner_exclude": (0, 5000),
        "inner_margin": (4, 500),
        "min_contrast": (1, 255),
        "min_gradient": (0, 255),
        "body_drop_ratio": (0.15, 0.8),
        "min_body_span": (3, 200),
        "smooth_sigma": (0, 5),
        "outlier_px": (0.3, 20),
        "min_inliers": (4, 5000),
        "binary_ratio_x": (0.05, 0.99),
        "binary_ratio_top": (0.05, 0.99),
        "binary_ratio_bottom": (0.05, 0.99),
        "edge_bias_x": (-5, 5),
        "edge_bias_y": (-5, 5),
        "edge_bias_left": (-5, 5),
        "edge_bias_right": (-5, 5),
        "edge_bias_top": (-5, 5),
        "edge_bias_bottom": (-5, 5),
        "allow_tolerance_x": (0, None),
        "allow_tolerance_y": (0, None),
        "pixel_size": (1e-6, None),
    }
    for key, (low, high) in numeric_ranges.items():
        if key not in params or params[key] is None:
            continue
        value = params[key]
        if not isinstance(value, (int, float)):
            errors.append(f"{key} 必须是数字，当前值: {value}")
            continue
        if value < low or (high is not None and value > high):
            errors.append(f"{key} 超出范围，当前值: {value}")
    if "pixel_size_x" in params and params["pixel_size_x"] is not None:
        value = params["pixel_size_x"]
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"pixel_size_x 必须是正数，当前值: {value}")
    if "std_size" in params:
        value = params["std_size"]
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or not all(isinstance(item, (int, float)) and item >= 0 for item in value)
        ):
            errors.append(f"std_size 必须是两个非负数字，当前值: {value}")
    if "roi" in params and params["roi"] is not None:
        value = params["roi"]
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            errors.append(f"roi 必须是 (x, y, w, h)，当前值: {value}")
    return errors


def _normalize_roi(roi, width, height):
    if roi is None:
        if width < 16 or height < 16:
            return None, "图像过小"
        return (0, 0, int(width), int(height)), None
    try:
        x, y, w, h = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    except (TypeError, ValueError, IndexError):
        return None, "roi 必须是 (x, y, w, h)"
    if w < 16 or h < 16:
        return None, "roi 过小"
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    if w < 16 or h < 16:
        return None, "roi 超出图像"
    return (x, y, w, h), None


def _corner_exclude(raw, body_width, body_height):
    value = int(raw or 0)
    if value > 0:
        return value
    return max(4, int(round(0.12 * min(body_width, body_height))))


def _rough_body_box(view, min_contrast):
    """从中心向外找连续亮区，避开缺口外侧的亮带。返回 ROI 内含边界的 (l, t, r, b)。"""
    height, width = view.shape
    if height < 16 or width < 16:
        return None
    cy0, cy1 = int(height * 0.35), int(height * 0.65)
    cx0, cx1 = int(width * 0.35), int(width * 0.65)
    inner = float(np.median(view[cy0:cy1, cx0:cx1]))
    dark = float(np.percentile(view, 20))
    if inner - dark < min_contrast:
        return None
    level = dark + 0.55 * (inner - dark)
    col_frac = np.mean(view[cy0:cy1] >= level, axis=0)
    row_frac = np.mean(view[:, cx0:cx1] >= level, axis=1)
    cols = _span_through_center(col_frac, 0.85)
    rows = _span_through_center(row_frac, 0.85)
    if cols is None or rows is None:
        return None
    if cols[1] - cols[0] < 8 or rows[1] - rows[0] < 8:
        return None
    return cols[0], rows[0], cols[1], rows[1]


def _span_through_center(frac, limit):
    center = len(frac) // 2
    if frac[center] < limit:
        hits = np.flatnonzero(frac >= limit)
        if hits.size == 0:
            return None
        center = int(hits[np.argmin(np.abs(hits - center))])
    left = center
    while left > 0 and frac[left - 1] >= limit:
        left -= 1
    right = center
    while right < len(frac) - 1 and frac[right + 1] >= limit:
        right += 1
    return left, right


def _measure_side(gray, side, window, params):
    """window 是搜索带的外边界像素闭区间：(outer 或 inner 的起止)。

    left/right: (x_outer_or_inner, y0, x_other, y1_exclusive)
    这里统一存成 (x0, y0, x1, y1)，x1/y1 为含边界的内或外端，由 side 决定方向。
    """
    x0, y0, x1, y1 = window
    window_rect = _window_rect(side, window)
    if side in ("left", "right"):
        if x1 < x0 or y1 - y0 < 4:
            return None, {"message": f"{side}边搜索窗无效", "window": window, "window_rect": window_rect}
        strip = gray[y0:y1, x0:x1 + 1].astype(np.float32)
        if side == "right":
            strip = strip[:, ::-1]
        origin = float(x0 if side == "left" else x1)
        sign = 1.0 if side == "left" else -1.0
    else:
        if y1 < y0 or x1 - x0 < 4:
            return None, {"message": f"{side}边搜索窗无效", "window": window, "window_rect": window_rect}
        strip = gray[y0:y1 + 1, x0:x1].astype(np.float32).T
        if side == "bottom":
            strip = strip[:, ::-1]
        origin = float(y0 if side == "top" else y1)
        sign = 1.0 if side == "top" else -1.0

    depths = _edge_depths(strip, params, side)
    finite = depths[np.isfinite(depths)]
    outlier_px = float(params.get("outlier_px", 2.0))
    min_inliers = int(params.get("min_inliers", 16))
    consensus = None if finite.size < min_inliers else _consensus(finite, outlier_px, min_inliers)
    center = float(np.median(finite)) if finite.size else None
    if center is None:
        inlier_mask = np.zeros(depths.shape, dtype=bool)
    else:
        inlier_mask = np.isfinite(depths) & (np.abs(depths - center) <= outlier_px)
        if consensus is None:
            inlier_mask[:] = False
    outlier_mask = np.isfinite(depths) & ~inlier_mask
    info = {
        "window": tuple(int(v) for v in window),
        "window_rect": window_rect,
        "samples": int(finite.size),
        "inliers": int(np.count_nonzero(inlier_mask)),
        "inlier_points": _sample_points(side, window, depths, inlier_mask),
        "outlier_points": _sample_points(side, window, depths, outlier_mask),
    }
    if consensus is None:
        if finite.size >= min_inliers and not params.get("_outlier_widened"):
            wide = dict(params)
            wide["outlier_px"] = 4.0
            wide["_outlier_widened"] = True
            return _measure_side(gray, side, window, wide)
        info["message"] = (
            f"{side}边有效样本不足 ({finite.size})"
            if finite.size < min_inliers
            else f"{side}边无法形成稳定塑封边"
        )
        return None, info
    depth, inliers, samples, spread, _center = consensus
    edge = origin + sign * depth
    info.update({
        "edge": float(edge),
        "depth": float(depth),
        "inliers": int(inliers),
        "samples": int(samples),
        "spread_px": float(spread),
    })
    return edge, info


def _edge_depths(profiles, params, side):
    """profiles[:, 0] 在外侧。返回每条扫描线升过二值阈值的亚像素深度。

    阈值由中值轮廓决定：外缘带里梯度最大的位置是上升沿，
    背景取这个峰前面几个像素，亮峰取沿内侧几个像素，再按比例得到二值电平。
    上下边如果锁到了外缘带末端的亮核坡脚，或逐行都找不到交点，再改用外侧那条脊。
    """
    sigma = float(params.get("smooth_sigma", 0.8))
    if sigma > 0:
        profiles = cv.GaussianBlur(
            profiles, (0, 0), sigmaX=sigma, sigmaY=0, borderType=cv.BORDER_REPLICATE
        )
    count, depth = profiles.shape
    if depth < 12:
        return np.full(count, np.nan)
    profile = np.median(profiles, axis=0)
    gradient = np.gradient(profile)
    outer_end = min(22, len(gradient) - 2)
    if outer_end <= 4:
        return np.full(count, np.nan)
    peak_at = 3 + int(np.argmax(gradient[3:outer_end]))
    depths = _depths_from_peak(profiles, profile, peak_at, params, side)
    if side in ("left", "right"):
        return depths
    finite = depths[np.isfinite(depths)]
    ridge = _outer_ridge(profile, gradient, outer_end)
    missing = finite.size < 16
    deep = finite.size >= 16 and float(np.median(finite)) >= 18.0
    if ridge is not None and (missing or (deep and ridge <= float(np.median(finite)) - 6)):
        replacement = _depths_from_peak(profiles, profile, ridge, params, side)
        if int(np.isfinite(replacement).sum()) >= 16:
            return replacement
    if missing:
        filled = _profile_depth(profiles, profile, gradient, peak_at, params, side)
        if filled is not None:
            return filled
    return depths


def _depths_from_peak(profiles, profile, peak_at, params, side):
    gradient = np.gradient(profile)
    if peak_at < 1 or peak_at >= len(profile) - 1:
        return np.full(profiles.shape[0], np.nan)
    if float(gradient[peak_at]) < 0.6:
        return np.full(profiles.shape[0], np.nan)
    # 背景只取峰前几个像素。整段前缀的中值会被邻颗亮区抬高，把真边判成没有对比度。
    base = float(np.median(profile[max(0, peak_at - 5):peak_at]))
    height = float(np.max(profile[peak_at : min(len(profile), peak_at + 5)]))
    if height - base < 2.0:
        return np.full(profiles.shape[0], np.nan)
    level = base + _binary_ratio(params, side) * (height - base)
    stop = min(profiles.shape[1] - 1, peak_at + 5)
    return np.array([_rising_cross(profiles[index], level, stop) for index in range(profiles.shape[0])])


def _outer_ridge(profile, gradient, outer_end):
    """外缘带里最外侧一条亮脊：先升高，随后又落回暗背景。"""
    for index in range(4, outer_end - 1):
        slope = float(gradient[index])
        if slope < 2.0:
            continue
        if slope < float(gradient[index - 1]) or slope < float(gradient[index + 1]):
            continue
        base = float(np.median(profile[max(0, index - 5):index]))
        height = float(np.max(profile[index:index + 4]))
        if height - base < 4.0:
            continue
        after = gradient[index + 1:index + 7]
        if after.size and float(np.min(after)) < -1.0:
            return index
    return None


def _profile_depth(profiles, profile, gradient, peak_at, params, side):
    """逐行交点太少时，用中值轮廓过弱脊的位置作为整条边。"""
    if float(gradient[peak_at]) < 1.0:
        return None
    base = float(np.min(profile[max(0, peak_at - 5):peak_at])) if peak_at else float(profile[0])
    height = float(np.max(profile[peak_at:min(len(profile), peak_at + 5)]))
    if height - base < 2.0:
        return None
    level = base + _binary_ratio(params, side) * (height - base)
    crossed = _rising_cross(profile, level, min(len(profile) - 1, peak_at + 5))
    if not np.isfinite(crossed):
        return None
    return np.full(profiles.shape[0], float(crossed))


def _binary_ratio(params, side):
    if side in ("left", "right"):
        return float(params.get("binary_ratio_x", 0.85))
    if side == "top":
        return float(params.get("binary_ratio_top", 0.99))
    return float(params.get("binary_ratio_bottom", 0.30))


def _rising_cross(line, level, stop):
    """从外侧走到 stop，灰度第一次升过 level 的亚像素位置。"""
    last = min(int(stop), len(line) - 1)
    for index in range(1, last + 1):
        previous = float(line[index - 1])
        current = float(line[index])
        if previous <= level <= current and current > previous:
            return (index - 1) + (level - previous) / (current - previous)
    return np.nan


def _inward_bias(params, side):
    shared = float(params.get("edge_bias_x" if side in ("left", "right") else "edge_bias_y", 0.0))
    return shared + float(params.get(f"edge_bias_{side}", 0.0))


def _shift_edge(info, side, inward):
    """inward 为正时，左/上边界坐标增加，右/下边界坐标减小。"""
    sign = 1.0 if side in ("left", "top") else -1.0
    info["edge"] = float(info["edge"]) + sign * inward
    for key in ("inlier_points", "outlier_points"):
        points = info.get(key) or []
        axis = 0 if side in ("left", "right") else 1
        for point in points:
            point[axis] = float(point[axis]) + sign * inward


def _window_rect(side, window):
    """搜索窗转成像素矩形 (x, y, w, h)。左右边的 y 终点、上下边的 x 终点是开区间。"""
    x0, y0, x1, y1 = (int(v) for v in window)
    if side in ("left", "right"):
        return (min(x0, x1), y0, abs(x1 - x0) + 1, max(0, y1 - y0))
    return (x0, min(y0, y1), max(0, x1 - x0), abs(y1 - y0) + 1)


def _sample_points(side, window, depths, mask):
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return []
    depth = depths[indexes].astype(np.float64)
    index = indexes.astype(np.float64)
    x0, y0, x1, y1 = window
    if side == "left":
        xs, ys = x0 + depth, y0 + index
    elif side == "right":
        xs, ys = x1 - depth, y0 + index
    elif side == "top":
        xs, ys = x0 + index, y0 + depth
    else:
        xs, ys = x0 + index, y1 - depth
    return np.column_stack([xs, ys]).tolist()


def _consensus(values, outlier_px, min_inliers):
    center = float(np.median(values))
    keep = values[np.abs(values - center) <= outlier_px]
    if keep.size < min_inliers:
        return None
    return float(np.median(keep)), int(keep.size), int(values.size), float(np.std(keep)), center

"""Dry 站尺寸检测：外侧最齐陡边 + 局部 50% 交点。

不依赖操作员二值阈值。每条边在 ROI 外侧找「向产品内部变暗或变亮」的陡沿，
用沿边投票选出最齐的一道（与最高票接近时取更外侧，躲开阴影和内框），
再在该处取局部明暗的 50% 作为亚像素位置。

edge_bias_x / edge_bias_y 是这条产品边上 50% 点相对外形轮廓的固定像素修正：
左右边 50% 落在饱和亮背景的外侧，上下边 50% 落在过渡带内侧。
正值表示向 ROI 内侧移动。
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from src.support.data_structure import Size_Result
from src.support.support_funs import ensure_gray_u8

MIN_ROI_DEPTH = 20
MIN_EDGE_VOTES = 8

QUALITY_GATES = {
    "min_transition_drop": 18.0,
    "peak_gradient_fraction": 0.28,
    "peak_gradient_floor": 4.0,
    "consensus_keep_ratio": 0.80,
    "guide_window": 4.0,
    "crossing_fraction": 0.50,
    "edge_bias_x": 0.60,
    "edge_bias_y": -1.40,
    "robust_sigma": 3.5,
    "max_outlier_distance": 2.0,
}


def resolve_gates(params):
    """合并固定门限与调用方传入的左右/上下修正。"""
    gates = dict(QUALITY_GATES)
    if not params:
        return gates
    for key in ("edge_bias_x", "edge_bias_y"):
        value = params.get(key)
        if value is None:
            continue
        gates[key] = float(value)
    return gates


def evaluate_edge_line(line, coordinate):
    return float(np.polyval([line["slope"], line["intercept"]], coordinate))


def _outside_first_profiles(plane, roi, roi_name):
    """单边剖面 (扫描线, 深度)，深度 0 在 ROI 外侧。"""
    roi_x, roi_y, roi_w, roi_h = roi
    sub = np.ascontiguousarray(plane[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w], dtype=np.float32)
    is_horizontal = roi_name in ("top", "bottom")
    if roi_name == "bottom":
        sub = sub[::-1, :]
    elif roi_name == "right":
        sub = sub[:, ::-1]
    profiles = np.ascontiguousarray(sub.T if is_horizontal else sub)
    lateral = cv.GaussianBlur(profiles, (1, 5), 0, borderType=cv.BORDER_REPLICATE)
    blurred = cv.GaussianBlur(
        lateral, (0, 0), sigmaX=0.8, sigmaY=0, borderType=cv.BORDER_REPLICATE,
    )
    return blurred, is_horizontal


def _expected_inward_sign(profiles):
    """外侧比内侧亮则向内变暗（-1），否则向内变亮（+1）。"""
    depth = profiles.shape[1]
    band = max(6, min(12, depth // 5))
    outer = float(np.percentile(profiles[:, :band], 75))
    inner = float(np.percentile(profiles[:, -band:], 35))
    return -1.0 if outer >= inner else 1.0


def _collect_peaks(profiles, sign, gates):
    strength = sign * np.diff(profiles, axis=1)
    n_lines, n_steps = strength.shape
    if n_steps < 6:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    limit = np.maximum(
        float(gates["peak_gradient_floor"]),
        float(gates["peak_gradient_fraction"]) * np.percentile(strength, 98, axis=1),
    )
    min_drop = float(gates["min_transition_drop"])
    cooldown = np.zeros(n_lines, dtype=np.int32)
    row_ids = []
    positions = []
    depth = profiles.shape[1]
    for index in range(2, n_steps - 2):
        ready = cooldown <= index
        is_peak = (
            ready
            & (strength[:, index] >= limit)
            & (strength[:, index] >= strength[:, index - 1])
            & (strength[:, index] >= strength[:, index + 1])
        )
        left = profiles[:, max(0, index - 3)]
        right = profiles[:, min(depth - 1, index + 4)]
        is_peak &= sign * (right - left) >= min_drop
        hit = np.flatnonzero(is_peak)
        if hit.size == 0:
            continue
        left_g = strength[hit, index - 1]
        mid = strength[hit, index]
        right_g = strength[hit, index + 1]
        denom = left_g - 2.0 * mid + right_g
        shift = np.divide(
            0.5 * (left_g - right_g), denom,
            out=np.zeros(hit.size, dtype=np.float64),
            where=np.abs(denom) > 1e-9,
        )
        row_ids.append(hit)
        positions.append(index + np.clip(shift, -0.6, 0.6))
        cooldown[hit] = index + 5
    if not row_ids:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return np.concatenate(row_ids), np.concatenate(positions)


def _consensus_depth(rows, positions, n_steps, n_lines, gates):
    if positions.size == 0:
        return None
    acc = np.bincount(
        np.clip(np.rint(positions).astype(np.int64), 0, n_steps - 1),
        minlength=n_steps,
    ).astype(np.float64)
    smooth = np.convolve(
        acc, np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0, mode="same",
    )
    peak = float(smooth.max())
    if peak < max(MIN_EDGE_VOTES, 0.08 * n_lines):
        return None
    qualified = np.flatnonzero(smooth >= float(gates["consensus_keep_ratio"]) * peak)
    best = int(qualified[0])
    window = smooth[max(0, best - 2): best + 3]
    return float(np.average(
        np.arange(max(0, best - 2), max(0, best - 2) + window.size),
        weights=np.maximum(window, 1e-6),
    ))


def _crossings(profiles, anchors, fraction):
    """锚点附近，局部外侧电平与内侧电平之间 fraction 处的亚像素深度。"""
    n_lines, depth = profiles.shape
    anchor = np.clip(np.rint(anchors).astype(np.int64), 8, max(8, depth - 12))
    outer_cols = anchor[:, None] + np.arange(-8, -1)
    inner_cols = np.clip(anchor[:, None] + np.arange(2, 12), 0, depth - 1)
    outer = np.median(np.take_along_axis(profiles, outer_cols, axis=1), axis=1)
    inner = np.median(np.take_along_axis(profiles, inner_cols, axis=1), axis=1)
    level = (1.0 - fraction) * inner + fraction * outer
    offsets = np.arange(-8, 9)
    columns = np.clip(anchor[:, None] + offsets[None, :], 0, depth - 1)
    window = np.take_along_axis(profiles, columns, axis=1)
    delta = window - level[:, None]
    crossed = delta[:, :-1] * delta[:, 1:] <= 0.0
    has = crossed.any(axis=1)
    index = np.argmax(crossed, axis=1)
    v0 = np.take_along_axis(window, index[:, None], axis=1)[:, 0]
    v1 = np.take_along_axis(
        window, np.clip(index + 1, 0, window.shape[1] - 1)[:, None], axis=1,
    )[:, 0]
    step = np.divide(level - v0, v1 - v0, out=np.zeros(n_lines), where=np.abs(v1 - v0) > 1e-9)
    depth_pos = anchor + (index - 8) + np.clip(step, 0.0, 1.0)
    return np.where(has, depth_pos, anchors), outer, inner, window


def _fit_depth_line(position, gates):
    index = np.flatnonzero(np.isfinite(position))
    if index.size < MIN_EDGE_VOTES:
        return None
    values = position[index]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = float(gates["robust_sigma"])
    tolerance = max(1.2, min(float(gates["max_outlier_distance"]) + 1.0, robust_sigma * 1.4826 * mad))
    if mad <= 1e-9:
        tolerance = 1.2
    keep = np.abs(values - median) <= tolerance
    if np.count_nonzero(keep) < MIN_EDGE_VOTES:
        keep = np.ones(index.size, dtype=bool)
    coeff = np.polyfit(index[keep].astype(np.float64), values[keep], 1)
    predicted = np.polyval(coeff, index.astype(np.float64))
    residual = values - predicted
    center = float(np.median(residual))
    mad2 = float(np.median(np.abs(residual - center)))
    cutoff = min(
        float(gates["max_outlier_distance"]),
        max(0.6, robust_sigma * 1.4826 * max(mad2, 0.05)),
    )
    refined = np.abs(residual - center) <= cutoff
    if np.count_nonzero(refined) >= MIN_EDGE_VOTES:
        keep = refined
        coeff = np.polyfit(index[keep].astype(np.float64), values[keep], 1)
        residual = values - np.polyval(coeff, index.astype(np.float64))
    chosen = index[keep]
    mask = np.zeros(position.shape[0], dtype=bool)
    mask[chosen] = True
    depth = float(np.polyval(coeff, float(np.median(chosen))))
    return {
        "depth": depth,
        "mask": mask,
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "line_rmse": float(np.sqrt(np.mean(residual[keep] ** 2))),
        "inlier_ratio": float(np.count_nonzero(keep) / max(1, index.size)),
        "inlier_count": int(np.count_nonzero(keep)),
    }


def _image_position(roi_name, roi, scanline, depth):
    roi_x, roi_y, roi_w, roi_h = roi
    if roi_name == "top":
        return roi_x + scanline, roi_y + depth
    if roi_name == "bottom":
        return roi_x + scanline, roi_y + roi_h - 1 - depth
    if roi_name == "left":
        return roi_x + depth, roi_y + scanline
    return roi_x + roi_w - 1 - depth, roi_y + scanline


def _measure_side(gray, roi, roi_name, gates):
    profiles, is_horizontal = _outside_first_profiles(gray, roi, roi_name)
    if profiles.shape[1] < MIN_ROI_DEPTH:
        return None, f"{roi_name}边ROI深度不足，至少需要 {MIN_ROI_DEPTH}px"
    sign = _expected_inward_sign(profiles)
    rows, peaks = _collect_peaks(profiles, sign, gates)
    guide = _consensus_depth(rows, peaks, profiles.shape[1] - 1, profiles.shape[0], gates)
    if guide is None:
        return None, f"{roi_name}边无可用外轮廓"
    position = np.full(profiles.shape[0], np.nan, dtype=np.float64)
    if peaks.size:
        grouped = {}
        for row, peak in zip(rows.tolist(), peaks.tolist()):
            grouped.setdefault(row, []).append(peak)
        window = float(gates["guide_window"])
        anchors = np.full(profiles.shape[0], np.nan)
        for row, candidates in grouped.items():
            near = [item for item in candidates if abs(item - guide) <= window]
            if near:
                anchors[row] = min(near, key=lambda item: abs(item - guide))
        valid_rows = np.flatnonzero(np.isfinite(anchors))
        if valid_rows.size:
            crossed, outer, inner, samples = _crossings(
                profiles[valid_rows], anchors[valid_rows], float(gates["crossing_fraction"]),
            )
            position[valid_rows] = crossed
        else:
            outer = inner = samples = None
            valid_rows = np.empty(0, dtype=np.int64)
    else:
        outer = inner = samples = None
        valid_rows = np.empty(0, dtype=np.int64)

    bias = float(gates["edge_bias_y"] if is_horizontal else gates["edge_bias_x"])
    fitted = _fit_depth_line(position, gates)
    if fitted is None:
        return None, f"{roi_name}边无可用外轮廓"
    depth = fitted["depth"] + bias
    scanline = float(np.median(np.flatnonzero(fitted["mask"])))
    origin_x, origin_y = _image_position(roi_name, roi, scanline, depth)
    inlier_rows = np.flatnonzero(fitted["mask"])
    inlier_depth = position[inlier_rows] + bias
    point_x, point_y = _image_position(
        roi_name, roi, inlier_rows.astype(np.float64), inlier_depth,
    )
    points = np.column_stack([point_x, point_y])
    if is_horizontal:
        coeff = np.polyfit(points[:, 0], points[:, 1], 1) if len(points) >= 2 else None
    else:
        coeff = np.polyfit(points[:, 1], points[:, 0], 1) if len(points) >= 2 else None
    line = {
        "slope": 0.0 if coeff is None else float(coeff[0]),
        "intercept": float(origin_y if is_horizontal else origin_x) if coeff is None else float(coeff[1]),
        "inlier_mask": np.ones(len(points), dtype=bool),
        "line_rmse": fitted["line_rmse"],
        "inlier_ratio": fitted["inlier_ratio"],
        "inlier_count": fitted["inlier_count"],
        "coverage": (
            float(points[:, 0].min()) if is_horizontal else float(points[:, 1].min()),
            float(points[:, 0].max()) if is_horizontal else float(points[:, 1].max()),
        ) if len(points) else (scanline, scanline),
    }
    if coeff is not None:
        line["slope"] = float(coeff[0])
        line["intercept"] = float(coeff[1])
    contrast = 0.0
    edge_width = 0.0
    representative = None
    if valid_rows.size:
        contrast = float(np.median(np.abs(outer - inner)))
        representative = {
            "window": samples[len(samples) // 2],
            "smoothed": samples[len(samples) // 2],
            "dark": float(min(outer[len(outer) // 2], inner[len(inner) // 2])),
            "bright": float(max(outer[len(outer) // 2], inner[len(inner) // 2])),
        }
    return {
        "roi": tuple(int(v) for v in roi),
        "is_horizontal": is_horizontal,
        "exterior_high": bool(sign < 0),
        "attempted": int(profiles.shape[0]),
        "valid": int(np.count_nonzero(np.isfinite(position))),
        "points": points,
        "plateau": 0,
        "binary_offset": bias,
        "contrast_ratio": contrast,
        "edge_width": edge_width,
        "representative": representative,
        "origin": (float(origin_x), float(origin_y)),
        **line,
    }, None


def detect_hybrid(detector, image):
    """在 SizeDetector 实例上执行外轮廓尺寸检测。"""
    from src.detectors.size_detector import SizeDetector

    try:
        gray = ensure_gray_u8(image, copy=True)
    except (TypeError, ValueError, cv.error) as exc:
        return detector._fail(str(exc))
    detector.image = gray
    img_h, img_w = gray.shape
    rois, roi_error = detector._validate_rois(img_w, img_h)
    if roi_error is not None:
        return detector._fail(roi_error)

    gates = resolve_gates(detector.params)
    direction = SizeDetector.normalize_detect_direction(
        detector.params.get("detect_direction", "outward")
    )
    debug = {}
    origins = {}
    for roi_name in detector.ROI_SIDES:
        measured, error = _measure_side(gray, rois[roi_name], roi_name, gates)
        if measured is None:
            return detector._fail(error, debug)
        debug[roi_name] = measured
        origins[roi_name] = measured["origin"]

    left_x, _ = origins["left"]
    right_x, _ = origins["right"]
    _, top_y = origins["top"]
    _, bottom_y = origins["bottom"]
    width_pixel = float(right_x - left_x)
    height_pixel = float(bottom_y - top_y)
    if width_pixel <= 0 or height_pixel <= 0:
        return detector._fail("尺寸边界顺序无效", debug)

    pixel_size_y = float(detector.params.get("pixel_size", 0.001))
    pixel_size_x = detector.params.get("pixel_size_x")
    pixel_size_x = pixel_size_y if pixel_size_x is None else float(pixel_size_x)
    width_mm = width_pixel * pixel_size_x
    height_mm = height_pixel * pixel_size_y

    std_width, std_height = detector.params.get("std_size", (0.0, 0.0))
    tolerance_x = float(detector.params.get("allow_tolerance_x", 0.0))
    tolerance_y = float(detector.params.get("allow_tolerance_y", 0.0))
    is_valid = True
    if float(std_width) > 0:
        is_valid = is_valid and abs(width_mm - float(std_width)) <= tolerance_x
    if float(std_height) > 0:
        is_valid = is_valid and abs(height_mm - float(std_height)) <= tolerance_y

    box_points = [float(left_x), float(top_y), width_pixel, height_pixel]
    min_threshold = int(detector.params.get("min_threshold", 0))
    max_threshold = int(detector.params.get("max_threshold", 255))
    debug["summary"] = {
        "x_reference": float(left_x + width_pixel / 2.0),
        "y_reference": float(top_y + height_pixel / 2.0),
        "raw_width_mm": float(width_mm),
        "raw_height_mm": float(height_mm),
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "box_points": box_points,
        "threshold": (min_threshold, max_threshold),
        "detect_direction": direction,
        "extrapolation_px": {side: 0.0 for side in detector.ROI_SIDES},
        "gates": gates,
    }
    detector.last_debug = debug
    detector.detection_result = Size_Result(
        width=float(width_mm), height=float(height_mm),
        box_points=box_points, is_valid=bool(is_valid),
    )
    return detector.detection_result

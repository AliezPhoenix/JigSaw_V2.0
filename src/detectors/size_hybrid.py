"""Dry 站尺寸检测：二值粗定位 + 灰度亮升沿 50% 精修。

质量门限固定为原 sensitivity=0 的宽松档，不再暴露灵敏度旋钮。
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from src.support.data_structure import Size_Result
from src.support.support_funs import ensure_gray_u8

BINARY_SMOOTH = 5
BINARY_HOLE_CLOSE = 7
BINARY_ISLAND_OPEN = 7
INNER_SPAN = 10
OUTER_SPAN = 20
MIN_ROI_DEPTH = 20
GUIDE_BAND = 6.0
ANCHOR_BRACKET = 6
SMOOTH_SIGMA = 1.0
LEVEL_SIGMA = 1.6
LATERAL_HALF_WIDTH = 3
POLARITY_BAND = 8
MIN_VALID_PROFILES = 12

# 原 sensitivity=0 档，写死后不再插值。
QUALITY_GATES = {
    "min_contrast_ratio": 0.12,
    "max_edge_width": 12.0,
    "max_binary_offset": 12.0,
    "min_profile_success_ratio": 0.20,
    "min_line_inlier_ratio": 0.20,
    "max_line_rmse": 1.00,
    "min_line_inlier_span_ratio": 0.50,
    "robust_sigma": 3.5,
    "max_outlier_distance": 2.5,
}


def profile_windows(depth):
    """按剖面深度给出粗定位边距与精修窗。深度不足 MIN_ROI_DEPTH 时返回 None。"""
    depth = int(depth)
    if depth < MIN_ROI_DEPTH:
        return None
    inner = min(INNER_SPAN, max(2, depth // 4))
    outer = min(OUTER_SPAN, max(3, depth - inner - 1))
    if inner + outer + 1 > depth:
        inner = max(2, depth - 4)
        outer = depth - inner - 1
    preferred_margin = max(INNER_SPAN, OUTER_SPAN) + 2
    min_band = 16
    max_margin = max(0, (depth - min_band) // 2)
    margin = min(preferred_margin, max_margin)
    return margin, inner, outer


def oriented_profiles(plane, roi, roi_name, smooth_sigma):
    """取出单边 ROI 剖面 (扫描线数, 深度)，索引 0 为产品内侧。"""
    from src.detectors.size_detector import SizeDetector

    roi_x, roi_y, roi_w, roi_h = roi
    sub = np.ascontiguousarray(
        plane[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w], dtype=np.float32
    )
    is_horizontal = roi_name in ("top", "bottom")
    if LATERAL_HALF_WIDTH > 0:
        kernel = 2 * LATERAL_HALF_WIDTH + 1
        sub = cv.blur(sub, (kernel, 1) if is_horizontal else (1, kernel))
    profiles = np.ascontiguousarray(sub.T) if is_horizontal else sub
    if SizeDetector._is_reverse_for_side(roi_name, "outward"):
        profiles = np.ascontiguousarray(profiles[:, ::-1])
    if smooth_sigma > 0:
        profiles = cv.GaussianBlur(
            profiles, (0, 0), sigmaX=float(smooth_sigma), sigmaY=0,
            borderType=cv.BORDER_REFLECT_101,
        )
    return profiles, is_horizontal


def binary_exterior_is_high(binary_profiles, band=POLARITY_BAND):
    """ROI 几何最外侧一带的二值中位数是否为 255（背景侧）。"""
    width = max(2, min(int(band), binary_profiles.shape[1] // 3))
    return bool(np.median(binary_profiles[:, -width:]) > 127.0)


def _first_run_end(mask):
    width = mask.shape[1]
    first = np.argmax(mask, axis=1)
    columns = np.arange(width, dtype=np.int64)[None, :]
    background = (~mask) & (columns >= first[:, None])
    first_background = np.where(
        background.any(axis=1), np.argmax(background, axis=1), width,
    )
    return first_background - 1


def binary_silhouette(binary_profiles, exterior_high, margin, direction="outward"):
    """按检测方向取产品轮廓：outward 第一道，inward 最外道。"""
    from src.detectors.size_detector import SizeDetector

    work = binary_profiles / 255.0
    if not exterior_high:
        work = 1.0 - work
    work = cv.blur(work, (BINARY_SMOOTH, 1))
    depth = work.shape[1]
    margin = max(0, min(int(margin), max(0, (depth - 3) // 2)))
    if depth - 2 * margin < 3:
        return None, None

    product = np.ascontiguousarray(((work < 0.5).astype(np.uint8)) * 255)
    depth_k = product.shape[1]
    close_k = min(BINARY_HOLE_CLOSE, depth_k if depth_k % 2 else depth_k - 1)
    open_k = min(BINARY_ISLAND_OPEN, depth_k if depth_k % 2 else depth_k - 1)
    if close_k >= 3:
        product = cv.morphologyEx(
            product, cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_RECT, (close_k, 1)),
        )
    if open_k >= 3:
        product = cv.morphologyEx(
            product, cv.MORPH_OPEN,
            cv.getStructuringElement(cv.MORPH_RECT, (open_k, 1)),
        )
    mask = product[:, margin:depth - margin] > 127
    has_edge = mask.any(axis=1)
    if SizeDetector.normalize_detect_direction(direction) == "inward":
        width = mask.shape[1]
        index = (width - 1) - np.argmax(mask[:, ::-1], axis=1)
    else:
        index = _first_run_end(mask)
    position = np.where(has_edge, index.astype(np.float64) + margin, np.nan)
    return position, has_edge


def consensus_line(position, trusted):
    if np.count_nonzero(trusted) < MIN_VALID_PROFILES:
        return None
    candidate = np.where(trusted, position, np.nan)
    median = float(np.nanmedian(candidate))
    mad = float(np.nanmedian(np.abs(candidate - median)))
    tolerance = max(3.0, 3.0 * 1.4826 * mad)
    rows = np.arange(len(position), dtype=np.float64)
    keep = trusted & (np.abs(position - median) <= tolerance)
    if np.count_nonzero(keep) < MIN_VALID_PROFILES:
        return np.full(len(position), median)
    guide = np.polyval(np.polyfit(rows[keep], position[keep], 1), rows)
    keep = trusted & (np.abs(position - guide) <= tolerance)
    if np.count_nonzero(keep) >= MIN_VALID_PROFILES:
        guide = np.polyval(np.polyfit(rows[keep], position[keep], 1), rows)
    return guide


def _plateau_offset(aggregate, inner, outer):
    dark = float(aggregate[:inner + 1].min())
    bright = float(np.median(aggregate[-max(3, outer // 3):]))
    span = bright - dark
    if not np.isfinite(span) or span <= 1e-9:
        return None
    tail = aggregate[inner:] >= dark + 0.98 * span
    reached = int(np.argmax(tail)) if tail.any() else outer
    hi = max(2, outer - 2)
    lo = min(2, hi)
    return int(np.clip(reached + 2, lo, hi))


def gradient_anchor(gray_profiles, guide, exterior_high, bracket=ANCHOR_BRACKET):
    signed = gray_profiles if exterior_high else -gray_profiles
    derivative = np.diff(signed, axis=1)
    depth = derivative.shape[1]
    width = 2 * int(bracket) + 1
    low = np.clip(
        np.rint(guide).astype(np.int64) - int(bracket), 0, max(0, depth - width)
    )
    columns = np.clip(low[:, None] + np.arange(width)[None, :], 0, depth - 1)
    return low + np.argmax(
        np.take_along_axis(derivative, columns, axis=1), axis=1
    )


def refine_rise50(gray_profiles, anchor, global_dynamic, inner_span=None, outer_span=None):
    depth = gray_profiles.shape[1]
    inner = INNER_SPAN if inner_span is None else int(inner_span)
    outer = OUTER_SPAN if outer_span is None else int(outer_span)
    lo = inner
    hi = max(inner, depth - outer - 1)
    base = np.clip(np.rint(anchor).astype(np.int64), lo, hi)
    offsets = np.arange(-inner, outer + 1)
    columns = np.clip(base[:, None] + offsets[None, :], 0, depth - 1)
    window = np.ascontiguousarray(
        np.take_along_axis(gray_profiles, columns, axis=1), dtype=np.float32
    )
    outer_level = float(np.median(window[:, -max(3, outer // 4):]))
    inner_level = float(np.median(window[:, :inner + 1]))
    if outer_level < inner_level:
        window = -window
    smoothed = cv.GaussianBlur(
        window, (0, 0), sigmaX=LEVEL_SIGMA, sigmaY=0,
        borderType=cv.BORDER_REFLECT_101,
    )
    plateau = _plateau_offset(
        np.median(smoothed, axis=0).astype(np.float64), inner, outer
    )
    if plateau is None:
        return None, "边缘对比度不足"
    window = window.astype(np.float64)
    smoothed = smoothed.astype(np.float64)
    bright = np.median(smoothed[:, inner + plateau:], axis=1)
    inner_band = smoothed[:, :inner + 1]
    trough_idx = np.argmin(inner_band, axis=1)
    center = np.clip(trough_idx, 1, max(1, inner - 1))
    left = np.take_along_axis(inner_band, (center - 1)[:, None], 1)[:, 0]
    middle = np.take_along_axis(inner_band, center[:, None], 1)[:, 0]
    right = np.take_along_axis(inner_band, (center + 1)[:, None], 1)[:, 0]
    curvature = left - 2.0 * middle + right
    shift = np.clip(
        np.divide(
            0.5 * (left - right), curvature,
            out=np.zeros_like(curvature), where=np.abs(curvature) > 1e-12,
        ),
        -1.0, 1.0,
    )
    dark = middle - 0.25 * (left - right) * shift
    contrast = bright - dark
    usable = np.isfinite(contrast) & (contrast > 1e-9)
    columns = np.arange(window.shape[1])[None, :]
    search_from = trough_idx[:, None]

    def crossing(level):
        reached = (window >= level[:, None]) & (columns >= search_from)
        any_reached = reached.any(axis=1)
        index = np.clip(np.argmax(reached, axis=1), 1, window.shape[1] - 1)
        upper = np.take_along_axis(window, index[:, None], 1)[:, 0]
        lower = np.take_along_axis(window, (index - 1)[:, None], 1)[:, 0]
        delta = upper - lower
        fraction = np.divide(
            level - lower, delta,
            out=np.zeros_like(delta), where=np.abs(delta) > 1e-12,
        )
        return np.where(
            any_reached, index - 1 + np.clip(fraction, 0.0, 1.0), np.nan
        )

    cross50 = crossing(dark + 0.50 * contrast)
    cross10 = crossing(dark + 0.10 * contrast)
    cross90 = crossing(dark + 0.90 * contrast)
    return {
        "position": base - inner + cross50,
        "contrast_ratio": contrast / max(float(global_dynamic), 1e-9),
        "edge_width": cross90 - cross10,
        "usable": usable,
        "window": window,
        "smoothed": smoothed,
        "dark": dark,
        "bright": bright,
        "plateau": plateau,
    }, None


def robust_fit_edge_line(points, is_horizontal, gates):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None
    independent = points[:, 0] if is_horizontal else points[:, 1]
    dependent = points[:, 1] if is_horizontal else points[:, 0]
    mask = np.ones(len(points), dtype=bool)
    robust_sigma = float(gates.get("robust_sigma", 3.5))
    max_outlier_distance = float(gates.get("max_outlier_distance", 2.5))
    coeff = None
    for _ in range(6):
        if np.count_nonzero(mask) < 2:
            return None
        coeff = np.polyfit(independent[mask], dependent[mask], 1)
        residual = dependent - np.polyval(coeff, independent)
        center = float(np.median(residual[mask]))
        mad = float(np.median(np.abs(residual[mask] - center)))
        cutoff = min(
            max_outlier_distance,
            max(0.10, robust_sigma * max(1.4826 * mad, 0.03)),
        )
        new_mask = np.abs(residual - center) <= cutoff
        if np.array_equal(mask, new_mask) or np.count_nonzero(new_mask) < 2:
            break
        mask = new_mask
    if coeff is None or np.count_nonzero(mask) < 2:
        return None
    coeff = np.polyfit(independent[mask], dependent[mask], 1)
    residual = dependent - np.polyval(coeff, independent)
    independent_span = max(float(np.ptp(independent)), 1e-9)
    return {
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "inlier_mask": mask,
        "line_rmse": float(np.sqrt(np.mean(residual[mask] ** 2))),
        "inlier_count": int(np.count_nonzero(mask)),
        "inlier_ratio": float(np.count_nonzero(mask) / len(mask)),
        "inlier_span_ratio": float(np.ptp(independent[mask]) / independent_span),
        "coverage": (float(independent[mask].min()), float(independent[mask].max())),
    }


def evaluate_edge_line(line, coordinate):
    return float(np.polyval([line["slope"], line["intercept"]], coordinate))


def side_debug(roi, is_horizontal, exterior_high, fine, offset,
               attempted, valid_count, line, points):
    usable = (
        fine["usable"]
        & np.isfinite(fine["position"])
        & np.isfinite(fine["edge_width"])
    )
    entry = {
        "roi": tuple(int(v) for v in roi),
        "is_horizontal": is_horizontal,
        "exterior_high": exterior_high,
        "attempted": attempted,
        "valid": valid_count,
        "points": np.asarray(points, dtype=np.float64).reshape(-1, 2),
        "plateau": int(fine["plateau"]),
        "binary_offset": float(offset),
    }
    if np.count_nonzero(usable) > 0:
        entry["contrast_ratio"] = float(np.median(fine["contrast_ratio"][usable]))
        entry["edge_width"] = float(np.median(fine["edge_width"][usable]))
        middle = int(np.flatnonzero(usable)[np.count_nonzero(usable) // 2])
    else:
        entry["contrast_ratio"] = 0.0
        entry["edge_width"] = 0.0
        middle = int(len(usable) // 2)
    entry["representative"] = {
        "window": fine["window"][middle],
        "smoothed": fine["smoothed"][middle],
        "dark": float(fine["dark"][middle]),
        "bright": float(fine["bright"][middle]),
    }
    if line is not None:
        entry.update(line)
    return entry


def detect_hybrid(detector, image):
    """在 SizeDetector 实例上执行混合尺寸检测。"""
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

    min_threshold = int(detector.params.get("min_threshold", 0))
    max_threshold = int(detector.params.get("max_threshold", 255))
    binary = cv.inRange(gray, min_threshold, max_threshold)
    if binary.min() == binary.max():
        return detector._fail(
            f"二值化图像无边界，请检查阈值 [{min_threshold}, {max_threshold}]"
        )

    gray_f = gray.astype(np.float32)
    q_low, q_high = np.percentile(gray_f[::3, ::3], [1.0, 99.0])
    global_dynamic = float(q_high - q_low)
    if global_dynamic <= 1e-9:
        return detector._fail("图像灰度动态范围不足")

    gates = QUALITY_GATES
    direction = SizeDetector.normalize_detect_direction(
        detector.params.get("detect_direction", "outward")
    )
    debug = {}
    lines = {}
    for roi_name in detector.ROI_SIDES:
        roi = rois[roi_name]
        gray_profiles, is_horizontal = oriented_profiles(
            gray_f, roi, roi_name, SMOOTH_SIGMA
        )
        binary_profiles, _ = oriented_profiles(binary, roi, roi_name, 0.0)
        windows = profile_windows(gray_profiles.shape[1])
        if windows is None:
            return detector._fail(
                f"{roi_name}边ROI深度不足，至少需要 {MIN_ROI_DEPTH}px", debug
            )
        margin, inner_span, outer_span = windows
        exterior_high = binary_exterior_is_high(binary_profiles)
        coarse, has_edge = binary_silhouette(
            binary_profiles, exterior_high, margin, direction
        )
        if coarse is None or has_edge is None:
            return detector._fail(f"{roi_name}边二值边界不足，请检查阈值", debug)
        guide = consensus_line(coarse, has_edge)
        if guide is None:
            return detector._fail(f"{roi_name}边二值边界不足，请检查阈值", debug)

        anchor = gradient_anchor(gray_profiles, guide, exterior_high)
        fine, error = refine_rise50(
            gray_profiles, anchor, global_dynamic, inner_span, outer_span
        )
        if fine is None:
            return detector._fail(f"{roi_name}边{error}", debug)

        deviation = fine["position"] - anchor
        measurable = fine["usable"] & np.isfinite(deviation)
        if np.count_nonzero(measurable) < MIN_VALID_PROFILES:
            return detector._fail(f"{roi_name}边无可用亚像素边缘", debug)
        offset = float(np.median(deviation[measurable]))
        binary_shift = abs(offset - float(np.median(anchor - guide)))
        if binary_shift > gates["max_binary_offset"]:
            debug[roi_name] = side_debug(
                roi, is_horizontal, exterior_high, fine, offset,
                int(len(deviation)), 0, None, np.empty((0, 2)),
            )
            return detector._fail(
                f"{roi_name}边二值与灰度判据不一致: {binary_shift:.2f}px",
                debug,
            )
        accepted = (
            measurable
            & np.isfinite(fine["edge_width"])
            & (np.abs(deviation - offset) <= GUIDE_BAND)
            & (fine["contrast_ratio"] >= gates["min_contrast_ratio"])
            & (fine["edge_width"] <= gates["max_edge_width"])
        )
        attempted = int(len(accepted))
        valid_count = int(np.count_nonzero(accepted))
        debug[roi_name] = side_debug(
            roi, is_horizontal, exterior_high, fine, offset,
            attempted, valid_count, None, np.empty((0, 2)),
        )
        if (
            valid_count < MIN_VALID_PROFILES
            or valid_count / max(1, attempted) < gates["min_profile_success_ratio"]
        ):
            return detector._fail(
                f"{roi_name}边有效扫描线不足: {valid_count}/{attempted}",
                debug,
            )

        depth = gray_profiles.shape[1]
        position = fine["position"][accepted]
        if SizeDetector._is_reverse_for_side(roi_name, "outward"):
            position = depth - 1 - position
        cross = np.flatnonzero(accepted)
        points = (
            np.column_stack([roi[0] + cross, roi[1] + position])
            if is_horizontal
            else np.column_stack([roi[0] + position, roi[1] + cross])
        )
        line = robust_fit_edge_line(points, is_horizontal, gates)
        debug[roi_name] = side_debug(
            roi, is_horizontal, exterior_high, fine, offset,
            attempted, valid_count, line, points,
        )
        if line is None:
            return detector._fail(f"{roi_name}边直线拟合失败", debug)
        if line["inlier_ratio"] < gates["min_line_inlier_ratio"]:
            return detector._fail(
                f"{roi_name}边内点比例不足: {line['inlier_ratio']:.3f}", debug
            )
        if line["line_rmse"] > gates["max_line_rmse"]:
            return detector._fail(
                f"{roi_name}边拟合残差过大: {line['line_rmse']:.3f}px", debug
            )
        if line["inlier_span_ratio"] < gates["min_line_inlier_span_ratio"]:
            return detector._fail(
                f"{roi_name}边内点跨度不足: {line['inlier_span_ratio']:.3f}",
                debug,
            )
        lines[roi_name] = line

    x_reference = float(np.median(np.concatenate(
        [debug["top"]["points"][:, 0], debug["bottom"]["points"][:, 0]]
    )))
    y_reference = float(np.median(np.concatenate(
        [debug["left"]["points"][:, 1], debug["right"]["points"][:, 1]]
    )))
    extrapolation = {}
    for roi_name in detector.ROI_SIDES:
        low, high = lines[roi_name]["coverage"]
        reference = (
            x_reference if debug[roi_name]["is_horizontal"] else y_reference
        )
        extrapolation[roi_name] = float(
            max(0.0, low - reference, reference - high)
        )

    top_boundary = evaluate_edge_line(lines["top"], x_reference)
    bottom_boundary = evaluate_edge_line(lines["bottom"], x_reference)
    left_boundary = evaluate_edge_line(lines["left"], y_reference)
    right_boundary = evaluate_edge_line(lines["right"], y_reference)
    width_pixel = right_boundary - left_boundary
    height_pixel = bottom_boundary - top_boundary
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

    box_points = [
        float(left_boundary), float(top_boundary),
        float(width_pixel), float(height_pixel),
    ]
    debug["summary"] = {
        "x_reference": x_reference,
        "y_reference": y_reference,
        "raw_width_mm": float(width_mm),
        "raw_height_mm": float(height_mm),
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "box_points": box_points,
        "threshold": (min_threshold, max_threshold),
        "detect_direction": direction,
        "extrapolation_px": extrapolation,
        "gates": gates,
    }
    detector.last_debug = debug
    detector.detection_result = Size_Result(
        width=float(width_mm), height=float(height_mm),
        box_points=box_points, is_valid=bool(is_valid),
    )
    return detector.detection_result

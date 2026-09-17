"""优化尺寸检测人工测试脚本。

本文件不修改生产 SizeDetector。优化管线：
  原始灰度/高位深图 → 可选暗场与平场校正 → 多扫描线局部灰度归一化
  → Logistic ESF 亚像素边缘拟合 → 鲁棒直线拟合 → 四边尺寸计算
  → 独立线性量具标定（不按标准尺寸逐件回拉）。

运行：
  python test/size_detector_test.py [图像或目录]

按键：
  Q 下一张，E 上一张，R 手工框选四边 ROI，D 恢复默认 ROI，
  S 保存调试图，ESC 退出。
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2 as cv
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.detectors.size_detector import SizeDetector  # noqa: E402
from src.support.data_structure import Size_Result  # noqa: E402


# ==================== 灰度与平场校正 ====================

def to_gray_float(image):
    """转为 float64 灰度图，保留 8/10/12/16 bit 的原始灰度级。"""
    if image is None:
        raise TypeError("输入图像为空")

    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            gray = image[:, :, 0]
        elif channels == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        raise ValueError(f"不支持的图像维度: {image.shape}")

    return np.ascontiguousarray(gray, dtype=np.float64)


def apply_flat_field(image_gray, dark_reference=None, flat_reference=None):
    """执行暗场/平场校正；无参考图时原样返回。"""
    work = np.asarray(image_gray, dtype=np.float64)

    if dark_reference is None and flat_reference is None:
        return work

    dark = np.zeros_like(work)
    if dark_reference is not None:
        dark = to_gray_float(dark_reference)
        if dark.shape != work.shape:
            raise ValueError(
                f"暗场图尺寸不一致: image={work.shape}, dark={dark.shape}"
            )

    if flat_reference is None:
        return work - dark

    flat = to_gray_float(flat_reference)
    if flat.shape != work.shape:
        raise ValueError(
            f"平场图尺寸不一致: image={work.shape}, flat={flat.shape}"
        )

    denominator = flat - dark
    positive = denominator[denominator > 0]
    if positive.size == 0:
        raise ValueError("平场图减暗场图后无有效亮度")

    median_response = float(np.median(positive))
    min_response = max(median_response * 0.05, 1e-9)
    valid = denominator >= min_response
    corrected = work - dark
    corrected[valid] = corrected[valid] * median_response / denominator[valid]
    return corrected


def gaussian_smooth_1d(values, sigma=1.0):
    """一维零相位高斯平滑。"""
    values = np.asarray(values, dtype=np.float64)
    sigma = float(sigma)
    if values.size < 3 or sigma <= 0:
        return values.copy()

    kernel_size = max(3, int(np.ceil(sigma * 6.0)) | 1)
    max_kernel = values.size if values.size % 2 == 1 else values.size - 1
    kernel_size = min(kernel_size, max_kernel)
    if kernel_size < 3:
        return values.copy()

    smoothed = cv.GaussianBlur(
        values.reshape(-1, 1),
        (1, kernel_size),
        sigmaX=0,
        sigmaY=sigma,
        borderType=cv.BORDER_REFLECT_101,
    )
    return smoothed.reshape(-1)


# ==================== 单扫描线 ESF 亚像素拟合 ====================

def _local_maxima(values, start_idx, end_idx):
    """返回指定区间内的局部极大值索引。"""
    out = []
    for idx in range(max(1, start_idx), min(len(values) - 1, end_idx)):
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            out.append(idx)
    return out


@lru_cache(maxsize=64)
def _logistic_esf_templates(
    fit_radius,
    max_transition_width,
    edge_polarity,
):
    """缓存固定窗口的 Logistic ESF 模板，避免每条扫描线重复生成。"""
    x_relative = np.arange(-fit_radius, fit_radius + 1, dtype=np.float64)
    center_offsets = np.tile(np.linspace(-1.25, 1.25, 31), 12)
    max_scale = max(0.12, float(max_transition_width) / 4.394449)
    min_scale = min(0.30, max_scale)
    scale_grid = np.repeat(np.linspace(min_scale, max_scale, 12), 31)
    z = np.clip(
        (x_relative[:, None] - center_offsets[None, :])
        / scale_grid[None, :],
        -30.0,
        30.0,
    )
    rising = 1.0 / (1.0 + np.exp(-z))
    edge_basis = rising if edge_polarity == "rising" else 1.0 - rising

    baseline = np.column_stack([np.ones_like(x_relative), x_relative])
    baseline_pinv = np.linalg.pinv(baseline)
    basis_residual = edge_basis - baseline @ (baseline_pinv @ edge_basis)
    denominator = np.sum(basis_residual * basis_residual, axis=0)
    transition_widths = 4.394449 * scale_grid
    return {
        "x_relative": x_relative,
        "center_offsets": center_offsets,
        "edge_basis": edge_basis,
        "baseline": baseline,
        "baseline_pinv": baseline_pinv,
        "basis_residual": basis_residual,
        "denominator": denominator,
        "transition_widths": transition_widths,
    }


def _fit_logistic_esf(
    profile_norm,
    coarse_idx,
    edge_polarity,
    fit_radius,
    max_transition_width,
):
    """在粗边缘附近网格搜索 Logistic ESF，返回最佳亚像素拟合。"""
    fit_radius = max(4, int(fit_radius))
    start = max(0, int(coarse_idx) - fit_radius)
    end = min(len(profile_norm), int(coarse_idx) + fit_radius + 1)
    if end - start < 9:
        return None

    y = np.asarray(profile_norm[start:end], dtype=np.float64)
    templates = _logistic_esf_templates(
        fit_radius,
        round(float(max_transition_width), 6),
        edge_polarity,
    )
    if len(y) != len(templates["x_relative"]):
        return None

    baseline = templates["baseline"]
    baseline_pinv = templates["baseline_pinv"]
    y_residual = y - baseline @ (baseline_pinv @ y)
    denominator = templates["denominator"]
    numerator = templates["basis_residual"].T @ y_residual
    amplitudes = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-12,
    )
    residual_sse = np.sum(y_residual * y_residual) - np.divide(
        numerator * numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    residual_sse = np.maximum(residual_sse, 0.0)
    rmse = np.sqrt(residual_sse / len(y))
    normalized_rmse = np.divide(
        rmse,
        amplitudes,
        out=np.full_like(rmse, np.inf),
        where=amplitudes > 0,
    )
    transition_widths = templates["transition_widths"]
    scores = normalized_rmse + transition_widths * 1e-4
    scores[~np.isfinite(scores)] = np.inf

    best_idx = int(np.argmin(scores))
    if not np.isfinite(scores[best_idx]):
        return None

    best_basis = templates["edge_basis"][:, best_idx]
    amplitude = float(amplitudes[best_idx])
    baseline_coeff = baseline_pinv @ (y - amplitude * best_basis)
    fitted = baseline @ baseline_coeff + amplitude * best_basis
    x = float(coarse_idx) + templates["x_relative"]
    return {
        "score": float(scores[best_idx]),
        "center": float(
            coarse_idx + templates["center_offsets"][best_idx]
        ),
        "amplitude": amplitude,
        "normalized_rmse": float(normalized_rmse[best_idx]),
        "transition_width": float(transition_widths[best_idx]),
        "fit_x": x,
        "fit_y": fitted,
    }


def fit_edge_profile(
    profile,
    need_reverse,
    edge_polarity,
    global_dynamic,
    params,
):
    """单条扫描线灰度边缘拟合，成功返回亚像素位置及质量信息。"""
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size < 12 or not np.all(np.isfinite(profile)):
        return None

    oriented = profile[::-1] if need_reverse else profile.copy()
    q_low, q_high = np.percentile(oriented, [5.0, 95.0])
    local_dynamic = float(q_high - q_low)
    if local_dynamic <= 1e-9:
        return None

    profile_norm = (oriented - q_low) / local_dynamic
    smoothed = gaussian_smooth_1d(
        profile_norm, params.get("smooth_sigma", 1.0)
    )
    gradient = np.gradient(smoothed)
    work_gradient = gradient if edge_polarity == "rising" else -gradient

    fit_radius = int(params.get("fit_radius", 8))
    margin = max(3, fit_radius)
    if len(work_gradient) <= margin * 2 + 1:
        return None

    valid_gradient = work_gradient[margin:len(work_gradient) - margin]
    max_strength = float(np.max(valid_gradient))
    if max_strength <= 1e-9:
        return None

    candidates = _local_maxima(
        work_gradient, margin, len(work_gradient) - margin
    )
    candidate_ratio = float(params.get("candidate_strength_ratio", 0.45))
    candidates = [
        idx for idx in candidates
        if work_gradient[idx] >= max_strength * candidate_ratio
    ]
    direction = SizeDetector.normalize_detect_direction(
        params.get("detect_direction", "outward")
    )
    candidates.sort(reverse=direction == "outward")
    max_candidates = max(1, int(params.get("max_candidates", 4)))

    min_contrast_ratio = float(params.get("min_contrast_ratio", 0.12))
    max_profile_rmse = float(params.get("max_profile_rmse", 0.12))
    max_transition_width = float(params.get("max_transition_width", 12.0))

    for coarse_idx in candidates[:max_candidates]:
        fit = _fit_logistic_esf(
            profile_norm,
            coarse_idx,
            edge_polarity,
            fit_radius,
            max_transition_width,
        )
        if fit is None:
            continue

        contrast_value = fit["amplitude"] * local_dynamic
        contrast_ratio = contrast_value / max(float(global_dynamic), 1e-9)
        if contrast_ratio < min_contrast_ratio:
            continue
        if fit["normalized_rmse"] > max_profile_rmse:
            continue
        if fit["transition_width"] > max_transition_width:
            continue

        oriented_position = float(fit["center"])
        original_position = (
            len(profile) - 1 - oriented_position
            if need_reverse else oriented_position
        )
        return {
            "position": float(original_position),
            "oriented_position": oriented_position,
            "contrast_ratio": float(contrast_ratio),
            "profile_rmse": float(fit["normalized_rmse"]),
            "transition_width": float(fit["transition_width"]),
            "profile": profile_norm,
            "smoothed": smoothed,
            "fit_x": fit["fit_x"],
            "fit_y": fit["fit_y"],
        }

    return None


# ==================== 多扫描线与鲁棒直线拟合 ====================

def normalize_edge_polarity(edge_polarity):
    """将边缘极性归一为 auto / rising / falling。"""
    if edge_polarity in ("rising", "falling"):
        return edge_polarity
    return "auto"


def estimate_edge_polarity(work, rois, direction, params):
    """根据四边二维一致性梯度，自动判断产品外轮廓的灰度极性。"""
    configured = normalize_edge_polarity(
        params.get("edge_polarity", "auto")
    )
    if configured != "auto":
        return configured, {}

    scores = {"rising": 0.0, "falling": 0.0}
    smooth_sigma = max(0.01, float(params.get("smooth_sigma", 1.0)))
    average_half_width = max(
        0.01,
        float(params.get("profile_average_half_width", 3)),
    )
    fit_radius = max(4, int(params.get("fit_radius", 8)))

    for roi_name, roi in rois.items():
        roi_x, roi_y, roi_w, roi_h = roi
        roi_image = work[
            roi_y:roi_y + roi_h,
            roi_x:roi_x + roi_w,
        ]
        is_horizontal = roi_name in ("top", "bottom")
        blurred = cv.GaussianBlur(
            roi_image,
            (0, 0),
            sigmaX=average_half_width if is_horizontal else smooth_sigma,
            sigmaY=smooth_sigma if is_horizontal else average_half_width,
            borderType=cv.BORDER_REFLECT_101,
        )
        depth_axis = 0 if is_horizontal else 1
        cross_axis = 1 if is_horizontal else 0
        gradient = np.gradient(blurred, axis=depth_axis)

        need_reverse = SizeDetector._is_reverse_for_side(
            roi_name,
            direction,
        )
        if need_reverse:
            gradient = -np.flip(gradient, axis=depth_axis)

        rising_score = np.percentile(
            np.maximum(gradient, 0.0),
            75.0,
            axis=cross_axis,
        )
        falling_score = np.percentile(
            np.maximum(-gradient, 0.0),
            75.0,
            axis=cross_axis,
        )
        if len(rising_score) <= fit_radius * 2 + 1:
            continue
        valid = slice(fit_radius, len(rising_score) - fit_radius)
        scores["rising"] += float(np.max(rising_score[valid]))
        scores["falling"] += float(np.max(falling_score[valid]))

    selected = max(scores, key=scores.get)
    return selected, scores


def collect_edge_points(
    roi_image,
    roi,
    roi_name,
    direction,
    edge_polarity,
    global_dynamic,
    params,
):
    """在单边 ROI 中采集多条扫描线并拟合亚像素边缘点。"""
    roi_x, roi_y, roi_w, roi_h = roi
    is_horizontal = roi_name in ("top", "bottom")
    need_reverse = SizeDetector._is_reverse_for_side(roi_name, direction)

    scan_step = max(1, int(params.get("scan_step", 4)))
    average_half_width = max(0, int(params.get("profile_average_half_width", 1)))
    cross_length = roi_w if is_horizontal else roi_h
    start = average_half_width
    stop = cross_length - average_half_width

    points = []
    representative = None
    center_cross = cross_length / 2.0
    attempted = 0

    for cross_idx in range(start, stop, scan_step):
        attempted += 1
        if is_horizontal:
            x0 = max(0, cross_idx - average_half_width)
            x1 = min(roi_w, cross_idx + average_half_width + 1)
            profile = np.mean(roi_image[:, x0:x1], axis=1)
        else:
            y0 = max(0, cross_idx - average_half_width)
            y1 = min(roi_h, cross_idx + average_half_width + 1)
            profile = np.mean(roi_image[y0:y1, :], axis=0)

        fit = fit_edge_profile(
            profile,
            need_reverse,
            edge_polarity,
            global_dynamic,
            params,
        )
        if fit is None:
            continue

        if is_horizontal:
            point = (float(roi_x + cross_idx), float(roi_y + fit["position"]))
        else:
            point = (float(roi_x + fit["position"]), float(roi_y + cross_idx))
        points.append(point)

        if (
            representative is None
            or abs(cross_idx - center_cross)
            < abs(representative["cross_idx"] - center_cross)
        ):
            representative = dict(fit)
            representative["cross_idx"] = int(cross_idx)

    return {
        "roi": tuple(int(v) for v in roi),
        "is_horizontal": is_horizontal,
        "need_reverse": need_reverse,
        "edge_polarity": edge_polarity,
        "attempted": int(attempted),
        "points": np.asarray(points, dtype=np.float64).reshape(-1, 2),
        "representative": representative,
    }


def robust_fit_edge_line(points, is_horizontal, params):
    """迭代 MAD 剔除离群点，拟合 y=ax+b 或 x=ay+b。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None

    independent = points[:, 0] if is_horizontal else points[:, 1]
    dependent = points[:, 1] if is_horizontal else points[:, 0]
    mask = np.ones(len(points), dtype=bool)
    robust_sigma = float(params.get("robust_sigma", 2.8))
    max_outlier_distance = float(params.get("max_outlier_distance", 2.0))

    coeff = None
    for _ in range(6):
        if np.count_nonzero(mask) < 2:
            return None
        coeff = np.polyfit(independent[mask], dependent[mask], 1)
        residual = dependent - np.polyval(coeff, independent)
        center = float(np.median(residual[mask]))
        mad = float(np.median(np.abs(residual[mask] - center)))
        robust_scale = max(1.4826 * mad, 0.03)
        cutoff = min(
            max_outlier_distance,
            max(0.10, robust_sigma * robust_scale),
        )
        new_mask = np.abs(residual - center) <= cutoff
        if np.array_equal(mask, new_mask):
            break
        if np.count_nonzero(new_mask) < 2:
            break
        mask = new_mask

    if coeff is None or np.count_nonzero(mask) < 2:
        return None

    coeff = np.polyfit(independent[mask], dependent[mask], 1)
    residual = dependent - np.polyval(coeff, independent)
    line_rmse = float(np.sqrt(np.mean(residual[mask] ** 2)))
    independent_span = max(float(np.ptp(independent)), 1e-9)
    inlier_span_ratio = float(np.ptp(independent[mask]) / independent_span)
    return {
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "inlier_mask": mask,
        "line_rmse": line_rmse,
        "inlier_count": int(np.count_nonzero(mask)),
        "inlier_ratio": float(np.count_nonzero(mask) / len(mask)),
        "inlier_span_ratio": inlier_span_ratio,
    }


def evaluate_edge_line(line, coordinate):
    """计算 y=ax+b 或 x=ay+b。"""
    return float(np.polyval([line["slope"], line["intercept"]], coordinate))


# ==================== 优化尺寸检测器 ====================

class OptimizedSizeDetector(SizeDetector):
    """人工测试用灰度亚像素尺寸检测器，不逐件向标准值回拉。"""

    def __init__(self, params=None):
        self.detection_result = None
        self.last_debug = {}
        self.dark_reference = None
        self.flat_reference = None
        self.params = {
            "allow_tolerance_x": 0.0,
            "allow_tolerance_y": 0.0,
            "rois": {side: None for side in self.ROI_SIDES},
            "std_size": (0.0, 0.0),
            "pixel_size": 0.014,
            "pixel_size_x": None,
            "detect_direction": "outward",
            "edge_polarity": "auto",
            "scan_step": 4,
            "profile_average_half_width": 3,
            "smooth_sigma": 1.0,
            "candidate_strength_ratio": 0.45,
            "max_candidates": 4,
            "fit_radius": 8,
            "min_contrast_ratio": 0.12,
            "max_profile_rmse": 0.12,
            "max_transition_width": 12.0,
            "min_valid_profiles": 12,
            "min_profile_success_ratio": 0.20,
            "min_line_inlier_ratio": 0.50,
            "max_line_rmse": 0.50,
            "min_line_inlier_span_ratio": 0.65,
            "max_sparse_line_rmse": 1.25,
            "robust_sigma": 2.8,
            "max_outlier_distance": 2.0,
            "calibration_x": (1.0, 0.0),
            "calibration_y": (1.0, 0.0),
        }
        if params:
            self.update_params(params)

    def update_params(self, params, clear_result=True):
        """更新人工测试参数。"""
        if not params:
            return True

        for key, value in params.items():
            if key == "rois":
                self.params[key] = SizeDetector._normalize_rois(value)
            elif key == "detect_direction":
                self.params[key] = SizeDetector.normalize_detect_direction(value)
            elif key == "edge_polarity":
                self.params[key] = normalize_edge_polarity(value)
            elif key in self.params:
                self.params[key] = value

        if clear_result:
            self.detection_result = None
            self.last_debug = {}
        return True

    def set_reference_images(self, dark_reference=None, flat_reference=None):
        """设置可选暗场与平场参考图。"""
        self.dark_reference = dark_reference
        self.flat_reference = flat_reference

    def _fail(self, message, debug=None):
        self.last_debug = debug or {}
        self.detection_result = Size_Result(
            error_code=2,
            error_msg=message,
            is_valid=False,
        )
        return self.detection_result

    @staticmethod
    def _calibrate(raw_mm, calibration):
        if not isinstance(calibration, (tuple, list)) or len(calibration) != 2:
            raise ValueError(f"量具标定参数无效: {calibration}")
        scale, offset = float(calibration[0]), float(calibration[1])
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
            raise ValueError(f"量具标定参数无效: {calibration}")
        return scale * float(raw_mm) + offset

    def prepare_image(self, image):
        """预计算固定图像的平场结果和动态范围，供人工调参重复使用。"""
        image_gray = to_gray_float(image)
        work = apply_flat_field(
            image_gray,
            self.dark_reference,
            self.flat_reference,
        )
        q_low, q_high = np.percentile(work, [1.0, 99.0])
        global_dynamic = float(q_high - q_low)
        if global_dynamic <= 1e-9:
            raise ValueError("图像灰度动态范围不足")
        return {
            "work": work,
            "global_dynamic": global_dynamic,
        }

    def detect(self, image):
        """执行优化后的四边灰度亚像素尺寸检测。"""
        try:
            prepared = self.prepare_image(image)
        except (TypeError, ValueError, cv.error) as exc:
            return self._fail(str(exc))
        return self.detect_prepared(prepared)

    def detect_prepared(self, prepared):
        """对 prepare_image 的结果执行检测，避免调参时重复预处理。"""
        work = prepared["work"]
        global_dynamic = float(prepared["global_dynamic"])
        img_h, img_w = work.shape
        rois, roi_error = self._validate_rois(img_w, img_h)
        if roi_error is not None:
            return self._fail(roi_error)

        direction = SizeDetector.normalize_detect_direction(
            self.params.get("detect_direction", "outward")
        )
        edge_polarity, polarity_scores = estimate_edge_polarity(
            work,
            rois,
            direction,
            self.params,
        )
        debug = {}
        lines = {}

        for roi_name in self.ROI_SIDES:
            roi = rois[roi_name]
            roi_x, roi_y, roi_w, roi_h = roi
            roi_image = work[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            edge_debug = collect_edge_points(
                roi_image,
                roi,
                roi_name,
                direction,
                edge_polarity,
                global_dynamic,
                self.params,
            )
            debug[roi_name] = edge_debug

            attempted = max(1, int(edge_debug["attempted"]))
            valid_count = len(edge_debug["points"])
            min_valid = min(
                int(self.params.get("min_valid_profiles", 12)),
                max(4, int(np.ceil(attempted * 0.5))),
            )
            success_ratio = valid_count / attempted
            if (
                valid_count < min_valid
                or success_ratio
                < float(self.params.get("min_profile_success_ratio", 0.20))
            ):
                return self._fail(
                    f"{roi_name}边有效扫描线不足: "
                    f"{valid_count}/{attempted}",
                    debug,
                )

            line = robust_fit_edge_line(
                edge_debug["points"],
                edge_debug["is_horizontal"],
                self.params,
            )
            if line is None:
                return self._fail(f"{roi_name}边直线拟合失败", debug)

            edge_debug.update(line)
            min_inlier_ratio = float(
                self.params.get("min_line_inlier_ratio", 0.50)
            )
            max_line_rmse = float(self.params.get("max_line_rmse", 0.50))
            min_inlier_span_ratio = float(
                self.params.get("min_line_inlier_span_ratio", 0.65)
            )
            max_sparse_line_rmse = float(
                self.params.get("max_sparse_line_rmse", 1.25)
            )
            sparse_line_valid = (
                line["inlier_count"] >= min_valid
                and line["inlier_span_ratio"] >= min_inlier_span_ratio
                and line["line_rmse"] <= max_sparse_line_rmse
            )
            if (
                line["inlier_ratio"] < min_inlier_ratio
                and not sparse_line_valid
            ):
                return self._fail(
                    f"{roi_name}边内点比例不足: "
                    f"{line['inlier_ratio']:.3f}",
                    debug,
                )
            if line["line_rmse"] > max_line_rmse and not sparse_line_valid:
                return self._fail(
                    f"{roi_name}边拟合残差过大: "
                    f"{line['line_rmse']:.3f}px",
                    debug,
                )
            lines[roi_name] = line

        horizontal_x = np.concatenate(
            [debug["top"]["points"][:, 0], debug["bottom"]["points"][:, 0]]
        )
        vertical_y = np.concatenate(
            [debug["left"]["points"][:, 1], debug["right"]["points"][:, 1]]
        )
        x_reference = float(np.median(horizontal_x))
        y_reference = float(np.median(vertical_y))

        top_boundary = evaluate_edge_line(lines["top"], x_reference)
        bottom_boundary = evaluate_edge_line(lines["bottom"], x_reference)
        left_boundary = evaluate_edge_line(lines["left"], y_reference)
        right_boundary = evaluate_edge_line(lines["right"], y_reference)

        width_pixel = right_boundary - left_boundary
        height_pixel = bottom_boundary - top_boundary
        if width_pixel <= 0 or height_pixel <= 0:
            return self._fail("尺寸边界顺序无效", debug)

        pixel_size_y = float(self.params.get("pixel_size", 0.014))
        pixel_size_x = self.params.get("pixel_size_x")
        pixel_size_x = (
            pixel_size_y if pixel_size_x is None else float(pixel_size_x)
        )

        raw_width_mm = width_pixel * pixel_size_x
        raw_height_mm = height_pixel * pixel_size_y
        try:
            width_mm = self._calibrate(
                raw_width_mm,
                self.params.get("calibration_x", (1.0, 0.0)),
            )
            height_mm = self._calibrate(
                raw_height_mm,
                self.params.get("calibration_y", (1.0, 0.0)),
            )
        except ValueError as exc:
            return self._fail(str(exc), debug)
        if (
            not np.isfinite(width_mm)
            or not np.isfinite(height_mm)
            or width_mm <= 0
            or height_mm <= 0
        ):
            return self._fail("量具标定后的尺寸必须是有限正数", debug)

        std_width, std_height = self.params.get("std_size", (0.0, 0.0))
        tolerance_x = float(self.params.get("allow_tolerance_x", 0.0))
        tolerance_y = float(self.params.get("allow_tolerance_y", 0.0))
        is_valid = True
        if float(std_width) > 0:
            is_valid = is_valid and abs(width_mm - float(std_width)) <= tolerance_x
        if float(std_height) > 0:
            is_valid = is_valid and abs(height_mm - float(std_height)) <= tolerance_y

        box_points = [
            float(left_boundary),
            float(top_boundary),
            float(width_pixel),
            float(height_pixel),
        ]
        debug["summary"] = {
            "x_reference": x_reference,
            "y_reference": y_reference,
            "raw_width_mm": float(raw_width_mm),
            "raw_height_mm": float(raw_height_mm),
            "width_mm": float(width_mm),
            "height_mm": float(height_mm),
            "box_points": box_points,
            "edge_polarity": edge_polarity,
            "polarity_scores": polarity_scores,
        }

        self.last_debug = debug
        self.detection_result = Size_Result(
            width=float(width_mm),
            height=float(height_mm),
            box_points=box_points,
            is_valid=bool(is_valid),
        )
        return self.detection_result


# ==================== 人工测试可视化 ====================

CONTROL_WINDOW = "optimized_controls"
RESULT_WINDOW = "optimized_result"
QUALITY_WINDOW = "edge_quality"
_WINDOWS_READY = False


def _noop(_):
    return


def normalize_for_display(image):
    """把任意位深图像归一化为 BGR uint8，仅用于显示。"""
    gray = to_gray_float(image)
    q_low, q_high = np.percentile(gray, [1.0, 99.0])
    if q_high - q_low <= 1e-9:
        display = np.zeros(gray.shape, dtype=np.uint8)
    else:
        display = np.clip(
            (gray - q_low) * 255.0 / (q_high - q_low),
            0,
            255,
        ).astype(np.uint8)
    return cv.cvtColor(display, cv.COLOR_GRAY2BGR)


def _put_hud_text(image, text, origin, color, scale=0.65, thickness=1):
    x, y = origin
    cv.putText(
        image,
        text,
        (x + 1, y + 1),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv.LINE_AA,
    )


def draw_detection_debug(image, detector, image_name="", base_display=None):
    """绘制 ROI、扫描线边缘点、内点直线及最终尺寸框。"""
    canvas = (
        normalize_for_display(image)
        if base_display is None else base_display.copy()
    )
    debug = detector.last_debug or {}

    for roi_name in detector.ROI_SIDES:
        info = debug.get(roi_name)
        if not info:
            continue

        roi_x, roi_y, roi_w, roi_h = info["roi"]
        cv.rectangle(
            canvas,
            (roi_x, roi_y),
            (roi_x + roi_w - 1, roi_y + roi_h - 1),
            (0, 165, 255),
            1,
        )

        points = np.asarray(info.get("points", []), dtype=np.float64)
        inlier_mask = info.get("inlier_mask")
        if len(points) > 0:
            if inlier_mask is None:
                inlier_mask = np.zeros(len(points), dtype=bool)
            for idx, point in enumerate(points):
                color = (0, 220, 0) if inlier_mask[idx] else (0, 0, 255)
                cv.circle(
                    canvas,
                    (int(round(point[0])), int(round(point[1]))),
                    1,
                    color,
                    -1,
                    cv.LINE_AA,
                )

        if "slope" in info and "intercept" in info:
            if info["is_horizontal"]:
                x0, x1 = roi_x, roi_x + roi_w - 1
                y0 = evaluate_edge_line(info, x0)
                y1 = evaluate_edge_line(info, x1)
                p0 = (int(x0), int(round(y0)))
                p1 = (int(x1), int(round(y1)))
            else:
                y0, y1 = roi_y, roi_y + roi_h - 1
                x0 = evaluate_edge_line(info, y0)
                x1 = evaluate_edge_line(info, y1)
                p0 = (int(round(x0)), int(y0))
                p1 = (int(round(x1)), int(y1))
            cv.line(canvas, p0, p1, (0, 255, 255), 1, cv.LINE_AA)

    result = detector.detection_result
    if result is not None and result.error_code == 0:
        x, y, width, height = result.box_points
        color = (0, 255, 0) if result.is_valid else (0, 0, 255)
        cv.rectangle(
            canvas,
            (int(round(x)), int(round(y))),
            (int(round(x + width)), int(round(y + height))),
            color,
            2,
            cv.LINE_AA,
        )
        status = "OK" if result.is_valid else "NG"
        lines = [
            f"{image_name}",
            f"W={result.width:.6f} mm  H={result.height:.6f} mm",
            f"STATUS={status}",
        ]
    else:
        error_msg = "" if result is None else str(result.error_msg or "")
        lines = [f"{image_name}", "DETECT FAILED", error_msg[:80]]

    lines.append("Q next  E previous  R select ROI  D default ROI  S save  ESC quit")
    for idx, line in enumerate(lines):
        color = (255, 255, 255)
        if "FAILED" in line or "NG" in line:
            color = (0, 0, 255)
        elif "STATUS=OK" in line:
            color = (0, 255, 0)
        _put_hud_text(canvas, line, (15, 28 + idx * 26), color)
    return canvas


def _draw_curve(canvas, values, rect, color, value_range=None):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return
    x, y, width, height = rect
    if value_range is None:
        v_min, v_max = float(np.min(values)), float(np.max(values))
    else:
        v_min, v_max = value_range
    if v_max - v_min <= 1e-9:
        return
    xs = np.linspace(x, x + width - 1, len(values))
    ys = y + height - 1 - (values - v_min) * (height - 1) / (v_max - v_min)
    points = np.column_stack([xs, ys]).astype(np.int32)
    cv.polylines(canvas, [points], False, color, 1, cv.LINE_AA)


def render_quality_panel(debug, width=1000, height=680):
    """绘制四边代表扫描线、ESF 拟合和质量参数。"""
    canvas = np.full((height, width, 3), 25, dtype=np.uint8)
    margin = 12
    cell_width = (width - margin * 3) // 2
    cell_height = (height - margin * 3) // 2

    for idx, roi_name in enumerate(SizeDetector.ROI_SIDES):
        row, col = divmod(idx, 2)
        x0 = margin + col * (cell_width + margin)
        y0 = margin + row * (cell_height + margin)
        cv.rectangle(
            canvas,
            (x0, y0),
            (x0 + cell_width - 1, y0 + cell_height - 1),
            (70, 70, 70),
            1,
        )

        info = debug.get(roi_name) or {}
        representative = info.get("representative")
        count = len(info.get("points", []))
        attempted = int(info.get("attempted", 0))
        rmse = info.get("line_rmse")
        label = f"{roi_name}: profiles={count}/{attempted}"
        if rmse is not None:
            label += f" line_rmse={rmse:.3f}px"
        _put_hud_text(canvas, label, (x0 + 8, y0 + 20), (240, 240, 240), 0.48)

        if not representative:
            continue

        plot = (x0 + 8, y0 + 35, cell_width - 16, cell_height - 48)
        _draw_curve(
            canvas,
            representative["profile"],
            plot,
            (130, 130, 130),
            (-0.2, 1.2),
        )
        _draw_curve(
            canvas,
            representative["smoothed"],
            plot,
            (0, 220, 255),
            (-0.2, 1.2),
        )

        fit_x = representative["fit_x"]
        fit_y = representative["fit_y"]
        fit_canvas = np.full(len(representative["profile"]), np.nan)
        fit_indices = np.clip(
            np.round(fit_x).astype(int),
            0,
            len(fit_canvas) - 1,
        )
        fit_canvas[fit_indices] = fit_y
        valid = np.isfinite(fit_canvas)
        if np.count_nonzero(valid) >= 2:
            x, y, plot_w, plot_h = plot
            xs = x + np.flatnonzero(valid) * (plot_w - 1) / max(
                1, len(fit_canvas) - 1
            )
            ys = (
                y + plot_h - 1
                - (fit_canvas[valid] + 0.2) * (plot_h - 1) / 1.4
            )
            points = np.column_stack([xs, ys]).astype(np.int32)
            cv.polylines(canvas, [points], False, (0, 255, 0), 1, cv.LINE_AA)

        edge_x = int(
            plot[0]
            + representative["oriented_position"]
            * (plot[2] - 1)
            / max(1, len(representative["profile"]) - 1)
        )
        cv.line(
            canvas,
            (edge_x, plot[1]),
            (edge_x, plot[1] + plot[3] - 1),
            (255, 180, 0),
            1,
            cv.LINE_AA,
        )

        quality = (
            f"contrast={representative['contrast_ratio']:.3f}  "
            f"profile_rmse={representative['profile_rmse']:.3f}  "
            f"width={representative['transition_width']:.2f}px"
        )
        _put_hud_text(
            canvas,
            quality,
            (x0 + 8, y0 + cell_height - 8),
            (190, 190, 190),
            0.42,
        )

    _put_hud_text(
        canvas,
        "gray=normalized profile  yellow=smoothed  green=ESF fit  blue=edge",
        (margin, height - 4),
        (190, 190, 190),
        0.42,
    )
    return canvas


def init_windows(args):
    """初始化人工调参窗口。"""
    global _WINDOWS_READY
    if _WINDOWS_READY:
        return

    cv.namedWindow(CONTROL_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(CONTROL_WINDOW, 760, 350)
    cv.namedWindow(RESULT_WINDOW, cv.WINDOW_NORMAL)
    cv.namedWindow(QUALITY_WINDOW, cv.WINDOW_NORMAL)

    cv.createTrackbar("roi_strip", CONTROL_WINDOW, int(args.roi_strip), 2000, _noop)
    cv.createTrackbar("scan_step", CONTROL_WINDOW, int(args.scan_step), 20, _noop)
    cv.createTrackbar(
        "avg_half",
        CONTROL_WINDOW,
        int(args.profile_average_half_width),
        8,
        _noop,
    )
    cv.createTrackbar(
        "smooth_x10",
        CONTROL_WINDOW,
        int(round(args.smooth_sigma * 10)),
        50,
        _noop,
    )
    cv.createTrackbar(
        "candidate_pct",
        CONTROL_WINDOW,
        int(round(args.candidate_strength_ratio * 100)),
        100,
        _noop,
    )
    cv.createTrackbar(
        "contrast_x1000",
        CONTROL_WINDOW,
        int(round(args.min_contrast_ratio * 1000)),
        1000,
        _noop,
    )
    cv.createTrackbar(
        "profile_rmse_x1000",
        CONTROL_WINDOW,
        int(round(args.max_profile_rmse * 1000)),
        1000,
        _noop,
    )
    cv.createTrackbar(
        "line_rmse_x1000",
        CONTROL_WINDOW,
        int(round(args.max_line_rmse * 1000)),
        2000,
        _noop,
    )
    cv.createTrackbar(
        "transition_x10",
        CONTROL_WINDOW,
        int(round(args.max_transition_width * 10)),
        500,
        _noop,
    )
    cv.createTrackbar(
        "inward",
        CONTROL_WINDOW,
        1 if args.direction == "inward" else 0,
        1,
        _noop,
    )
    polarity_value = {
        "auto": 0,
        "falling": 1,
        "rising": 2,
    }[args.edge_polarity]
    cv.createTrackbar(
        "polarity",
        CONTROL_WINDOW,
        polarity_value,
        2,
        _noop,
    )
    _WINDOWS_READY = True


def read_control_params():
    """读取轨迹条参数，返回 ROI 宽度与检测参数。"""
    roi_strip = max(12, cv.getTrackbarPos("roi_strip", CONTROL_WINDOW))
    params = {
        "scan_step": max(1, cv.getTrackbarPos("scan_step", CONTROL_WINDOW)),
        "profile_average_half_width": cv.getTrackbarPos("avg_half", CONTROL_WINDOW),
        "smooth_sigma": cv.getTrackbarPos("smooth_x10", CONTROL_WINDOW) / 10.0,
        "candidate_strength_ratio": max(
            0.01,
            cv.getTrackbarPos("candidate_pct", CONTROL_WINDOW) / 100.0,
        ),
        "min_contrast_ratio": cv.getTrackbarPos(
            "contrast_x1000", CONTROL_WINDOW
        ) / 1000.0,
        "max_profile_rmse": max(
            0.001,
            cv.getTrackbarPos("profile_rmse_x1000", CONTROL_WINDOW) / 1000.0,
        ),
        "max_line_rmse": max(
            0.001,
            cv.getTrackbarPos("line_rmse_x1000", CONTROL_WINDOW) / 1000.0,
        ),
        "max_transition_width": max(
            1.0,
            cv.getTrackbarPos("transition_x10", CONTROL_WINDOW) / 10.0,
        ),
        "detect_direction": (
            "inward"
            if cv.getTrackbarPos("inward", CONTROL_WINDOW)
            else "outward"
        ),
        "edge_polarity": {
            0: "auto",
            1: "falling",
            2: "rising",
        }[cv.getTrackbarPos("polarity", CONTROL_WINDOW)],
    }
    signature = (roi_strip,) + tuple(params.values())
    return roi_strip, params, signature


def render_control_help(width=760, height=300):
    """显示参数含义与操作提示。"""
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    lines = [
        "Optimized grayscale subpixel detector",
        "roi_strip: default four-side ROI depth",
        "scan_step / avg_half: profile spacing and lateral averaging",
        "smooth_x10: Gaussian sigma",
        "candidate_pct: candidate edge / strongest gradient threshold",
        "contrast_x1000: minimum fitted contrast / image dynamic range",
        "profile_rmse_x1000: maximum normalized ESF residual",
        "line_rmse_x1000: maximum inlier line residual (pixel)",
        "transition_x10: maximum 10%-90% edge width (pixel)",
        "inward: 0=inside-to-outside, 1=outside-to-inside",
        "polarity: 0=auto, 1=falling, 2=rising",
    ]
    for idx, line in enumerate(lines):
        _put_hud_text(panel, line, (12, 22 + idx * 24), (220, 220, 220), 0.48)
    return panel


def select_four_rois(image):
    """依次人工框选 top/left/bottom/right ROI；取消任一边则整体取消。"""
    display = normalize_for_display(image)
    rois = {}
    for side in SizeDetector.ROI_SIDES:
        window_name = f"select_{side}_roi"
        print(f"请框选 {side} 边 ROI，Enter/Space 确认，ESC 取消")
        roi = cv.selectROI(
            window_name,
            display,
            showCrosshair=True,
            fromCenter=False,
        )
        cv.destroyWindow(window_name)
        if roi[2] <= 0 or roi[3] <= 0:
            return None
        rois[side] = tuple(int(v) for v in roi)
    return rois


def save_debug_images(save_dir, image_path, result_image, quality_image):
    """保存人工检查用结果图与边缘质量图。"""
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        result_path = save_dir / f"{stem}_optimized_result.png"
        quality_path = save_dir / f"{stem}_edge_quality.png"
        result_saved = cv.imwrite(str(result_path), result_image)
        quality_saved = cv.imwrite(str(quality_path), quality_image)
    except (OSError, cv.error) as exc:
        print(f"保存调试图失败: {exc}")
        return False

    if not result_saved or not quality_saved:
        print(f"保存调试图失败: {save_dir}")
        return False

    print(f"已保存: {result_path}")
    print(f"已保存: {quality_path}")
    return True


def run_manual_ui(detector, image_path, args):
    """运行单张图的人工调试界面。"""
    image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return "next"

    init_windows(args)
    try:
        prepared = detector.prepare_image(image)
    except (TypeError, ValueError, cv.error) as exc:
        print(f"{Path(image_path).name}: {exc}")
        return "next"

    custom_rois = None
    last_signature = None
    base_display = normalize_for_display(image)
    result_image = base_display.copy()
    quality_image = np.full((680, 1000, 3), 25, dtype=np.uint8)
    cv.imshow(CONTROL_WINDOW, render_control_help())

    while True:
        roi_strip, control_params, signature = read_control_params()
        current_signature = (signature, str(custom_rois))
        if current_signature != last_signature:
            img_h, img_w = image.shape[:2]
            rois = custom_rois or SizeDetector.default_rois(
                img_w,
                img_h,
                strip=min(roi_strip, max(img_h, img_w)),
            )
            detector.update_params(
                {
                    **control_params,
                    "rois": rois,
                },
                clear_result=False,
            )
            result = detector.detect_prepared(prepared)
            result_image = draw_detection_debug(
                image,
                detector,
                Path(image_path).name,
                base_display=base_display,
            )
            quality_image = render_quality_panel(detector.last_debug)
            if result.error_code != 0:
                print(f"{Path(image_path).name}: {result.error_msg}")
            else:
                summary = detector.last_debug.get("summary", {})
                print(
                    f"{Path(image_path).name}: "
                    f"W={result.width:.6f}mm H={result.height:.6f}mm "
                    f"raw=({summary.get('raw_width_mm', 0):.6f}, "
                    f"{summary.get('raw_height_mm', 0):.6f})mm"
                )
            last_signature = current_signature
            cv.imshow(RESULT_WINDOW, result_image)
            cv.imshow(QUALITY_WINDOW, quality_image)

        key = cv.waitKey(30) & 0xFF

        if key == 27:
            return "quit"
        if key in (ord("q"), ord("Q")):
            return "next"
        if key in (ord("e"), ord("E")):
            return "previous"
        if key in (ord("r"), ord("R")):
            selected = select_four_rois(image)
            if selected is not None:
                custom_rois = selected
                last_signature = None
        if key in (ord("d"), ord("D")):
            custom_rois = None
            last_signature = None
        if key in (ord("s"), ord("S")):
            save_debug_images(
                args.save_dir,
                image_path,
                result_image,
                quality_image,
            )


# ==================== 命令行入口 ====================

def iter_image_paths(input_path):
    """返回文件或目录中的可测试图像。"""
    path = Path(input_path)
    extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in extensions
    )


def load_optional_image(path):
    """读取可选参考图。"""
    if not path:
        return None
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取参考图: {path}")
    return image


def validate_args(args):
    """统一验证命令行与人工轨迹条共用参数。"""
    errors = []

    def require_finite(name, value):
        if value is None or not np.isfinite(float(value)):
            errors.append(f"{name} 必须是有限数值")
            return False
        return True

    positive_values = {
        "pixel_size": args.pixel_size,
        "max_profile_rmse": args.max_profile_rmse,
        "max_transition_width": args.max_transition_width,
        "max_line_rmse": args.max_line_rmse,
        "max_sparse_line_rmse": args.max_sparse_line_rmse,
        "calibration_x_scale": args.calibration_x_scale,
        "calibration_y_scale": args.calibration_y_scale,
    }
    if args.pixel_size_x is not None:
        positive_values["pixel_size_x"] = args.pixel_size_x
    for name, value in positive_values.items():
        if require_finite(name, value) and float(value) <= 0:
            errors.append(f"{name} 必须大于 0")

    nonnegative_values = {
        "std_width": args.std_width,
        "std_height": args.std_height,
        "tolerance_x": args.tolerance_x,
        "tolerance_y": args.tolerance_y,
        "smooth_sigma": args.smooth_sigma,
        "min_contrast_ratio": args.min_contrast_ratio,
    }
    for name, value in nonnegative_values.items():
        if require_finite(name, value) and float(value) < 0:
            errors.append(f"{name} 不能小于 0")

    for name, value in {
        "calibration_x_offset": args.calibration_x_offset,
        "calibration_y_offset": args.calibration_y_offset,
    }.items():
        require_finite(name, value)

    bounded_values = {
        "candidate_strength_ratio": (args.candidate_strength_ratio, 0.01, 1.0),
        "min_contrast_ratio": (args.min_contrast_ratio, 0.0, 1.0),
        "max_profile_rmse": (args.max_profile_rmse, 0.001, 1.0),
        "min_profile_success_ratio": (
            args.min_profile_success_ratio,
            0.0,
            1.0,
        ),
        "min_line_inlier_ratio": (args.min_line_inlier_ratio, 0.0, 1.0),
        "min_line_inlier_span_ratio": (
            args.min_line_inlier_span_ratio,
            0.0,
            1.0,
        ),
    }
    for name, (value, lower, upper) in bounded_values.items():
        if require_finite(name, value) and not lower <= float(value) <= upper:
            errors.append(f"{name} 必须在 [{lower}, {upper}] 范围内")

    integer_ranges = {
        "roi_strip": (args.roi_strip, 12, 2000),
        "scan_step": (args.scan_step, 1, 20),
        "profile_average_half_width": (
            args.profile_average_half_width,
            0,
            8,
        ),
        "fit_radius": (args.fit_radius, 4, 100),
        "min_valid_profiles": (args.min_valid_profiles, 2, 10000),
    }
    for name, (value, lower, upper) in integer_ranges.items():
        if not lower <= int(value) <= upper:
            errors.append(f"{name} 必须在 [{lower}, {upper}] 范围内")

    ui_ranges = {
        "smooth_sigma": (args.smooth_sigma, 0.0, 5.0),
        "max_line_rmse": (args.max_line_rmse, 0.001, 2.0),
        "max_transition_width": (args.max_transition_width, 1.0, 50.0),
    }
    for name, (value, lower, upper) in ui_ranges.items():
        if require_finite(name, value) and not lower <= float(value) <= upper:
            errors.append(f"{name} 必须在 [{lower}, {upper}] 范围内")

    if errors:
        raise ValueError("参数验证失败:\n  - " + "\n  - ".join(errors))


def build_detector(args):
    """根据命令行参数创建优化检测器。"""
    detector = OptimizedSizeDetector(
        {
            "pixel_size": args.pixel_size,
            "pixel_size_x": args.pixel_size_x,
            "std_size": (args.std_width, args.std_height),
            "allow_tolerance_x": args.tolerance_x,
            "allow_tolerance_y": args.tolerance_y,
            "detect_direction": args.direction,
            "edge_polarity": args.edge_polarity,
            "scan_step": args.scan_step,
            "profile_average_half_width": args.profile_average_half_width,
            "smooth_sigma": args.smooth_sigma,
            "candidate_strength_ratio": args.candidate_strength_ratio,
            "fit_radius": args.fit_radius,
            "min_contrast_ratio": args.min_contrast_ratio,
            "max_profile_rmse": args.max_profile_rmse,
            "max_transition_width": args.max_transition_width,
            "min_valid_profiles": args.min_valid_profiles,
            "min_profile_success_ratio": args.min_profile_success_ratio,
            "min_line_inlier_ratio": args.min_line_inlier_ratio,
            "max_line_rmse": args.max_line_rmse,
            "min_line_inlier_span_ratio": args.min_line_inlier_span_ratio,
            "max_sparse_line_rmse": args.max_sparse_line_rmse,
            "calibration_x": (
                args.calibration_x_scale,
                args.calibration_x_offset,
            ),
            "calibration_y": (
                args.calibration_y_scale,
                args.calibration_y_offset,
            ),
        }
    )
    detector.set_reference_images(
        load_optional_image(args.dark),
        load_optional_image(args.flat),
    )
    return detector


def run_without_ui(detector, image_paths, args):
    """无界面批量执行，输出结果并可保存调试图。"""
    exit_code = 0
    success_count = 0
    elapsed_values = []
    for image_path in image_paths:
        image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
        if image is None:
            print(f"无法读取图像: {image_path}")
            exit_code = 1
            continue

        img_h, img_w = image.shape[:2]
        detector.update_params(
            {
                "rois": SizeDetector.default_rois(
                    img_w,
                    img_h,
                    strip=args.roi_strip,
                )
            },
            clear_result=False,
        )
        start_time = time.perf_counter()
        result = detector.detect(image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        elapsed_values.append(elapsed_ms)
        if result.error_code != 0:
            print(
                f"{image_path}: FAIL - {result.error_msg} "
                f"time={elapsed_ms:.1f}ms"
            )
            exit_code = 1
        else:
            success_count += 1
            summary = detector.last_debug.get("summary", {})
            print(
                f"{image_path}: width={result.width:.6f}mm "
                f"height={result.height:.6f}mm "
                f"valid={result.is_valid} "
                f"polarity={summary.get('edge_polarity', 'unknown')} "
                f"time={elapsed_ms:.1f}ms"
            )
            if not result.is_valid:
                exit_code = 1
        if args.save_debug:
            saved = save_debug_images(
                args.save_dir,
                image_path,
                draw_detection_debug(image, detector, image_path.name),
                render_quality_panel(detector.last_debug),
            )
            if not saved:
                exit_code = 1

    total_count = len(image_paths)
    total_ms = float(np.sum(elapsed_values))
    average_ms = total_ms / max(1, len(elapsed_values))
    print(
        f"SUMMARY: success={success_count}/{total_count} "
        f"total={total_ms:.1f}ms average={average_ms:.1f}ms/image"
    )
    return exit_code


def parse_args():
    parser = argparse.ArgumentParser(
        description="优化灰度亚像素尺寸检测人工测试脚本"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(_ROOT / "Image" / "Dry"),
        help="图像文件或图像目录",
    )
    parser.add_argument("--pixel-size", type=float, default=0.014)
    parser.add_argument("--pixel-size-x", type=float, default=None)
    parser.add_argument("--std-width", type=float, default=0.0)
    parser.add_argument("--std-height", type=float, default=0.0)
    parser.add_argument("--tolerance-x", type=float, default=0.0)
    parser.add_argument("--tolerance-y", type=float, default=0.0)
    parser.add_argument(
        "--direction",
        choices=("outward", "inward"),
        default="outward",
    )
    parser.add_argument(
        "--edge-polarity",
        choices=("auto", "falling", "rising"),
        default="auto",
    )
    parser.add_argument("--roi-strip", type=int, default=120)
    parser.add_argument("--scan-step", type=int, default=4)
    parser.add_argument("--profile-average-half-width", type=int, default=3)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--candidate-strength-ratio", type=float, default=0.45)
    parser.add_argument("--fit-radius", type=int, default=8)
    parser.add_argument("--min-contrast-ratio", type=float, default=0.12)
    parser.add_argument("--max-profile-rmse", type=float, default=0.12)
    parser.add_argument("--max-transition-width", type=float, default=12.0)
    parser.add_argument("--min-valid-profiles", type=int, default=12)
    parser.add_argument("--min-profile-success-ratio", type=float, default=0.20)
    parser.add_argument("--min-line-inlier-ratio", type=float, default=0.50)
    parser.add_argument("--max-line-rmse", type=float, default=0.50)
    parser.add_argument(
        "--min-line-inlier-span-ratio",
        type=float,
        default=0.65,
    )
    parser.add_argument("--max-sparse-line-rmse", type=float, default=1.25)
    parser.add_argument("--calibration-x-scale", type=float, default=1.0)
    parser.add_argument("--calibration-x-offset", type=float, default=0.0)
    parser.add_argument("--calibration-y-scale", type=float, default=1.0)
    parser.add_argument("--calibration-y-offset", type=float, default=0.0)
    parser.add_argument("--dark", default=None, help="可选暗场参考图")
    parser.add_argument("--flat", default=None, help="可选平场参考图")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument(
        "--save-dir",
        default=str(_ROOT / "test" / "output" / "size_detector_test"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(str(exc))
        return 2

    image_paths = iter_image_paths(args.input)
    if not image_paths:
        print(f"未找到可测试图像: {args.input}")
        return 1

    try:
        detector = build_detector(args)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.no_ui:
        return run_without_ui(detector, image_paths, args)

    index = 0
    while 0 <= index < len(image_paths):
        action = run_manual_ui(detector, image_paths[index], args)
        if action == "quit":
            break
        if action == "previous":
            index = max(0, index - 1)
        else:
            index += 1
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

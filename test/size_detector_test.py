"""优化尺寸检测人工测试脚本。

本文件不修改生产 SizeDetector。优化管线：
  原始灰度/高位深图 → 可选暗场与平场校正 → 极性自动判别
  → 梯度候选共识直线（Consensus）→ 约束窗口内亮升沿 50% 亚像素定位（Refine）
  → 鲁棒直线拟合 → 四边尺寸计算
  → 独立线性量具标定（不按标准尺寸逐件回拉）。

边缘判据：产品内侧暗环最低点与外侧第一段亮平台的中点（亮升沿 50%）。
该判据相对其它判据存在约 6 µm 的固定偏移，由 calibration_x/y 吸收。

调试人员只需调整两个参数：
  roi_strip   ROI 搜索深度（px）。只决定能否找到边，不影响测量值。
  sensitivity 边缘灵敏度 0~100。50 为验证过的工作点；调高只会增加拒检，
              调低会放宽门限并可能接受劣质边缘。

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
from pathlib import Path

import cv2 as cv
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.detectors.size_detector import SizeDetector  # noqa: E402
from src.support.data_structure import Size_Result  # noqa: E402


# 边缘搜索窗几何：内侧只需覆盖暗环，外侧必须够深以在离焦时仍触及亮平台。
INNER_SPAN = 10
OUTER_SPAN = 20
# 精修阶段允许偏离共识直线的最大距离（px）。
GUIDE_BAND = 6.0
# 剖面沿深度方向的高斯平滑，稳定梯度峰与宽度测量。
SMOOTH_SIGMA = 1.0
# 判据电平估计所用的额外平滑，消除“噪声最小值”统计偏差。
LEVEL_SIGMA = 1.6


# ==================== 灰度与平场校正 ====================

def to_gray_float(image):
    """转为 float32 灰度图，保留 8/10/12/16 bit 的原始灰度级。"""
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

    return np.ascontiguousarray(gray, dtype=np.float32)


def apply_flat_field(image_gray, dark_reference=None, flat_reference=None):
    """执行暗场/平场校正；无参考图时原样返回。"""
    work = np.asarray(image_gray, dtype=np.float32)

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


def image_dynamic_range(work):
    """抽稀网格上的 p1~p99 灰度跨度，比全图 percentile 快一个数量级。"""
    sample = work[::3, ::3].ravel()
    q_low, q_high = np.percentile(sample, [1.0, 99.0])
    return float(q_high - q_low)


# ==================== 灵敏度映射 ====================

# sensitivity=0 宽松 / 50 工作点 / 100 严格。工作点由 19 张实测图的门限扫描
# 确定：该点通过 16/19 且无粗差，再放宽 max_line_rmse 即出现粗差。
_SENSITIVITY_TABLE = {
    "min_contrast_ratio": (0.12, 0.30, 0.50),
    "max_edge_width": (12.0, 7.0, 4.5),
    "min_profile_success_ratio": (0.20, 0.35, 0.55),
    "min_line_inlier_ratio": (0.20, 0.35, 0.60),
    "max_line_rmse": (1.00, 0.60, 0.35),
    "min_line_inlier_span_ratio": (0.50, 0.70, 0.85),
    "robust_sigma": (3.5, 2.8, 2.2),
    "max_outlier_distance": (2.5, 1.5, 1.0),
}


def sensitivity_to_gates(sensitivity):
    """把 0~100 的灵敏度插值成一整套质量门限。"""
    s = float(np.clip(sensitivity, 0.0, 100.0))
    gates = {}
    for key, (low, mid, high) in _SENSITIVITY_TABLE.items():
        if s <= 50.0:
            gates[key] = low + (mid - low) * (s / 50.0)
        else:
            gates[key] = mid + (high - mid) * ((s - 50.0) / 50.0)
    return gates


# ==================== 极性与剖面 ====================

def normalize_edge_polarity(edge_polarity):
    """将边缘极性归一为 auto / rising / falling。"""
    if edge_polarity in ("rising", "falling"):
        return edge_polarity
    return "auto"


def oriented_profiles(work, roi, roi_name, direction, scan_step,
                      average_half_width, smooth_sigma=SMOOTH_SIGMA):
    """取出单边 ROI 的定向剖面矩阵 (扫描线数, 深度)，深度索引 0 为产品内侧。

    沿边长方向做箱式平均抑制随机噪声，沿深度方向做零相位高斯平滑稳定梯度峰
    与 10-90% 宽度的测量；两者都不移动对称边缘的中心位置。
    """
    roi_x, roi_y, roi_w, roi_h = roi
    sub = np.ascontiguousarray(work[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])
    is_horizontal = roi_name in ("top", "bottom")

    if average_half_width > 0:
        kernel = 2 * int(average_half_width) + 1
        sub = cv.blur(sub, (kernel, 1) if is_horizontal else (1, kernel))

    profiles = np.ascontiguousarray(sub.T) if is_horizontal else sub
    if SizeDetector._is_reverse_for_side(roi_name, direction):
        profiles = np.ascontiguousarray(profiles[:, ::-1])
    profiles = np.ascontiguousarray(profiles[::max(1, int(scan_step))])
    if smooth_sigma > 0:
        profiles = cv.GaussianBlur(
            profiles, (0, 0), sigmaX=float(smooth_sigma), sigmaY=0,
            borderType=cv.BORDER_REFLECT_101,
        )
    return profiles, is_horizontal


def estimate_edge_polarity(work, rois, direction, params):
    """比较四边 ROI 内侧与外侧四分之一的平均灰度，判断产品外轮廓极性。"""
    configured = normalize_edge_polarity(params.get("edge_polarity", "auto"))
    if configured != "auto":
        return configured, {}

    rising_score = 0.0
    for roi_name, roi in rois.items():
        profiles, _ = oriented_profiles(
            work, roi, roi_name, direction, 8, 0, smooth_sigma=0.0
        )
        quarter = max(2, profiles.shape[1] // 4)
        rising_score += float(
            profiles[:, -quarter:].mean() - profiles[:, :quarter].mean()
        )

    selected = "rising" if rising_score >= 0.0 else "falling"
    return selected, {"rising_score": rising_score}


# ==================== 共识定位与亚像素精修 ====================

def _consensus_guide(profiles_signed, margin):
    """由各扫描线的最强梯度位置拟合共识直线，作为精修阶段的搜索中心。

    中位数对不足半数扫描线的错误锁定免疫；随后两轮直线拟合把倾斜也吃掉。
    """
    gradient = np.diff(profiles_signed, axis=1)
    depth = profiles_signed.shape[1]
    peak = (
        np.argmax(gradient[:, margin:depth - margin], axis=1) + margin
    ).astype(np.float64)

    median_peak = float(np.median(peak))
    mad = float(np.median(np.abs(peak - median_peak)))
    tolerance = max(3.0, 3.0 * 1.4826 * mad)
    rows = np.arange(len(peak), dtype=np.float64)

    keep = np.abs(peak - median_peak) <= tolerance
    if np.count_nonzero(keep) < 8:
        return np.full(len(peak), median_peak)

    guide = np.polyval(np.polyfit(rows[keep], peak[keep], 1), rows)
    keep = np.abs(peak - guide) <= tolerance
    if np.count_nonzero(keep) >= 8:
        guide = np.polyval(np.polyfit(rows[keep], peak[keep], 1), rows)
    return guide


def _plateau_offset(aggregate, inner_span, outer_span):
    """从跨扫描线的中位剖面找出上升沿结束、进入亮平台的偏移量。"""
    span = float(np.median(aggregate[-max(3, outer_span // 3):])
                 - aggregate[:inner_span + 1].min())
    if not np.isfinite(span) or span <= 1e-9:
        return None
    threshold = aggregate[:inner_span + 1].min() + 0.98 * span
    tail = aggregate[inner_span:] >= threshold
    reached = int(np.argmax(tail)) if tail.any() else outer_span
    return int(np.clip(reached + 2, 4, outer_span - 2))


def locate_rise50_edges(profiles, edge_polarity, global_dynamic, gates):
    """两阶段单边边缘定位，返回亚像素位置与逐扫描线质量。"""
    signed = profiles if edge_polarity == "rising" else -profiles
    n_scan, depth = signed.shape
    margin = max(INNER_SPAN, OUTER_SPAN) + 2
    if depth - 2 * margin < 3:
        return None, "ROI 搜索深度不足"

    guide = _consensus_guide(signed, margin)
    base = np.clip(
        np.rint(guide).astype(np.int64), INNER_SPAN, depth - OUTER_SPAN - 1
    )
    offsets = np.arange(-INNER_SPAN, OUTER_SPAN + 1)
    window = np.ascontiguousarray(
        np.take_along_axis(signed, base[:, None] + offsets[None, :], axis=1),
        dtype=np.float32,
    )
    smoothed = cv.GaussianBlur(
        window, (0, 0), sigmaX=LEVEL_SIGMA, sigmaY=0,
        borderType=cv.BORDER_REFLECT_101,
    )

    plateau = _plateau_offset(
        np.median(smoothed, axis=0).astype(np.float64), INNER_SPAN, OUTER_SPAN
    )
    if plateau is None:
        return None, "边缘对比度不足"

    window = window.astype(np.float64)
    smoothed = smoothed.astype(np.float64)
    bright = np.median(smoothed[:, INNER_SPAN + plateau:], axis=1)

    # 暗环最低点：抛物线顶点细化，避免“噪声最小值”把电平压低
    inner = smoothed[:, :INNER_SPAN + 1]
    trough_idx = np.argmin(inner, axis=1)
    center = np.clip(trough_idx, 1, INNER_SPAN - 1)
    left = np.take_along_axis(inner, (center - 1)[:, None], 1)[:, 0]
    middle = np.take_along_axis(inner, center[:, None], 1)[:, 0]
    right = np.take_along_axis(inner, (center + 1)[:, None], 1)[:, 0]
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

    position = base - INNER_SPAN + cross50
    edge_width = cross90 - cross10
    contrast_ratio = contrast / max(float(global_dynamic), 1e-9)

    accepted = (
        usable
        & np.isfinite(position)
        & np.isfinite(edge_width)
        & (np.abs(position - guide) <= GUIDE_BAND)
        & (contrast_ratio >= float(gates["min_contrast_ratio"]))
        & (edge_width <= float(gates["max_edge_width"]))
    )

    return {
        "position": position,
        "contrast_ratio": contrast_ratio,
        "edge_width": edge_width,
        "accepted": accepted,
        "guide": guide,
        "window": window,
        "smoothed": smoothed,
        "dark": dark,
        "bright": bright,
        "plateau": plateau,
    }, None


# ==================== 鲁棒直线拟合 ====================

def robust_fit_edge_line(points, is_horizontal, gates):
    """迭代 MAD 剔除离群点，拟合 y=ax+b 或 x=ay+b。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None

    independent = points[:, 0] if is_horizontal else points[:, 1]
    dependent = points[:, 1] if is_horizontal else points[:, 0]
    mask = np.ones(len(points), dtype=bool)
    robust_sigma = float(gates.get("robust_sigma", 2.8))
    max_outlier_distance = float(gates.get("max_outlier_distance", 1.5))

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
            max_outlier_distance, max(0.10, robust_sigma * robust_scale)
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
    independent_span = max(float(np.ptp(independent)), 1e-9)
    return {
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "inlier_mask": mask,
        "line_rmse": float(np.sqrt(np.mean(residual[mask] ** 2))),
        "inlier_count": int(np.count_nonzero(mask)),
        "inlier_ratio": float(np.count_nonzero(mask) / len(mask)),
        "inlier_span_ratio": float(np.ptp(independent[mask]) / independent_span),
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
            # 两个调试参数
            "roi_strip": 120,
            "sensitivity": 50.0,
            # 以下为固定实现常量，非调试参数
            "scan_step": 1,
            "profile_average_half_width": 3,
            "min_valid_profiles": 12,
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
            elif key == "sensitivity":
                self.params[key] = float(np.clip(float(value), 0.0, 100.0))
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
        work = apply_flat_field(
            to_gray_float(image), self.dark_reference, self.flat_reference
        )
        global_dynamic = image_dynamic_range(work)
        if global_dynamic <= 1e-9:
            raise ValueError("图像灰度动态范围不足")
        return {"work": work, "global_dynamic": global_dynamic}

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

        gates = sensitivity_to_gates(self.params.get("sensitivity", 50.0))
        direction = SizeDetector.normalize_detect_direction(
            self.params.get("detect_direction", "outward")
        )
        edge_polarity, polarity_scores = estimate_edge_polarity(
            work, rois, direction, self.params
        )
        scan_step = max(1, int(self.params.get("scan_step", 1)))
        average_half_width = max(
            0, int(self.params.get("profile_average_half_width", 3))
        )
        min_valid = max(4, int(self.params.get("min_valid_profiles", 12)))

        debug = {}
        lines = {}
        for roi_name in self.ROI_SIDES:
            roi = rois[roi_name]
            profiles, is_horizontal = oriented_profiles(
                work, roi, roi_name, direction, scan_step, average_half_width
            )
            info, error = locate_rise50_edges(
                profiles, edge_polarity, global_dynamic, gates
            )
            if info is None:
                return self._fail(f"{roi_name}边{error}", debug)

            accepted = info["accepted"]
            attempted = int(len(accepted))
            valid_count = int(np.count_nonzero(accepted))
            success_ratio = valid_count / max(1, attempted)
            if (
                valid_count < min_valid
                or success_ratio < float(gates["min_profile_success_ratio"])
            ):
                debug[roi_name] = self._side_debug(
                    roi, is_horizontal, edge_polarity, info, attempted,
                    valid_count, None, np.empty((0, 2)),
                )
                return self._fail(
                    f"{roi_name}边有效扫描线不足: {valid_count}/{attempted}",
                    debug,
                )

            depth = profiles.shape[1]
            position = info["position"][accepted]
            if SizeDetector._is_reverse_for_side(roi_name, direction):
                position = depth - 1 - position
            cross = np.flatnonzero(accepted) * scan_step
            roi_x, roi_y = roi[0], roi[1]
            points = (
                np.column_stack([roi_x + cross, roi_y + position])
                if is_horizontal
                else np.column_stack([roi_x + position, roi_y + cross])
            )

            line = robust_fit_edge_line(points, is_horizontal, gates)
            debug[roi_name] = self._side_debug(
                roi, is_horizontal, edge_polarity, info, attempted,
                valid_count, line, points,
            )
            if line is None:
                return self._fail(f"{roi_name}边直线拟合失败", debug)
            if line["inlier_ratio"] < float(gates["min_line_inlier_ratio"]):
                return self._fail(
                    f"{roi_name}边内点比例不足: {line['inlier_ratio']:.3f}", debug
                )
            if line["line_rmse"] > float(gates["max_line_rmse"]):
                return self._fail(
                    f"{roi_name}边拟合残差过大: {line['line_rmse']:.3f}px", debug
                )
            if line["inlier_span_ratio"] < float(
                gates["min_line_inlier_span_ratio"]
            ):
                return self._fail(
                    f"{roi_name}边内点跨度不足: "
                    f"{line['inlier_span_ratio']:.3f}",
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
                raw_width_mm, self.params.get("calibration_x", (1.0, 0.0))
            )
            height_mm = self._calibrate(
                raw_height_mm, self.params.get("calibration_y", (1.0, 0.0))
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
            "gates": gates,
        }

        self.last_debug = debug
        self.detection_result = Size_Result(
            width=float(width_mm),
            height=float(height_mm),
            box_points=box_points,
            is_valid=bool(is_valid),
        )
        return self.detection_result

    @staticmethod
    def _side_debug(roi, is_horizontal, edge_polarity, info, attempted,
                    valid_count, line, points):
        """整理单边调试信息，供可视化与质量面板使用。"""
        accepted = info["accepted"]
        entry = {
            "roi": tuple(int(v) for v in roi),
            "is_horizontal": is_horizontal,
            "edge_polarity": edge_polarity,
            "attempted": attempted,
            "valid": valid_count,
            "points": np.asarray(points, dtype=np.float64).reshape(-1, 2),
            "plateau": int(info["plateau"]),
        }
        if valid_count > 0:
            entry["contrast_ratio"] = float(
                np.median(info["contrast_ratio"][accepted])
            )
            entry["edge_width"] = float(np.median(info["edge_width"][accepted]))
            middle = int(np.flatnonzero(accepted)[valid_count // 2])
        else:
            entry["contrast_ratio"] = 0.0
            entry["edge_width"] = 0.0
            middle = int(len(accepted) // 2)
        entry["representative"] = {
            "window": info["window"][middle],
            "smoothed": info["smoothed"][middle],
            "dark": float(info["dark"][middle]),
            "bright": float(info["bright"][middle]),
            "inner_span": INNER_SPAN,
        }
        if line is not None:
            entry.update(line)
        return entry


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
            (gray - q_low) * 255.0 / (q_high - q_low), 0, 255
        ).astype(np.uint8)
    return cv.cvtColor(display, cv.COLOR_GRAY2BGR)


def _put_hud_text(image, text, origin, color, scale=0.65, thickness=1):
    x, y = origin
    cv.putText(
        image, text, (x + 1, y + 1), cv.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv.LINE_AA,
    )
    cv.putText(
        image, text, origin, cv.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv.LINE_AA,
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
            canvas, (roi_x, roi_y), (roi_x + roi_w - 1, roi_y + roi_h - 1),
            (0, 165, 255), 1,
        )

        points = np.asarray(info.get("points", []), dtype=np.float64)
        inlier_mask = info.get("inlier_mask")
        if len(points) > 0:
            if inlier_mask is None:
                inlier_mask = np.zeros(len(points), dtype=bool)
            for idx, point in enumerate(points):
                color = (0, 220, 0) if inlier_mask[idx] else (0, 0, 255)
                cv.circle(
                    canvas, (int(round(point[0])), int(round(point[1]))),
                    1, color, -1, cv.LINE_AA,
                )

        if "slope" in info and "intercept" in info:
            if info["is_horizontal"]:
                x0, x1 = roi_x, roi_x + roi_w - 1
                p0 = (int(x0), int(round(evaluate_edge_line(info, x0))))
                p1 = (int(x1), int(round(evaluate_edge_line(info, x1))))
            else:
                y0, y1 = roi_y, roi_y + roi_h - 1
                p0 = (int(round(evaluate_edge_line(info, y0))), int(y0))
                p1 = (int(round(evaluate_edge_line(info, y1))), int(y1))
            cv.line(canvas, p0, p1, (0, 255, 255), 1, cv.LINE_AA)

    result = detector.detection_result
    if result is not None and result.error_code == 0:
        x, y, width, height = result.box_points
        color = (0, 255, 0) if result.is_valid else (0, 0, 255)
        cv.rectangle(
            canvas, (int(round(x)), int(round(y))),
            (int(round(x + width)), int(round(y + height))),
            color, 2, cv.LINE_AA,
        )
        summary = debug.get("summary", {})
        lines = [
            f"{image_name}",
            f"W={result.width:.6f} mm  H={result.height:.6f} mm",
            f"STATUS={'OK' if result.is_valid else 'NG'}"
            f"  polarity={summary.get('edge_polarity', '?')}",
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


def _draw_curve(canvas, values, rect, color, value_range):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return
    x, y, width, height = rect
    v_min, v_max = value_range
    if v_max - v_min <= 1e-9:
        return
    xs = np.linspace(x, x + width - 1, len(values))
    ys = y + height - 1 - (values - v_min) * (height - 1) / (v_max - v_min)
    points = np.column_stack([xs, ys]).astype(np.int32)
    cv.polylines(canvas, [points], False, color, 1, cv.LINE_AA)


def render_quality_panel(debug, width=1000, height=680):
    """绘制四边代表扫描线、判据电平和质量指标。"""
    canvas = np.full((height, width, 3), 25, dtype=np.uint8)
    margin = 12
    cell_width = (width - margin * 3) // 2
    cell_height = (height - margin * 3) // 2

    for idx, roi_name in enumerate(SizeDetector.ROI_SIDES):
        row, col = divmod(idx, 2)
        x0 = margin + col * (cell_width + margin)
        y0 = margin + row * (cell_height + margin)
        cv.rectangle(
            canvas, (x0, y0), (x0 + cell_width - 1, y0 + cell_height - 1),
            (70, 70, 70), 1,
        )

        info = debug.get(roi_name) or {}
        representative = info.get("representative")
        label = (
            f"{roi_name}: profiles={info.get('valid', 0)}/"
            f"{info.get('attempted', 0)}"
        )
        if info.get("line_rmse") is not None:
            label += f" line_rmse={info['line_rmse']:.3f}px"
        _put_hud_text(canvas, label, (x0 + 8, y0 + 20), (240, 240, 240), 0.48)

        if not representative:
            continue

        window = np.asarray(representative["window"], dtype=np.float64)
        dark = representative["dark"]
        bright = representative["bright"]
        span = max(bright - dark, 1e-9)
        plot = (x0 + 8, y0 + 35, cell_width - 16, cell_height - 48)
        value_range = (dark - 0.25 * span, bright + 0.25 * span)

        _draw_curve(canvas, window, plot, (130, 130, 130), value_range)
        _draw_curve(
            canvas, representative["smoothed"], plot, (0, 220, 255), value_range
        )

        # 暗环 / 50% / 亮平台三条判据电平
        for level, color in (
            (dark, (255, 120, 60)),
            (dark + 0.5 * span, (0, 255, 0)),
            (bright, (255, 120, 60)),
        ):
            y = int(
                plot[1] + plot[3] - 1
                - (level - value_range[0]) * (plot[3] - 1)
                / (value_range[1] - value_range[0])
            )
            cv.line(canvas, (plot[0], y), (plot[0] + plot[2] - 1, y), color, 1)

        quality = (
            f"contrast={info.get('contrast_ratio', 0):.3f}  "
            f"edge_width={info.get('edge_width', 0):.2f}px  "
            f"inlier={info.get('inlier_ratio', 0):.3f}"
        )
        _put_hud_text(
            canvas, quality, (x0 + 8, y0 + cell_height - 8), (190, 190, 190), 0.42
        )

    _put_hud_text(
        canvas,
        "gray=window  yellow=smoothed  orange=dark/bright levels  green=50% level",
        (margin, height - 4), (190, 190, 190), 0.42,
    )
    return canvas


def init_windows(args):
    """初始化人工调参窗口，只暴露两个调试参数。"""
    global _WINDOWS_READY
    if _WINDOWS_READY:
        return

    cv.namedWindow(CONTROL_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(CONTROL_WINDOW, 760, 300)
    cv.namedWindow(RESULT_WINDOW, cv.WINDOW_NORMAL)
    cv.namedWindow(QUALITY_WINDOW, cv.WINDOW_NORMAL)

    cv.createTrackbar("roi_strip", CONTROL_WINDOW, int(args.roi_strip), 400, _noop)
    cv.createTrackbar(
        "sensitivity", CONTROL_WINDOW, int(round(args.sensitivity)), 100, _noop
    )
    _WINDOWS_READY = True


def read_control_params():
    """读取两个轨迹条，返回 ROI 深度与检测参数。"""
    roi_strip = max(40, cv.getTrackbarPos("roi_strip", CONTROL_WINDOW))
    sensitivity = float(cv.getTrackbarPos("sensitivity", CONTROL_WINDOW))
    params = {"roi_strip": roi_strip, "sensitivity": sensitivity}
    return roi_strip, params, (roi_strip, sensitivity)


def render_control_help(width=760, height=260):
    """显示两个参数的含义与操作提示。"""
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    lines = [
        "Optimized grayscale subpixel detector (rise-50% criterion)",
        "",
        "roi_strip: four-side ROI search depth in pixels.",
        "  Only decides whether the edge is inside the ROI.",
        "  Verified neutral: 80..250 px changes the reading by <0.02 um.",
        "",
        "sensitivity: edge quality strictness, 0..100.",
        "  50 = validated operating point (16/19 images, no gross error).",
        "  Higher  -> only crisp edges accepted, more rejects, never wrong.",
        "  Lower   -> weak/soft edges accepted, may admit a bad edge.",
    ]
    for idx, line in enumerate(lines):
        _put_hud_text(panel, line, (12, 22 + idx * 24), (220, 220, 220), 0.46)
    return panel


def select_four_rois(image):
    """依次人工框选 top/left/bottom/right ROI；取消任一边则整体取消。"""
    display = normalize_for_display(image)
    rois = {}
    for side in SizeDetector.ROI_SIDES:
        window_name = f"select_{side}_roi"
        print(f"请框选 {side} 边 ROI，Enter/Space 确认，ESC 取消")
        roi = cv.selectROI(
            window_name, display, showCrosshair=True, fromCenter=False
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
                img_w, img_h, strip=min(roi_strip, max(img_h, img_w))
            )
            detector.update_params(
                {**control_params, "rois": rois}, clear_result=False
            )
            start_time = time.perf_counter()
            result = detector.detect_prepared(prepared)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            result_image = draw_detection_debug(
                image, detector, Path(image_path).name,
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
                    f"{summary.get('raw_height_mm', 0):.6f})mm "
                    f"time={elapsed_ms:.1f}ms"
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
                args.save_dir, image_path, result_image, quality_image
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
    """验证命令行参数。"""
    errors = []

    def require_finite(name, value):
        if value is None or not np.isfinite(float(value)):
            errors.append(f"{name} 必须是有限数值")
            return False
        return True

    positive_values = {
        "pixel_size": args.pixel_size,
        "calibration_x_scale": args.calibration_x_scale,
        "calibration_y_scale": args.calibration_y_scale,
    }
    if args.pixel_size_x is not None:
        positive_values["pixel_size_x"] = args.pixel_size_x
    for name, value in positive_values.items():
        if require_finite(name, value) and float(value) <= 0:
            errors.append(f"{name} 必须大于 0")

    for name, value in {
        "std_width": args.std_width,
        "std_height": args.std_height,
        "tolerance_x": args.tolerance_x,
        "tolerance_y": args.tolerance_y,
    }.items():
        if require_finite(name, value) and float(value) < 0:
            errors.append(f"{name} 不能小于 0")

    for name, value in {
        "calibration_x_offset": args.calibration_x_offset,
        "calibration_y_offset": args.calibration_y_offset,
    }.items():
        require_finite(name, value)

    if not 0.0 <= float(args.sensitivity) <= 100.0:
        errors.append("sensitivity 必须在 [0, 100] 范围内")
    if not 40 <= int(args.roi_strip) <= 400:
        errors.append("roi_strip 必须在 [40, 400] 范围内")

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
            "roi_strip": args.roi_strip,
            "sensitivity": args.sensitivity,
            "calibration_x": (
                args.calibration_x_scale, args.calibration_x_offset
            ),
            "calibration_y": (
                args.calibration_y_scale, args.calibration_y_offset
            ),
        }
    )
    detector.set_reference_images(
        load_optional_image(args.dark), load_optional_image(args.flat)
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
                    img_w, img_h, strip=args.roi_strip
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

    total_ms = float(np.sum(elapsed_values))
    average_ms = total_ms / max(1, len(elapsed_values))
    print(
        f"SUMMARY: success={success_count}/{len(image_paths)} "
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
        "--direction", choices=("outward", "inward"), default="outward"
    )
    parser.add_argument(
        "--edge-polarity",
        choices=("auto", "falling", "rising"),
        default="auto",
    )
    parser.add_argument(
        "--roi-strip", type=int, default=120, help="ROI 搜索深度（调试参数 1）"
    )
    parser.add_argument(
        "--sensitivity", type=float, default=50.0,
        help="边缘灵敏度 0~100，50 为验证工作点（调试参数 2）",
    )
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

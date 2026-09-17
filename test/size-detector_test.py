"""实验沙箱：四边 ROI + 1D 投影的频域滤波尺寸检测。

本文件不是生产 SizeDetector。管线：
  灰度（可选二值）ROI 投影 → 实数 FFT → Butterworth 低通/带通 → IFFT
  → 黄线局部极小（最多 5 个）且 raw < mean(raw)*0.6，否则回退黄线全局最低。

运行可视化：
  python test/size-detector_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.detectors.size_detector import (  # noqa: E402
    SizeDetector,
    calculate_projection_curve,
)
from src.support.data_structure import Product, Size_Result  # noqa: E402
from src.support.support_funs import draw_detection_results, ensure_gray_u8  # noqa: E402


# ==================== 频域滤波 ====================

def fft_band_filter_curve(curve, lp_ratio, hp_ratio=0.0, order=4):
    """对 1D 曲线做零相位 Butterworth 低通（hp=0）或带通。

    lp_ratio / hp_ratio 是相对于奈奎斯特频率的比例，范围约 (0, 1]。
    使用反射填充减轻 FFT 循环卷积在边界上的振铃。
    """
    x = np.asarray(curve, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return x.copy()

    order = max(1, int(order))
    lp_ratio = float(np.clip(lp_ratio, 1e-3, 1.0))
    hp_ratio = float(np.clip(hp_ratio, 0.0, 0.999))

    pad = n
    xp = np.pad(x, pad, mode="reflect")
    spec = np.fft.rfft(xp)
    freqs = np.fft.rfftfreq(xp.size)

    nyquist = 0.5
    df = float(freqs[1]) if freqs.size > 1 else 1e-6
    fc_lp = float(np.clip(lp_ratio * nyquist, df, nyquist * 0.99))
    h_lp = 1.0 / (1.0 + (freqs / fc_lp) ** (2 * order))

    if hp_ratio <= 1e-6:
        h = h_lp
    else:
        fc_hp = float(np.clip(hp_ratio * nyquist, df, fc_lp * 0.99))
        h_hp = np.ones_like(freqs)
        h_hp[0] = 0.0
        nz = freqs > 0
        h_hp[nz] = 1.0 / (1.0 + (fc_hp / freqs[nz]) ** (2 * order))
        h = h_lp * h_hp

    y = np.fft.irfft(spec * h, n=xp.size).real
    return y[pad : pad + n]


def _is_local_min(curve, i):
    n = len(curve)
    if n == 1:
        return True
    if i == 0:
        return curve[0] <= curve[1]
    if i == n - 1:
        return curve[-1] <= curve[-2]
    return curve[i] <= curve[i - 1] and curve[i] < curve[i + 1]


def select_boundary_from_curves(raw, filtered, raw_ratio=0.6, max_candidates=5):
    """黄线局部极小从低到高最多 5 个；raw < mean(raw)*ratio 则采用，否则回退黄线全局最低。"""
    raw = np.asarray(raw, dtype=np.float64)
    filtered = np.asarray(filtered, dtype=np.float64)
    n = int(filtered.size)
    if n == 0 or int(raw.size) != n:
        return None

    thresh = float(np.mean(raw)) * float(raw_ratio)
    minima = [i for i in range(n) if _is_local_min(filtered, i)]
    minima.sort(key=lambda i: (float(filtered[i]), i))
    for i in minima[: max(0, int(max_candidates))]:
        if raw[i] < thresh:
            return float(i)
    return float(np.argmin(filtered))


# ==================== 实验检测器 ====================

class FftSizeDetector:
    """沙箱用四边 ROI 尺寸检测，不写入生产 SizeDetector。"""

    def __init__(self, params: dict = None):
        self.image = None
        self.detection_result = None
        self.last_debug = {}
        self.params = {
            "min_threshold": 0,
            "max_threshold": 255,
            "allow_tolerance_x": 0.0,
            "allow_tolerance_y": 0.0,
            "rois": {side: None for side in SizeDetector.ROI_SIDES},
            "std_size": (0.0, 0.0),
            "pixel_size": 0.001,
            "pixel_size_x": None,
            "detect_direction": "outward",
            "lp_ratio": 0.15,
            "hp_ratio": 0.0,
            "filter_order": 4,
            "use_binary": False,
            "raw_dark_ratio": 0.6,
            "max_minima": 5,
        }
        if params:
            self.update_params(params)

    def update_params(self, params: dict, clear_result: bool = True) -> bool:
        if not params:
            return True
        for key, val in params.items():
            if key == "rois":
                self.params["rois"] = SizeDetector._normalize_rois(val)
            elif key == "detect_direction":
                self.params["detect_direction"] = SizeDetector.normalize_detect_direction(
                    val
                )
            elif key == "use_binary":
                self.params["use_binary"] = bool(val)
            else:
                self.params[key] = val
        if clear_result:
            self.detection_result = None
            self.last_debug = {}
        return True

    def detect(self, image) -> Size_Result:
        image_gray = ensure_gray_u8(image, copy=True)
        self.image = image_gray
        h, w = image_gray.shape

        rois, roi_err = self._validate_rois(w, h)
        if roi_err is not None:
            self.detection_result = Size_Result(
                error_code=2, error_msg=roi_err, is_valid=False
            )
            self.last_debug = {}
            return self.detection_result

        work = image_gray
        if self.params.get("use_binary"):
            work = cv.inRange(
                image_gray,
                int(self.params["min_threshold"]),
                int(self.params["max_threshold"]),
            )

        lp_ratio = float(self.params.get("lp_ratio", 0.15))
        hp_ratio = float(self.params.get("hp_ratio", 0.0))
        order = int(self.params.get("filter_order", 4))
        raw_ratio = float(self.params.get("raw_dark_ratio", 0.6))
        max_minima = int(self.params.get("max_minima", 5))

        boundaries = {}
        debug = {}
        for roi_name, (roi_x, roi_y, roi_w, roi_h) in rois.items():
            roi_image = work[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            if roi_image.size == 0:
                self.detection_result = Size_Result(
                    error_code=2, error_msg=f"{roi_name}边ROI无效", is_valid=False
                )
                self.last_debug = debug
                return self.detection_result

            is_horizontal = roi_name in ("top", "bottom")
            raw = np.asarray(
                calculate_projection_curve(
                    roi_image, "horizontal" if is_horizontal else "vertical"
                ),
                dtype=np.float64,
            )
            filtered = fft_band_filter_curve(
                raw, lp_ratio=lp_ratio, hp_ratio=hp_ratio, order=order
            )
            boundary_offset = select_boundary_from_curves(
                raw,
                filtered,
                raw_ratio=raw_ratio,
                max_candidates=max_minima,
            )
            debug[roi_name] = {
                "roi": (roi_x, roi_y, roi_w, roi_h),
                "raw": raw,
                "filtered": filtered,
                "offset": boundary_offset,
                "is_horizontal": is_horizontal,
            }
            if boundary_offset is None:
                self.detection_result = Size_Result(
                    error_code=2,
                    error_msg=f"无法检测到{roi_name}边边界",
                    is_valid=False,
                )
                self.last_debug = debug
                return self.detection_result
            if is_horizontal:
                boundaries[roi_name] = float(roi_y + boundary_offset)
            else:
                boundaries[roi_name] = float(roi_x + boundary_offset)

        top_b, bottom_b = boundaries["top"], boundaries["bottom"]
        left_b, right_b = boundaries["left"], boundaries["right"]
        if right_b <= left_b or bottom_b <= top_b:
            self.detection_result = Size_Result(
                error_code=2, error_msg="尺寸边界无效", is_valid=False
            )
            self.last_debug = debug
            return self.detection_result

        width_pixel = right_b - left_b
        height_pixel = bottom_b - top_b
        pixel_size = float(self.params.get("pixel_size", 0.001))
        ps_x = self.params.get("pixel_size_x")
        width_scale = pixel_size if ps_x is None else float(ps_x)
        product_width_mm = width_pixel * width_scale
        product_height_mm = height_pixel * pixel_size

        is_valid = True
        std_width, std_height = self.params.get("std_size", (0.0, 0.0))
        if std_width > 0 and std_height > 0:
            is_valid = (
                abs(std_width - product_width_mm)
                <= float(self.params.get("allow_tolerance_x", 0.0))
                and abs(std_height - product_height_mm)
                <= float(self.params.get("allow_tolerance_y", 0.0))
            )

        result = Size_Result(
            width=product_width_mm,
            height=product_height_mm,
            box_points=[float(left_b), float(top_b), float(width_pixel), float(height_pixel)],
            is_valid=is_valid,
        )
        self.detection_result = result
        self.last_debug = debug
        return result

    def _validate_rois(self, img_w, img_h):
        rois = self.params.get("rois") or {}
        out = {}
        for side in SizeDetector.ROI_SIDES:
            roi = SizeDetector._normalize_roi(rois.get(side))
            if roi is None:
                return None, "未框选完整的四个 ROI"
            clamped = SizeDetector._clamp_roi_to_image(roi, img_w, img_h)
            if clamped is None:
                return None, "未框选完整的四个 ROI"
            out[side] = clamped
        return out, None


# ==================== 可视化 ====================

def _draw_curve(canvas, curve, color, x0, y0, cw, ch):
    if curve is None or len(curve) < 2 or cw < 2 or ch < 2:
        return
    c = np.asarray(curve, dtype=np.float64)
    ymin, ymax = float(np.min(c)), float(np.max(c))
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0
    xs = np.linspace(0, cw - 1, len(c))
    ys = y0 + (ch - 1) - (c - ymin) / (ymax - ymin) * (ch - 1)
    pts = np.column_stack([x0 + xs, ys]).astype(np.int32)
    cv.polylines(canvas, [pts], False, color, 1, cv.LINE_AA)


def render_curve_panel(debug: dict, width=960, height=720) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    sides = ("top", "left", "bottom", "right")
    cols, rows = 2, 2
    margin = 12
    cell_w = (width - margin * (cols + 1)) // cols
    cell_h = (height - margin * (rows + 1)) // rows
    for i, side in enumerate(sides):
        r, c = divmod(i, cols)
        x0 = margin + c * (cell_w + margin)
        y0 = margin + r * (cell_h + margin)
        cv.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (60, 60, 60), 1)
        info = debug.get(side) or {}
        raw = info.get("raw")
        filtered = info.get("filtered")
        offset = info.get("offset")
        plot_x, plot_y = x0 + 8, y0 + 28
        plot_w, plot_h = cell_w - 16, cell_h - 40
        _draw_curve(canvas, raw, (160, 160, 160), plot_x, plot_y, plot_w, plot_h)
        _draw_curve(canvas, filtered, (0, 220, 255), plot_x, plot_y, plot_w, plot_h)
        if offset is not None and filtered is not None and len(filtered) > 1:
            px = int(plot_x + float(offset) / (len(filtered) - 1) * (plot_w - 1))
            cv.line(canvas, (px, plot_y), (px, plot_y + plot_h - 1), (0, 180, 0), 1)
        label = f"{side}  off={offset:.2f}" if offset is not None else f"{side}  miss"
        cv.putText(
            canvas,
            label,
            (x0 + 8, y0 + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv.LINE_AA,
        )
    cv.putText(
        canvas,
        "gray=raw  cyan=fft-filtered  green=boundary",
        (margin, height - 8),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv.LINE_AA,
    )
    return canvas


def _put_hud_text(img, text, org, color, scale=0.85, thickness=2):
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv.putText(
            img,
            text,
            (x + dx, y + dy),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv.LINE_AA,
        )
    cv.putText(
        img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA
    )


def overlay_rois_and_boundaries(image_bgr, debug: dict):
    vis = image_bgr.copy()
    for side, info in (debug or {}).items():
        roi = info.get("roi")
        if not roi:
            continue
        x, y, w, h = [int(v) for v in roi]
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 1)
        offset = info.get("offset")
        if offset is None:
            continue
        if info.get("is_horizontal"):
            gy = int(round(y + float(offset)))
            cv.line(vis, (x, gy), (x + w, gy), (0, 255, 255), 1)
        else:
            gx = int(round(x + float(offset)))
            cv.line(vis, (gx, y), (gx, y + h), (0, 255, 255), 1)
    return vis


# ==================== pytest：合成图表征 ====================

def test_select_lowest_local_min_when_raw_dark():
    filtered = np.array([8, 6, 9, 4, 7, 5, 8], dtype=np.float64)
    # 局部极小：idx1=6, idx3=4（最低）, idx5=5
    raw = np.array([10, 10, 10, 1, 10, 10, 10], dtype=np.float64)
    # mean=8.714..., 0.6*mean≈5.23；raw[3]=1 过门限
    assert select_boundary_from_curves(raw, filtered) == 3.0


def test_select_second_lowest_local_min_when_lowest_raw_bright():
    filtered = np.array([8, 3, 9, 5, 8], dtype=np.float64)
    # 局部极小：idx1=3（最低）, idx3=5（第二）
    raw = np.array([10, 10, 10, 1, 10], dtype=np.float64)
    # mean=8.2, 0.6*mean=4.92；raw[1]=10 不过，raw[3]=1 过
    assert select_boundary_from_curves(raw, filtered) == 3.0


def test_select_falls_back_to_filtered_global_min():
    filtered = np.array([8, 2, 9, 4, 7, 5, 8], dtype=np.float64)
    # 局部极小 idx1=2, idx3=4, idx5=5；全部 raw 都亮
    raw = np.full(7, 10.0)
    assert select_boundary_from_curves(raw, filtered) == 1.0


def test_select_caps_at_five_local_minima():
    # 6 个局部极小；仅最浅的第 6 个 raw 够暗。只看最低的 5 个谷，应忽略第 6 个并回退全局最低。
    filtered = np.array(
        [5, 4, 5, 3.5, 5, 3, 5, 2.5, 5, 2, 5, 1, 5], dtype=np.float64
    )
    raw = np.full(13, 10.0)
    raw[1] = 1.0
    assert select_boundary_from_curves(raw, filtered) == float(np.argmin(filtered))


def test_fft_filter_preserves_length():
    curve = np.linspace(0, 255, 137, dtype=np.float64)
    out = fft_band_filter_curve(curve, lp_ratio=0.2, hp_ratio=0.0, order=4)
    assert out.shape == curve.shape


def test_fft_lowpass_attenuates_high_frequency():
    n = 256
    t = np.arange(n, dtype=np.float64)
    low = np.sin(2 * np.pi * t / n)
    high = 0.5 * np.sin(2 * np.pi * t * 40 / n)
    mixed = low + high
    filtered = fft_band_filter_curve(mixed, lp_ratio=0.08, hp_ratio=0.0, order=4)
    err_to_low = float(np.mean((filtered - low) ** 2))
    err_to_mixed = float(np.mean((filtered - mixed) ** 2))
    assert err_to_low < err_to_mixed


def test_fft_zero_phase_keeps_step_center():
    curve = np.zeros(101, dtype=np.float64)
    curve[50:] = 200.0
    filtered = fft_band_filter_curve(curve, lp_ratio=0.2, hp_ratio=0.0, order=4)
    # 零相位低通后，50% 过零应仍在阶跃附近
    target = 0.5 * (filtered[0] + filtered[-1])
    idx = int(np.argmin(np.abs(filtered - target)))
    assert abs(idx - 50) <= 2


def test_fft_size_detect_synthetic_white_rect():
    h, w = 200, 300
    image = np.zeros((h, w), dtype=np.uint8)
    image[40:160, 50:250] = 200
    det = FftSizeDetector()
    det.update_params(
        {
            "rois": SizeDetector.default_rois(w, h, strip=50),
            "pixel_size": 1.0,
            "std_size": (0.0, 0.0),
            "lp_ratio": 0.2,
            "hp_ratio": 0.0,
            "use_binary": False,
            "detect_direction": "outward",
        }
    )
    result = det.detect(image)
    assert result.error_code == 0
    x, y, bw, bh = result.box_points
    assert bw > 0 and bh > 0
    assert x + bw > x
    assert y + bh > y


def test_fft_missing_roi_fails():
    det = FftSizeDetector()
    image = np.zeros((80, 80), dtype=np.uint8)
    result = det.detect(image)
    assert result.error_code == 2


# ==================== OpenCV 调试 UI ====================

PANEL_NAME = "fft_panel"
BINARY_NAME = "binary"
RESULT_NAME = "result"
CURVE_NAME = "curves"
_WINDOWS_READY = False


def _noop(_):
    return


def init_windows():
    global _WINDOWS_READY
    if _WINDOWS_READY:
        return
    cv.namedWindow(PANEL_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(PANEL_NAME, 640, 220)
    cv.namedWindow(BINARY_NAME, cv.WINDOW_NORMAL)
    cv.namedWindow(RESULT_NAME, cv.WINDOW_NORMAL)
    cv.namedWindow(CURVE_NAME, cv.WINDOW_NORMAL)
    cv.createTrackbar("min_th", PANEL_NAME, 0, 255, _noop)
    cv.createTrackbar("max_th", PANEL_NAME, 255, 255, _noop)
    cv.createTrackbar("roi_w", PANEL_NAME, 30, 300, _noop)
    cv.createTrackbar("lp_pct", PANEL_NAME, 1, 100, _noop)
    cv.createTrackbar("hp_pct", PANEL_NAME, 0, 99, _noop)
    cv.createTrackbar("order", PANEL_NAME, 1, 8, _noop)
    cv.createTrackbar("binary", PANEL_NAME, 0, 1, _noop)
    cv.createTrackbar("inward", PANEL_NAME, 0, 1, _noop)
    _WINDOWS_READY = True


def run_fft_debug_ui(detector: FftSizeDetector, image_path: str):
    image_ori = cv.imread(image_path)
    if image_ori is None:
        print(f"无法读取图像: {image_path}")
        return "next"

    init_windows()
    cv.setTrackbarPos("min_th", PANEL_NAME, int(detector.params.get("min_threshold", 0)))
    cv.setTrackbarPos("max_th", PANEL_NAME, int(detector.params.get("max_threshold", 255)))
    roi_w0 = 150
    rois0 = detector.params.get("rois") or {}
    top_roi = SizeDetector._normalize_roi(rois0.get("top"))
    if top_roi is not None:
        roi_w0 = int(top_roi[3])
    cv.setTrackbarPos("roi_w", PANEL_NAME, int(np.clip(roi_w0, 30, 300)))
    cv.setTrackbarPos(
        "lp_pct", PANEL_NAME, int(round(float(detector.params.get("lp_ratio", 0.15)) * 100))
    )
    cv.setTrackbarPos(
        "hp_pct", PANEL_NAME, int(round(float(detector.params.get("hp_ratio", 0.0)) * 100))
    )
    cv.setTrackbarPos("order", PANEL_NAME, int(detector.params.get("filter_order", 4)))
    cv.setTrackbarPos("binary", PANEL_NAME, 1 if detector.params.get("use_binary") else 0)
    cv.setTrackbarPos(
        "inward",
        PANEL_NAME,
        1 if detector.params.get("detect_direction") == "inward" else 0,
    )

    while True:
        min_th = cv.getTrackbarPos("min_th", PANEL_NAME)
        max_th = cv.getTrackbarPos("max_th", PANEL_NAME)
        if min_th > max_th:
            max_th = min_th
            cv.setTrackbarPos("max_th", PANEL_NAME, max_th)
        roi_w = max(30, cv.getTrackbarPos("roi_w", PANEL_NAME))
        lp_pct = max(1, cv.getTrackbarPos("lp_pct", PANEL_NAME))
        hp_pct = cv.getTrackbarPos("hp_pct", PANEL_NAME)
        order = max(1, cv.getTrackbarPos("order", PANEL_NAME))
        use_binary = bool(cv.getTrackbarPos("binary", PANEL_NAME))
        inward = bool(cv.getTrackbarPos("inward", PANEL_NAME))

        gray = ensure_gray_u8(image_ori, copy=True)
        ih, iw = gray.shape
        detector.update_params(
            {
                "min_threshold": min_th,
                "max_threshold": max_th,
                "rois": SizeDetector.default_rois(iw, ih, strip=roi_w),
                "lp_ratio": lp_pct / 100.0,
                "hp_ratio": hp_pct / 100.0,
                "filter_order": order,
                "use_binary": use_binary,
                "detect_direction": "inward" if inward else "outward",
            },
            clear_result=False,
        )

        image_binary = cv.inRange(gray, min_th, max_th)
        size_result = detector.detect(image_ori)
        product = Product()
        product.size_result = size_result
        _, _, result_img = draw_detection_results(
            image_result=image_ori.copy(),
            product=product,
        )
        if result_img is None:
            result_img = image_ori.copy()
        result_img = overlay_rois_and_boundaries(result_img, detector.last_debug)

        width_mm = float(size_result.width or 0.0)
        height_mm = float(size_result.height or 0.0)
        if size_result.error_code != 0:
            status_text = f"FAIL:{size_result.error_msg or ''}"
            status_color = (0, 0, 255)
        else:
            status_text = "OK" if size_result.is_valid else "NG"
            status_color = (0, 255, 0) if size_result.is_valid else (0, 0, 255)

        lines = [
            (f"min={min_th} max={max_th} roi={roi_w} bin={int(use_binary)}", (0, 255, 255)),
            (
                f"lp={lp_pct}% hp={hp_pct}% order={order} dir={'inward' if inward else 'outward'}",
                (0, 255, 255),
            ),
            (f"W={width_mm:.3f} mm  H={height_mm:.3f} mm", (255, 255, 255)),
            (f"RESULT: {status_text}", status_color),
            ("ESC quit  E prev  Q next", (0, 255, 255)),
        ]
        for i, (text, color) in enumerate(lines):
            _put_hud_text(result_img, text, (16, 36 + i * 32), color)

        cv.imshow(BINARY_NAME, image_binary)
        cv.imshow(RESULT_NAME, result_img)
        cv.imshow(CURVE_NAME, render_curve_panel(detector.last_debug))

        key = cv.waitKey(30) & 0xFF
        if key == 27:
            return "quit"
        if key == ord("e"):
            return "prev"
        if key == ord("q"):
            return "next"


def _iter_image_paths():
    folder = _ROOT / "Image" / "Dry"
    if not folder.is_dir():
        return []
    names = sorted(os.listdir(folder))
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return [
        str(folder / name)
        for name in names
        if Path(name).suffix.lower() in exts
    ]


def main():
    detector = FftSizeDetector()
    detector.update_params(
        {
            "min_threshold": 100,
            "max_threshold": 200,
            "pixel_size": 0.008841,
            "allow_tolerance_x": 0.07,
            "allow_tolerance_y": 0.07,
            "std_size": (10.0, 15.0),
            "lp_ratio": 0.15,
            "hp_ratio": 0.0,
            "filter_order": 4,
            "use_binary": False,
            "detect_direction": "outward",
        }
    )
    image_paths = _iter_image_paths()
    if not image_paths:
        print(f"未找到图像: {_ROOT / 'Image' / 'Dry'}")
        return

    idx = 0
    while 0 <= idx < len(image_paths):
        action = run_fft_debug_ui(detector, image_paths[idx])
        if action == "quit":
            break
        if action == "prev":
            idx = max(0, idx - 1)
        else:
            idx += 1
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

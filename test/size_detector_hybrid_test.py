"""生产 SizeDetector（Dry 混合算法）的人工测试脚本。

管线已同步到 src/detectors/size_detector.py（algorithm=hybrid）：
  灰度 -> 外侧陡边投票 -> 局部 50% 交点 -> 固定像素修正 -> pixel_size 换算
定位不使用二值阈值。edge_bias_x / edge_bias_y 见 QUALITY_GATES。
Transfer 仍走 algorithm=legacy，本脚本只测 Dry 混合路径。

运行：
  python test/size_detector_hybrid_test.py [图像或目录]
  python test/size_detector_hybrid_test.py --no-ui
  python test/size_detector_hybrid_test.py --no-ui --report

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
from src.detectors.size_hybrid import (  # noqa: E402
    MIN_ROI_DEPTH,
    QUALITY_GATES,
    evaluate_edge_line,
)
from src.support.support_funs import ensure_gray_u8  # noqa: E402


# ==================== 可视化 ====================

CONTROL_WINDOW = "hybrid_controls"
BINARY_WINDOW = "hybrid_binary"
RESULT_WINDOW = "hybrid_result"
QUALITY_WINDOW = "hybrid_edge_quality"
_WINDOWS_READY = False


def _noop(_):
    return


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


def draw_detection_debug(image, detector, image_name=""):
    """绘制 ROI、边缘点、内点直线及最终尺寸框。"""
    canvas = cv.cvtColor(ensure_gray_u8(image, copy=True), cv.COLOR_GRAY2BGR)
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
        if "slope" in info:
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
        threshold = summary.get("threshold", ("?", "?"))
        worst = max(summary.get("extrapolation_px", {"-": 0.0}).values())
        lines = [
            f"{image_name}",
            f"W={result.width:.6f} mm  H={result.height:.6f} mm",
            f"STATUS={'OK' if result.is_valid else 'NG'}"
            f"  threshold=[{threshold[0]}, {threshold[1]}]"
            f"  dir={_direction_label(summary.get('detect_direction'))}"
            f"  extrap={worst:.0f}px",
        ]
    else:
        error_msg = "" if result is None else str(result.error_msg or "")
        lines = [f"{image_name}", "DETECT FAILED", error_msg[:80]]

    lines.append("Q next  E prev  R select ROI  D default ROI  S save  ESC quit")
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
    cv.polylines(
        canvas, [np.column_stack([xs, ys]).astype(np.int32)], False,
        color, 1, cv.LINE_AA,
    )


def render_quality_panel(debug, width=1000, height=680):
    """绘制四边代表扫描线、判据电平与质量指标。"""
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

        dark = representative["dark"]
        bright = representative["bright"]
        span = max(bright - dark, 1e-9)
        plot = (x0 + 8, y0 + 35, cell_width - 16, cell_height - 48)
        value_range = (dark - 0.25 * span, bright + 0.25 * span)
        _draw_curve(
            canvas, representative["window"], plot, (130, 130, 130), value_range
        )
        _draw_curve(
            canvas, representative["smoothed"], plot, (0, 220, 255), value_range
        )
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

        _put_hud_text(
            canvas,
            f"contrast={info.get('contrast_ratio', 0):.3f}  "
            f"edge_width={info.get('edge_width', 0):.2f}px  "
            f"bin_offset={info.get('binary_offset', 0):+.2f}px",
            (x0 + 8, y0 + cell_height - 8), (190, 190, 190), 0.42,
        )

    _put_hud_text(
        canvas,
        "gray=window  yellow=smoothed  orange=dark/bright  green=50% level",
        (margin, height - 4), (190, 190, 190), 0.42,
    )
    return canvas


def _direction_label(direction):
    if SizeDetector.normalize_detect_direction(direction) == "inward":
        return "inward(从外往内)"
    return "outward(从内到外)"


def render_control_help(width=700, height=230):
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    lines = [
        "Hybrid detector: outer steep-edge vote + local 50% crossing",
        "min_th / max_th are ignored by localization",
        "roi_strip: default ROI depth, min 20px; the edge must lie inside the strip",
        "edge_bias_x/y shift the 50% point onto the outline (px, + = inward)",
    ]
    for idx, line in enumerate(lines):
        _put_hud_text(panel, line, (12, 24 + idx * 26), (220, 220, 220), 0.46)
    return panel


def init_windows(args):
    global _WINDOWS_READY
    if _WINDOWS_READY:
        return
    cv.namedWindow(CONTROL_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(CONTROL_WINDOW, 700, 260)
    for name in (BINARY_WINDOW, RESULT_WINDOW, QUALITY_WINDOW):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.createTrackbar("min_th", CONTROL_WINDOW, int(args.min_threshold), 255, _noop)
    cv.createTrackbar("max_th", CONTROL_WINDOW, int(args.max_threshold), 255, _noop)
    strip_max = max(400, MIN_ROI_DEPTH, int(args.roi_strip))
    cv.createTrackbar(
        "roi_strip", CONTROL_WINDOW,
        max(MIN_ROI_DEPTH, int(args.roi_strip)), strip_max, _noop,
    )
    cv.createTrackbar(
        "direction", CONTROL_WINDOW,
        0 if SizeDetector.normalize_detect_direction(args.direction) == "outward" else 1,
        1, _noop,
    )
    _WINDOWS_READY = True


def read_control_params():
    min_th = cv.getTrackbarPos("min_th", CONTROL_WINDOW)
    max_th = cv.getTrackbarPos("max_th", CONTROL_WINDOW)
    if min_th > max_th:
        max_th = min_th
        cv.setTrackbarPos("max_th", CONTROL_WINDOW, max_th)
    roi_strip = cv.getTrackbarPos("roi_strip", CONTROL_WINDOW)
    if roi_strip < MIN_ROI_DEPTH:
        roi_strip = MIN_ROI_DEPTH
        cv.setTrackbarPos("roi_strip", CONTROL_WINDOW, roi_strip)
    direction = (
        "inward" if cv.getTrackbarPos("direction", CONTROL_WINDOW) else "outward"
    )
    params = {
        "min_threshold": min_th,
        "max_threshold": max_th,
        "detect_direction": direction,
        "algorithm": "hybrid",
    }
    return roi_strip, params, (min_th, max_th, roi_strip, direction)


def select_four_rois(image):
    """依次人工框选 top/left/bottom/right ROI；取消任一边则整体取消。"""
    display = cv.cvtColor(ensure_gray_u8(image, copy=True), cv.COLOR_GRAY2BGR)
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
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        result_path = save_dir / f"{stem}_hybrid_result.png"
        quality_path = save_dir / f"{stem}_hybrid_quality.png"
        ok_result = cv.imwrite(str(result_path), result_image)
        ok_quality = cv.imwrite(str(quality_path), quality_image)
    except (OSError, cv.error) as exc:
        print(f"保存调试图失败: {exc}")
        return False
    if not ok_result or not ok_quality:
        print(f"保存调试图失败: {save_dir}")
        return False
    print(f"已保存: {result_path}")
    print(f"已保存: {quality_path}")
    return True


def run_manual_ui(detector, image_path, args):
    image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return "next"

    init_windows(args)
    custom_rois = None
    last_signature = None
    result_image = cv.cvtColor(ensure_gray_u8(image, copy=True), cv.COLOR_GRAY2BGR)
    quality_image = np.full((680, 1000, 3), 25, dtype=np.uint8)

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
            result = detector.detect(image)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            result_image = draw_detection_debug(
                image, detector, Path(image_path).name
            )
            quality_image = render_quality_panel(detector.last_debug)
            binary_image = cv.inRange(
                ensure_gray_u8(image, copy=True),
                control_params["min_threshold"],
                control_params["max_threshold"],
            )
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
            cv.imshow(BINARY_WINDOW, binary_image)
            cv.imshow(RESULT_WINDOW, result_image)
            cv.imshow(QUALITY_WINDOW, quality_image)
            cv.imshow(CONTROL_WINDOW, render_control_help())

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


def build_detector(args):
    return SizeDetector(
        {
            "min_threshold": args.min_threshold,
            "max_threshold": args.max_threshold,
            "pixel_size": args.pixel_size,
            "pixel_size_x": args.pixel_size_x,
            "std_size": (args.std_width, args.std_height),
            "allow_tolerance_x": args.tolerance_x,
            "allow_tolerance_y": args.tolerance_y,
            "detect_direction": args.direction,
            "algorithm": "hybrid",
        }
    )


def run_without_ui(detector, image_paths, args):
    exit_code = 0
    success_count = 0
    elapsed_values = []
    print(
        f"algorithm=hybrid "
        f"bias_x={QUALITY_GATES['edge_bias_x']} "
        f"bias_y={QUALITY_GATES['edge_bias_y']} "
        f"detect_direction={detector.params.get('detect_direction', 'outward')} "
        f"({_direction_label(detector.params.get('detect_direction'))})"
    )
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
                f"{image_path.name}: FAIL - {result.error_msg} "
                f"time={elapsed_ms:.1f}ms"
            )
            exit_code = 1
        else:
            success_count += 1
            summary = detector.last_debug.get("summary", {})
            worst = max(summary.get("extrapolation_px", {"-": 0.0}).values())
            print(
                f"{image_path.name}: width={result.width:.6f}mm "
                f"height={result.height:.6f}mm valid={result.is_valid} "
                f"dir={summary.get('detect_direction', '?')} "
                f"extrap={worst:.0f}px time={elapsed_ms:.1f}ms"
            )
        if args.report and detector.last_debug:
            for side in SizeDetector.ROI_SIDES:
                info = detector.last_debug.get(side)
                if not info:
                    continue
                print(
                    f"    {side:6s} valid={info['valid']}/{info['attempted']} "
                    f"contrast={info.get('contrast_ratio', 0):.3f} "
                    f"width={info.get('edge_width', 0):.2f}px "
                    f"bin_offset={info.get('binary_offset', 0):+6.2f}px "
                    f"rmse={info.get('line_rmse', float('nan')):.4f} "
                    f"inlier={info.get('inlier_ratio', float('nan')):.3f}"
                )
        if args.save_debug:
            if not save_debug_images(
                args.save_dir, image_path,
                draw_detection_debug(image, detector, image_path.name),
                render_quality_panel(detector.last_debug),
            ):
                exit_code = 1

    total_ms = float(np.sum(elapsed_values))
    print(
        f"SUMMARY: success={success_count}/{len(image_paths)} "
        f"total={total_ms:.1f}ms "
        f"average={total_ms / max(1, len(elapsed_values)):.1f}ms/image"
    )
    return exit_code


def parse_args():
    parser = argparse.ArgumentParser(
        description="灰度+二值化混合尺寸检测候选算法测试脚本"
    )
    parser.add_argument(
        "input", nargs="?", default=str(_ROOT / "Image" / "Dry"),
        help="图像文件或图像目录",
    )
    parser.add_argument("--min-threshold", type=int, default=210)
    parser.add_argument("--max-threshold", type=int, default=255)
    parser.add_argument("--pixel-size", type=float, default=0.014)
    parser.add_argument("--pixel-size-x", type=float, default=None)
    parser.add_argument("--std-width", type=float, default=0.0)
    parser.add_argument("--std-height", type=float, default=0.0)
    parser.add_argument("--tolerance-x", type=float, default=0.0)
    parser.add_argument("--tolerance-y", type=float, default=0.0)
    parser.add_argument(
        "--direction", choices=("outward", "inward"), default="outward",
        help="outward=从内到外取第一道本体边; inward=从外往内取最外侧轮廓",
    )
    parser.add_argument(
        "--roi-strip", type=int, default=120,
        help=f"批量默认 ROI 深度，下限 {MIN_ROI_DEPTH}px，无上限；产线上 ROI 由人工框选",
    )
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--report", action="store_true", help="输出逐边质量明细")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument(
        "--save-dir",
        default=str(_ROOT / "test" / "output" / "size_detector_hybrid"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.min_threshold <= 255 or not 0 <= args.max_threshold <= 255:
        print("min_threshold / max_threshold 必须在 [0, 255] 范围内")
        return 2
    if args.min_threshold > args.max_threshold:
        print("min_threshold 不能大于 max_threshold")
        return 2
    if args.roi_strip < MIN_ROI_DEPTH:
        print(f"roi_strip 不能小于 {MIN_ROI_DEPTH}px")
        return 2

    image_paths = iter_image_paths(args.input)
    if not image_paths:
        print(f"未找到可测试图像: {args.input}")
        return 1

    detector = build_detector(args)
    if args.no_ui:
        return run_without_ui(detector, image_paths, args)

    index = 0
    while 0 <= index < len(image_paths):
        action = run_manual_ui(detector, image_paths[index], args)
        if action == "quit":
            break
        index = max(0, index - 1) if action == "previous" else index + 1
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

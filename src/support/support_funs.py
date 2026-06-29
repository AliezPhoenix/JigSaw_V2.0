from itertools import product
import re
from typing import Dict, List, Sequence, Union, Optional, Tuple

import numpy as np
import cv2 as cv
import datetime
from datetime import datetime as dt

# 满盘检测深度学习依赖（程序启动时预加载）
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as torch_transforms
from PIL import Image as PILImage

from src.support import data_structure


def convert_numpy_obj(obj):
    """
    将 numpy 对象转换为 Python 原生类型，用于序列化。

    Args:
        obj: 可能为 numpy 标量/数组、列表、字典等

    Returns:
        可 JSON/Excel 序列化的原生类型
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (tuple, list)):
        return [convert_numpy_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_numpy_obj(value) for key, value in obj.items()}
    return obj


def calculate_cpk(data_list, usl, lsl):
    """
    计算 CPK（过程能力指数）。
    Cpu = (USL - μ) / (3σ)，Cpl = (μ - LSL) / (3σ)；双侧均存在时返回 max(cpu, cpl)。

    Args:
        data_list: 数据列表
        usl: 上规格限
        lsl: 下规格限

    Returns:
        CPK 值（float）或 None（无法计算）
    """
    if not data_list or len(data_list) < 2:
        return None
    try:
        mean = np.mean(data_list)
        std = np.std(data_list, ddof=1)
        if std == 0:
            return None
        cpu = (usl - mean) / (3 * std) if usl is not None else None
        cpl = (mean - lsl) / (3 * std) if lsl is not None else None
        if cpu is not None and cpl is not None:
            return max(cpu, cpl)
        if cpu is not None:
            return cpu
        if cpl is not None:
            return cpl
        return None
    except Exception:
        return None


def sanitize_filename_part(s: str) -> str:
    """移除路径遍历危险字符，仅保留安全字符用于文件名（防止 Modbus 数据注入）"""
    s = str(s).strip()
    s = re.sub(r'[^\w\-.]', '_', s)  # 只保留字母数字下划线横线点
    return s[:64] if s else "unknown"


def ensure_gray_u8(image: np.ndarray, *, copy: bool = True) -> np.ndarray:
    """
    将常见格式的图像 ndarray 转为单通道灰度 (H, W)，uint8。
    支持 2D 灰度、(H,W,1)、BGR、BGRA；通道数异常时取第 0 平面。
    避免对 (H,W,1) 误用 COLOR_BGR2GRAY 导致 OpenCV scn/bad channel 报错。
    """
    if image is None:
        raise TypeError("ensure_gray_u8: image is None")
    if image.ndim == 2:
        return image.copy() if copy else np.ascontiguousarray(image)
    if image.ndim != 3:
        raise ValueError(f"ensure_gray_u8: unsupported ndim={image.ndim}, shape={getattr(image, 'shape', None)}")
    c = image.shape[2]
    if c == 1:
        g = np.squeeze(image, axis=2)
        return g.copy() if copy else np.ascontiguousarray(g)
    if c == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if c == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    g = np.ascontiguousarray(image[:, :, 0])
    return g.copy() if copy else g


def ensure_bgr_u8(image: np.ndarray, *, copy: bool = True) -> np.ndarray:
    """
    将输入转为 3 通道 BGR (H, W, 3)，供绘制、叠加与 Qt 显示。
    支持 2D 灰度、(H,W,1)、BGR、BGRA。
    """
    if image is None:
        raise TypeError("ensure_bgr_u8: image is None")
    if image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"ensure_bgr_u8: unsupported ndim={image.ndim}, shape={getattr(image, 'shape', None)}")
    c = image.shape[2]
    if c == 1:
        g = np.squeeze(image, axis=2)
        return cv.cvtColor(g, cv.COLOR_GRAY2BGR)
    if c == 3:
        return image.copy() if copy else np.ascontiguousarray(image)
    if c == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    return cv.cvtColor(np.ascontiguousarray(image[:, :, 0]), cv.COLOR_GRAY2BGR)


def normalize_rect(start_point, end_point):
    """将拖拽端点规范化为左上角 + 宽高。"""
    x1, y1 = start_point
    x2, y2 = end_point
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h


def calc_bounds(origin: int, length: int, parts: int):
    """
    生成整数边界序列，确保 [origin, origin + length] 被完整覆盖。
    余数优先分配给前面的块，避免像素丢失。
    """
    if parts <= 0:
        return [origin, origin + max(0, int(length))]
    base = int(length) // int(parts)
    rem = int(length) % int(parts)
    bounds = [int(origin)]
    current = int(origin)
    for idx in range(parts):
        step = base + (1 if idx < rem else 0)
        current += step
        bounds.append(current)
    return bounds


def split_roi_to_grid(x: int, y: int, w: int, h: int, grid_rows: int, grid_cols: int):
    """
    把总 ROI 按网格切分为 cell_roi 列表（原图坐标，行优先）。
    返回: List[(x, y, w, h)]。
    """
    if w <= 0 or h <= 0 or grid_rows <= 0 or grid_cols <= 0:
        return []
    col_bounds = calc_bounds(x, w, grid_cols)
    row_bounds = calc_bounds(y, h, grid_rows)
    cells = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            cx = col_bounds[c]
            cy = row_bounds[r]
            cw = col_bounds[c + 1] - col_bounds[c]
            ch = row_bounds[r + 1] - row_bounds[r]
            if cw > 0 and ch > 0:
                cells.append((int(cx), int(cy), int(cw), int(ch)))
    return cells


def build_pyramid_display(image: np.ndarray, max_display_edge: int = 2000):
    """
    构建用于交互显示的金字塔缩放图。
    返回: (display_img, scale_factor)，scale_factor 为 display->source 倍率（1/2/4...）。
    """
    display = image.copy()
    scale_factor = 1
    h, w = display.shape[:2]
    while max(h, w) > max_display_edge and min(h, w) >= 2:
        display = cv.pyrDown(display)
        scale_factor *= 2
        h, w = display.shape[:2]
    return display, scale_factor


def display_to_source_point(px: int, py: int, src_w: int, src_h: int, scale_factor: int):
    """将显示层坐标映射回原图坐标，并做边界裁剪。"""
    sx = int(round(px * scale_factor))
    sy = int(round(py * scale_factor))
    sx = max(0, min(src_w - 1, sx))
    sy = max(0, min(src_h - 1, sy))
    return sx, sy


def draw_grid_overlay(
    canvas: np.ndarray,
    start_point,
    end_point,
    grid_rows: int,
    grid_cols: int,
    rect_color,
    line_thickness: int,
    show_crosshair: bool,
):
    """在框选阶段绘制外框 + 网格线 + 可选十字线。"""
    x, y, w, h = normalize_rect(start_point, end_point)
    cv.rectangle(canvas, (x, y), (x + w, y + h), rect_color, line_thickness)

    if show_crosshair:
        center_x = x + w // 2
        center_y = y + h // 2
        img_h, img_w = canvas.shape[:2]
        cv.line(canvas, (center_x, 0), (center_x, img_h), rect_color, line_thickness)
        cv.line(canvas, (0, center_y), (img_w, center_y), rect_color, line_thickness)

    if w <= 0 or h <= 0 or grid_rows <= 0 or grid_cols <= 0:
        return

    col_bounds = calc_bounds(x, w, grid_cols)
    row_bounds = calc_bounds(y, h, grid_rows)
    for cx in col_bounds[1:-1]:
        cv.line(canvas, (cx, y), (cx, y + h), rect_color, 1)
    for cy in row_bounds[1:-1]:
        cv.line(canvas, (x, cy), (x + w, cy), rect_color, 1)


def select_roi_grid(
    window_name: str,
    image: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    *,
    show_crosshair: bool = True,
    rect_color=(0, 255, 255),
    line_thickness: int = 2,
    max_display_edge: int = 1500,
):
    """
    在原图上框选总 ROI，然后切成网格 cells（原图坐标，行优先）。
    交互按键：空格/回车确认，ESC取消。
    """
    if grid_rows <= 0 or grid_cols <= 0:
        return []

    source_image = ensure_bgr_u8(image)
    src_h, src_w = source_image.shape[:2]
    display_image, scale_factor = build_pyramid_display(
        source_image, max_display_edge=max_display_edge
    )
    dynamic_canvas = display_image.copy()

    drawing = False
    start_point = None
    end_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, dynamic_canvas
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            dynamic_canvas = display_image.copy()
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            end_point = (x, y)
            dynamic_canvas = display_image.copy()
            draw_grid_overlay(
                dynamic_canvas,
                start_point,
                end_point,
                grid_rows,
                grid_cols,
                rect_color,
                line_thickness,
                show_crosshair,
            )
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            dynamic_canvas = display_image.copy()
            draw_grid_overlay(
                dynamic_canvas,
                start_point,
                end_point,
                grid_rows,
                grid_cols,
                rect_color,
                line_thickness,
                show_crosshair,
            )

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, mouse_callback)

    while True:
        cv.imshow(window_name, dynamic_canvas)
        key = cv.waitKey(20) & 0xFF

        if key in (ord(" "), 13):
            if start_point and end_point:
                dx, dy, dw, dh = normalize_rect(start_point, end_point)
                cv.destroyWindow(window_name)
                if dw <= 0 or dh <= 0:
                    return []
                p1 = display_to_source_point(dx, dy, src_w, src_h, scale_factor)
                p2 = display_to_source_point(dx + dw, dy + dh, src_w, src_h, scale_factor)
                x, y, w, h = normalize_rect(p1, p2)
                return split_roi_to_grid(x, y, w, h, grid_rows, grid_cols)
            cv.destroyWindow(window_name)
            return []

        if key == 27:
            cv.destroyWindow(window_name)
            return []


def is_flat_roi(roi) -> bool:
    """判断是否为单个矩形 ROI：[x, y, w, h]。"""
    return (
        isinstance(roi, (list, tuple))
        and len(roi) >= 4
        and not isinstance(roi[0], (list, tuple, np.ndarray))
    )


def is_grid_roi(roi) -> bool:
    """判断是否为网格 ROI（二维或一维 cell 列表）。"""
    if not isinstance(roi, (list, tuple)) or len(roi) == 0:
        return False
    first = roi[0]
    if not isinstance(first, (list, tuple)):
        return False
    if len(first) >= 4 and not isinstance(first[0], (list, tuple, np.ndarray)):
        return True  # 一维 cell 列表
    if len(first) > 0 and isinstance(first[0], (list, tuple)):
        return True  # 二维网格
    return False


def _normalize_flat_roi_to_image(roi, img_w: int, img_h: int):
    """将单个 ROI 限制在图像范围内；无效返回 None。"""
    if not is_flat_roi(roi) or img_w <= 0 or img_h <= 0:
        return None
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]


def normalize_grid_search_roi(search_roi, img_w: int, img_h: int, grid_rows: int, grid_cols: int):
    """
    归一化网格 ROI，返回二维 `[rows][cols][x,y,w,h]`。
    仅接受网格结构，不做 flat->grid 自动迁移。
    """
    if img_w <= 0 or img_h <= 0 or grid_rows <= 0 or grid_cols <= 0:
        return None
    if not is_grid_roi(search_roi):
        return None

    cells = []
    # 二维: rows x cols x 4
    if (
        len(search_roi) == grid_rows
        and isinstance(search_roi[0], (list, tuple))
        and len(search_roi[0]) > 0
        and isinstance(search_roi[0][0], (list, tuple))
    ):
        for r in range(grid_rows):
            row = search_roi[r]
            if not isinstance(row, (list, tuple)) or len(row) != grid_cols:
                return None
            for c in range(grid_cols):
                normalized = _normalize_flat_roi_to_image(row[c], img_w, img_h)
                if normalized is None:
                    return None
                cells.append(normalized)
    else:
        # 一维: (rows*cols) x 4
        expected = grid_rows * grid_cols
        if len(search_roi) != expected:
            return None
        for cell in search_roi:
            normalized = _normalize_flat_roi_to_image(cell, img_w, img_h)
            if normalized is None:
                return None
            cells.append(normalized)

    out = []
    idx = 0
    for _ in range(grid_rows):
        row = []
        for _ in range(grid_cols):
            row.append(cells[idx])
            idx += 1
        out.append(row)
    return out


def get_bga_animation_grid_layout(rows: int, cols: int, margin: int = 2, canvas_h: int = 480, canvas_w: int = 150):
    """
    计算 BGA animation 网格布局参数，与 get_full_animation 保持一致。
    返回: (margin, block_height, block_width, img_h, img_w)
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    margin = max(0, int(margin))
    block_height = max(1, int(canvas_h) // rows)
    block_width = max(1, int(canvas_w) // cols)
    img_h = rows * block_height + (rows + 1) * margin
    img_w = cols * block_width + (cols + 1) * margin
    return margin, block_height, block_width, img_h, img_w


def build_bga_animation_cell_rects(rows: int, cols: int, margin: int = 2, canvas_h: int = 480, canvas_w: int = 150):
    """
    生成 animation 网格每个 cell 的像素矩形（行优先）。
    返回: List[(row, col, x, y, w, h)]
    """
    margin, block_h, block_w, _, _ = get_bga_animation_grid_layout(
        rows, cols, margin=margin, canvas_h=canvas_h, canvas_w=canvas_w
    )
    rects = []
    for r in range(max(1, int(rows))):
        for c in range(max(1, int(cols))):
            y = margin + r * (block_h + margin)
            x = margin + c * (block_w + margin)
            rects.append((r, c, x, y, block_w, block_h))
    return rects


def create_alternating_array(rows, cols, start_element, values=(0, 3)):
    """
    创建交替填充的数组
    
    Args:
        rows: 行数
        cols: 列数
        start_element: 起始元素（0或1），决定交替模式从哪个值开始
        values: 用于交替的两个值，默认为(0, 1)。例如：(0, 1), (1, 0), (2, 3)等
    
    Returns:
        一个rows x cols的数组，按照交替模式填充values中的两个值
    """
    start = int(start_element) % 2
    row_indices = np.arange(rows)[:, np.newaxis]
    col_indices = np.arange(cols)
    pattern = (row_indices + col_indices + start) % 2
    # 将0和1的模式映射到用户指定的两个值
    result = np.where(pattern == 0, values[0], values[1])
    return result


def calculate_write_positions(target_array, window_array):
    """
    计算所有可写入位置（使用不重叠滑动，步长等于窗口大小）
    按照蛇形轨迹：从左上到左下，向右移动一格然后往上，到达最上方之后，往右移动再往下
    
    参数:
        target_array: 目标数组（numpy数组）
        window_array: 窗口数组（numpy数组），要赋值的数据块
    
    返回:
        positions: 位置列表，每个元素为 (row_start, col_start, row_end, col_end, 
                 actual_row_end, actual_col_end, source_row_size, source_col_size)
    """
    target_rows, target_cols = target_array.shape
    window_rows, window_cols = window_array.shape
    
    # 强制使用不重叠滑动，步长等于窗口大小
    stride_row = window_rows
    stride_col = window_cols
    
    # 计算列位置
    col_positions = list(range(0, target_cols, stride_col))
    
    positions = []
    
    # 按列遍历，每列内按蛇形移动
    for col_idx, col_start in enumerate(col_positions):
        # 计算当前列的所有行位置
        row_positions_list = list(range(0, target_rows, stride_row))
        
        # 偶数索引列（0, 2, 4...）：从上到下
        # 奇数索引列（1, 3, 5...）：从下到上
        if col_idx % 2 == 0:
            # 从上到下
            row_positions_iter = row_positions_list
        else:
            # 从下到上（反转列表）
            row_positions_iter = reversed(row_positions_list)
        
        for row_start in row_positions_iter:
            # 计算窗口的结束位置
            row_end = row_start + window_rows
            col_end = col_start + window_cols
            
            # 计算实际可写入的区域（考虑边界）
            actual_row_end = min(row_end, target_rows)
            actual_col_end = min(col_end, target_cols)
            
            # 计算需要从 window_array 中取出的部分
            source_row_size = actual_row_end - row_start
            source_col_size = actual_col_end - col_start
            
            # 只记录有效的写入位置
            if source_row_size > 0 and source_col_size > 0:
                positions.append((
                    row_start, col_start, 
                    row_end, col_end,
                    actual_row_end, actual_col_end,
                    source_row_size, source_col_size
                ))
    
    return positions


def assign_matches_to_grid(
    matches,
    template_width: int,
    template_height: int,
    search_roi,
    image_hw: tuple,
    grid_rows: int,
    grid_cols: int,
):
    """
    按模板中心点将匹配落入均匀网格；同格多点时保留置信度最高者。

    Args:
        matches: [(x, y, conf), ...]，全图坐标
        search_roi: [x, y, w, h] 或 None，与模板检测分桶一致
        image_hw: (img_h, img_w)
        grid_rows / grid_cols: 当前视野行、列数（与 params current_row / current_col 一致）

    Returns:
        grid_rows × grid_cols，每格为 None 或 (x, y) 整数左上角坐标
    """
    img_h, img_w = image_hw
    if grid_rows <= 0 or grid_cols <= 0:
        return [[None] * max(1, grid_cols) for _ in range(max(1, grid_rows))]

    if search_roi and isinstance(search_roi, (list, tuple)) and len(search_roi) >= 4:
        rx, ry, rw, rh = (
            int(search_roi[0]),
            int(search_roi[1]),
            int(search_roi[2]),
            int(search_roi[3]),
        )
    else:
        rx, ry, rw, rh = 0, 0, img_w, img_h

    rw = max(1, rw)
    rh = max(1, rh)
    cell_w = rw / float(grid_cols)
    cell_h = rh / float(grid_rows)

    best = [[None] * grid_cols for _ in range(grid_rows)]

    for x, y, conf in matches:
        cx = float(x) + template_width / 2.0
        cy = float(y) + template_height / 2.0
        gc = int((cx - rx) / cell_w) if cell_w > 0 else 0
        gr = int((cy - ry) / cell_h) if cell_h > 0 else 0
        gc = max(0, min(grid_cols - 1, gc))
        gr = max(0, min(grid_rows - 1, gr))
        prev = best[gr][gc]
        if prev is None or conf > prev[2]:
            best[gr][gc] = (int(x), int(y), float(conf))

    out = [[None] * grid_cols for _ in range(grid_rows)]
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            if best[gr][gc] is not None:
                bx, by, _ = best[gr][gc]
                out[gr][gc] = (bx, by)
    return out


def mask_roi_regions(image_gray: np.ndarray, roi_blocks: list) -> np.ndarray:
    """
    屏蔽ROI区域
    
    Args:
        image_gray: 灰度图像
        roi_blocks: 需要屏蔽的区域列表，每个元素为 (x, y, w, h) 格式的元组或列表
    
    Returns:
        屏蔽后的灰度图像
    """
    # 确保数组是可写的，如果是只读数组则创建副本
    if not image_gray.flags.writeable:
        image_gray = image_gray.copy()
    
    for roi in roi_blocks:
        try:
            # 验证ROI格式：必须是 (x, y, w, h) 格式的元组或列表
            if isinstance(roi, (tuple, list)) and len(roi) >= 4:
                x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            else:
                continue
            
            # 边界检查：确保ROI坐标在图像范围内
            h_img, w_img = image_gray.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            
            # 屏蔽该区域：将像素值设置为0
            image_gray[y:y+h, x:x+w] = 0
        except (ValueError, IndexError, TypeError) as e:
            # 忽略无效的ROI，继续处理下一个
            continue
    
    return image_gray


def hex_to_string(hex_list):
    """
    将16进制列表转换为字符串
    Args:
        hex_list: 16进制列表
    Returns:
        s: 字符串
    """
    s = ''
    for reg in hex_list:
        if reg == 0:
            break  # 遇到0时停止处理
        # 提取高字节和低字节
        high_byte = (reg >> 8) & 0xFF
        low_byte = reg & 0xFF
        # 转换为ASCII字符（先低字节后高字节，忽略空字符）
        if low_byte != 0:
            s += chr(low_byte)
        if high_byte != 0:
            s += chr(high_byte)
    return s


def value_transmit(value,mode):
    value = flip_horizontal_in_place(value)
    value =np.array(value)
    if mode == 1 :
        return value.flatten()
    if mode == 2:
        return flip270(value).flatten()
    if mode == 3:
        return flip180(value).flatten()
    if mode == 4:
        return flip90(value).flatten()


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def flip270(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def flip90(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr

def flip_horizontal_in_place(arr):
    for row in arr:
        row = row[::-1]
    return arr

def selectROI(window_name, image, showCrosshair=True, fromCenter=False, 
              rect_color=(0, 255, 0), line_thickness=2):
    """
    自定义ROI选择函数，支持自定义颜色和线宽
    
    参数:
        window_name: str - 窗口名称
        image: np.ndarray - 输入图像
        showCrosshair: bool - 是否显示十字准线（默认True）
        fromCenter: bool - 是否从中心开始绘制（默认False，未实现）
        rect_color: tuple - 矩形框颜色，BGR格式，默认绿色(0, 255, 0)
        line_thickness: int - 矩形框线宽，默认2像素
    
    返回:
        tuple - (x, y, w, h) 格式的边界框，如果取消则返回(0, 0, 0, 0)
    """
    # 全局变量用于鼠标回调
    drawing = False
    start_point = None
    end_point = None
    current_image = image.copy()
    display_image = current_image.copy()
    
    # 确保图像是3通道的（用于绘制彩色矩形）
    display_image = ensure_bgr_u8(display_image, copy=True)
    current_image = display_image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, display_image, current_image
        
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            display_image = current_image.copy()
            
            # 绘制十字准线（初始时在鼠标位置）
            if showCrosshair:
                h, w = display_image.shape[:2]
                cv.line(display_image, (x, 0), (x, h), rect_color, line_thickness)
                cv.line(display_image, (0, y), (w, y), rect_color, line_thickness)
        
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                display_image = current_image.copy()
                end_point = (x, y)
                
                # 绘制矩形
                cv.rectangle(display_image, start_point, end_point, rect_color, line_thickness)
                
                # 绘制十字准线在矩形框的中心位置
                if showCrosshair:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    h, w = display_image.shape[:2]
                    cv.line(display_image, (center_x, 0), (center_x, h), rect_color, line_thickness)
                    cv.line(display_image, (0, center_y), (w, center_y), rect_color, line_thickness)
        
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            display_image = current_image.copy()
            cv.rectangle(display_image, start_point, end_point, rect_color, line_thickness)
            
            # 绘制十字准线在矩形框的中心位置
            if showCrosshair:
                x1, y1 = start_point
                x2, y2 = end_point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                h, w = display_image.shape[:2]
                cv.line(display_image, (center_x, 0), (center_x, h), rect_color, line_thickness)
                cv.line(display_image, (0, center_y), (w, center_y), rect_color, line_thickness)
    
    # 设置鼠标回调
    cv.setMouseCallback(window_name, mouse_callback)
    
    # 显示图像并等待用户操作
    while True:
        cv.namedWindow(window_name,cv.WINDOW_NORMAL)
        cv.imshow(window_name, display_image)
        key = cv.waitKey(1) & 0xFF
        
        # 空格键或回车键确认选择
        if key == ord(' ') or key == 13:  # 13是回车键
            if start_point and end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                cv.destroyWindow(window_name)
                return (x, y, w, h)
            else:
                cv.destroyWindow(window_name)
                return (0, 0, 0, 0)
        
        # ESC键取消
        elif key == 27:  # ESC键
            cv.destroyWindow(window_name)
            return (0, 0, 0, 0)



def draw_detection_results(image_result: np.ndarray, product: 'data_structure.Product',
                          mark_color: str = "red"):
    """
    在图像上绘制检测结果（通用方法），消费 data_structure.Product。

    Args:
        image_result: 结果图像（BGR格式或灰度图）
        product: data_structure.Product 实例，读取其 size_result/ball_result/
            mark_result/scratch_result/shift_result 数据类字段进行绘制
        mark_color: Mark绘制颜色，字符串类型。"green"表示绿色，"red"或其他值表示红色，默认"red"

    Returns:
        tuple: (success: bool, msg: str, image_result: np.ndarray)
            - success: 是否成功
            - msg: 错误消息（如果有），成功时为空字符串
            - image_result: 绘制后的图像（BGR格式），失败时返回None
    """
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    
    # 根据mark_color参数确定Mark绘制颜色
    if mark_color and mark_color.lower() == "green":
        mark_draw_color = COLOR_GREEN
    else:
        mark_draw_color = COLOR_RED
    
    # 收集错误信息
    error_messages = []
    
    # 边界检查辅助函数
    def check_image_bounds(img, x, y, width, height):
        """检查坐标和尺寸是否在图像范围内"""
        if img is None or len(img.shape) < 2:
            return False
        img_h, img_w = img.shape[:2]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return False
        if x + width > img_w or y + height > img_h:
            return False
        return True
    
    if image_result is None:
        return False, "输入图像为空", None
    
    # 确保图像是BGR格式
    try:
        image_result = ensure_bgr_u8(image_result, copy=True)
    except Exception as e:
        error_msg = f"图像格式转换错误: {e}"
        error_messages.append(error_msg)
        return False, error_msg, None
    
    img_h, img_w = image_result.shape[:2]

    size_result = product.size_result
    ball_result = product.ball_result
    mark_result = product.mark_result
    scratch_result = product.scratch_result
    shift_result = product.shift_result

    ####尺寸绘制####
    if size_result is not None:
        try:
            box_points = size_result.box_points or []
            is_valid = size_result.is_valid

            if len(box_points) >= 4:
                x, y, w, h = box_points[0], box_points[1], box_points[2], box_points[3]
                x, y, w, h = float(x), float(y), float(w), float(h)
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                # 边界检查
                x1 = max(0, min(x1, img_w-1))
                y1 = max(0, min(y1, img_h-1))
                x2 = max(0, min(x2, img_w-1))
                y2 = max(0, min(y2, img_h-1))
                cv.rectangle(image_result, (x1, y1), (x2, y2), 
                            COLOR_GREEN if is_valid else COLOR_RED,2)
        except Exception as e:
            error_messages.append(f"绘制尺寸结果错误: {e}")
    
    ####ball绘制####
    if ball_result is not None:
        try:
            for contour in ball_result.ball_contour:
                try:
                    x, y, w, h = cv.boundingRect(contour)
                    if check_image_bounds(image_result, x, y, w, h):
                        cv.rectangle(image_result, (x, y), (x+w, y+h), COLOR_GREEN, 2)
                except Exception as e:
                    error_messages.append(f"绘制OK球结果错误: {e}")

            for contour in ball_result.ng_ball_contour:
                try:
                    x, y, w, h = cv.boundingRect(contour)
                    if check_image_bounds(image_result, x, y, w, h):
                        cv.rectangle(image_result, (x, y), (x+w, y+h), COLOR_RED, 2)
                except Exception as e:
                    error_messages.append(f"绘制NG球结果错误: {e}")
        except Exception as e:
            error_messages.append(f"获取球检测结果错误: {e}")
    
    ####mark绘制####
    if mark_result is not None:
        try:
            contours = mark_result.mark_contour or []
            if contours:
                mark_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for contour in contours:
                    if contour is not None:
                        cv.drawContours(mark_mask, [contour], -1, 255, -1)
                if cv.countNonZero(mark_mask) > 0:
                    overlay = image_result.copy()
                    overlay[mark_mask > 0] = mark_draw_color
                    cv.addWeighted(overlay, 0.8, image_result, 0.2, 0, image_result)
        except Exception as e:
            error_messages.append(f"绘制Mark结果错误: {e}")
    
    ####scratch绘制####
    if scratch_result is not None:
        try:
            for contour in (scratch_result.scratch_contour or []):
                cv.drawContours(image_result, [contour], -1, COLOR_RED, 2)
        except Exception as e:
            error_messages.append(f"绘制划痕结果错误: {e}")
    
    ####shift绘制####
    if shift_result is not None:
        try:
            is_valid_shift = shift_result.is_valid
            ball_center = shift_result.ball_center
            size_center = shift_result.size_center
            
            # # 绘制球中心点（如果存在）
            # if ball_center is not None and len(ball_center) >= 2:
            #     ball_x, ball_y = int(ball_center[0]), int(ball_center[1])
            #     if 0 <= ball_x < img_w and 0 <= ball_y < img_h:
            #         cv.circle(image_result, (ball_x, ball_y), 5, 
            #                 COLOR_GREEN if is_valid_shift else COLOR_RED, -1)
            #         cv.circle(image_result, (ball_x, ball_y), 10, 
            #                 COLOR_GREEN if is_valid_shift else COLOR_RED, 2)
            
            # # 绘制尺寸中心点（如果存在）
            # if size_center is not None and len(size_center) >= 2:
            #     size_x, size_y = int(size_center[0]), int(size_center[1])
            #     if 0 <= size_x < img_w and 0 <= size_y < img_h:
            #         cv.circle(image_result, (size_x, size_y), 5, COLOR_BLUE, -1)
            #         cv.circle(image_result, (size_x, size_y), 10, COLOR_BLUE, 2)
            
            # 绘制偏移向量（从尺寸中心到球中心）
            if (ball_center is not None and size_center is not None and 
                len(ball_center) >= 2 and len(size_center) >= 2):
                ball_x, ball_y = int(ball_center[0]), int(ball_center[1])
                size_x, size_y = int(size_center[0]), int(size_center[1])
                if (0 <= ball_x < img_w and 0 <= ball_y < img_h and 
                    0 <= size_x < img_w and 0 <= size_y < img_h):
                    cv.arrowedLine(image_result, (size_x, size_y), (ball_x, ball_y), 
                                 COLOR_GREEN if is_valid_shift else COLOR_RED, 2, tipLength=5)
                    cv.line(image_result, (size_x, size_y), (ball_x, ball_y), COLOR_GREEN if is_valid_shift else COLOR_RED, 2,cv.LINE_8)
        except Exception as e:
            error_messages.append(f"绘制shift结果错误: {e}")
    
    # 汇总错误信息
    if error_messages:
        msg = "; ".join(error_messages)
        return True, msg, image_result  # 有警告但图像已绘制，返回成功但带警告信息
    else:
        return True, "", image_result  # 完全成功


def execute_product_detection(
    image: np.ndarray,
    detectors: dict,
    params: dict,
    detect_type: str = None,
    early_return_on_ng: bool = False,
    error_callback = None
) -> tuple:
    """
    执行产品检测的通用函数
    
    Args:
        image: 输入的灰度图像或BGR图像
        detectors: 检测器字典，包含以下键：
            - "ball_detector": BallDetector实例
            - "size_detector": SizeDetector实例
            - "mark_detector": MarkDetector实例
            - "shift_detector": ShiftDetector实例
            - "scratch_detector": ScratchDetector实例
        params: 参数字典，包含以下键：
            - "mark_check_enable": bool 是否启用Mark检测
            - "allow_mark": bool 是否允许Mark（默认False）
                - True: 检测到Mark判定为OK，未检测到Mark判定为NG
                - False: 检测到Mark判定为NG，未检测到Mark判定为OK（默认行为）
            - "size_check_enable": bool 是否启用尺寸检测
            - "ball_check_enable": bool 是否启用锡球检测
            - "shift_check_enable": bool 是否启用偏移检测
            - "scratch_check_enable": bool 是否启用划痕检测
            - "roi_block"/"roi_blocks"/"roi-block": list 屏蔽区域列表，每项为 (x, y, w, h)
        detect_type: 检测类型，可选值：
            - None 或 "all": 执行所有启用的检测
            - "mark": 仅执行Mark检测
            - "size": 仅执行尺寸检测
            - "ball": 仅执行锡球检测
            - "shift": 仅执行偏移检测
            - "scratch": 仅执行划痕检测
        early_return_on_ng: bool 是否在检测到NG时提前返回
        error_callback: 可选的错误回调函数，接收 (error_msg: str) 参数
    
    Returns:
        tuple: (success: bool, msg: str, product: data_structure.Product)
            - success: bool 是否成功执行（False 表示某检测器返回 error_code != 0）
            - msg: str 消息（错误信息或成功信息）
            - product: data_structure.Product 产品数据类，字段包含：
                - defect_type: list 缺陷类型列表，如 ["OK"] 或 ["NG", "Size"]
                - product_image_result: np.ndarray 产品图像（原始图像副本，调用方可覆盖为绘制结果）
                - size_result / ball_result / mark_result / shift_result / scratch_result:
                  对应的检测结果数据类（未执行的检测保持默认空实例）
    """
    # 确定要执行的检测类型
    if detect_type is None or detect_type == "all":
        mark_check_enable = params.get("mark_check_enable", False)
        size_check_enable = params.get("size_check_enable", False)
        ball_check_enable = params.get("ball_check_enable", False)
        shift_check_enable = params.get("shift_check_enable", False)
        scratch_check_enable = params.get("scratch_check_enable", False)
    else:
        mark_check_enable = (detect_type == "mark")
        size_check_enable = (detect_type == "size")
        ball_check_enable = (detect_type == "ball")
        shift_check_enable = (detect_type == "shift")
        scratch_check_enable = (detect_type == "scratch")

    # 初始化产品数据结构（单一数据源）
    product = data_structure.Product()
    defect_type = ["OK"]
    product.product_image_result = image

    def _fail(msg: str):
        if error_callback:
            error_callback(msg)
        product.defect_type = defect_type
        product.product_image_result = image.copy()
        return False, msg, product

    def _mark_ng():
        if "OK" in defect_type:
            defect_type.remove("OK")

    ball_ran = False
    size_ran = False

    # 解析屏蔽区域参数，兼容不同键名与单区域格式
    raw_roi_blocks = (
        params.get("roi_block")if params.get("roi_block") is not None
        else (params.get("roi_blocks")if params.get("roi_blocks") is not None else params.get("roi-block"))
    )
    roi_blocks = []
    if isinstance(raw_roi_blocks, (list, tuple)):
        if len(raw_roi_blocks) >= 4 and not isinstance(raw_roi_blocks[0], (list, tuple, np.ndarray)):
            roi_blocks = [raw_roi_blocks]
        else:
            roi_blocks = list(raw_roi_blocks)

    # 对图像类检测统一使用屏蔽后的图像；无屏蔽时保留原逻辑
    if roi_blocks:
        image_for_detection = ensure_gray_u8(image, copy=True)
        image_for_detection = mask_roi_regions(image_for_detection, roi_blocks)
    else:
        image_for_detection = image

    # Mark检测
    if mark_check_enable:
        mark_detector = detectors.get("mark_detector")
        if mark_detector is None:
            return _fail("mark_detector 未提供")
        mark_result = mark_detector.detect(image_for_detection)
        if mark_result.error_code != 0:
            return _fail(f"Mark检测失败: {mark_result.error_msg}")
        product.mark_result = mark_result

        allow_mark = params.get("allow_mark", False)
        is_valid = mark_result.is_valid
        # allow_mark==True: 有Mark为OK，无Mark为NG；allow_mark==False: 有Mark为NG
        if (allow_mark and not is_valid) or ((not allow_mark) and is_valid):
            _mark_ng()
            defect_type.append("Mark")
            if early_return_on_ng:
                product.defect_type = defect_type
                product.product_image_result = image.copy()
                return True, "成功", product

    # 尺寸检测
    if size_check_enable:
        size_detector = detectors.get("size_detector")
        if size_detector is None:
            return _fail("size_detector 未提供")
        size_result = size_detector.detect(image_for_detection)
        if size_result.error_code != 0:
            return _fail(f"Size检测失败: {size_result.error_msg}")
        product.size_result = size_result
        size_ran = True
        if not size_result.is_valid:
            _mark_ng()
            defect_type.append("Size")
            if early_return_on_ng:
                product.defect_type = defect_type
                product.product_image_result = image.copy()
                return True, "成功", product

    # 锡球检测
    if ball_check_enable:
        ball_detector = detectors.get("ball_detector")
        if ball_detector is None:
            return _fail("ball_detector 未提供")
        ball_result = ball_detector.detect(image_for_detection)
        if ball_result.error_code != 0:
            return _fail(f"Ball检测失败: {ball_result.error_msg}")
        product.ball_result = ball_result
        ball_ran = True
        if not ball_result.is_valid:
            _mark_ng()
            # 数量问题 vs 面积问题
            expected_count = ball_detector.params.get("expected_ball_count", 0)
            if ball_result.ball_count != expected_count:
                defect_type.append("Ball Count")
            else:
                defect_type.append("Ball_Area")
            if early_return_on_ng:
                product.defect_type = defect_type
                product.product_image_result = image.copy()
                return True, "成功", product

    # 偏移检测（需要 ball 与 size 检测均已执行）
    if shift_check_enable and ball_ran and size_ran:
        shift_detector = detectors.get("shift_detector")
        if shift_detector is None:
            return _fail("shift_detector 未提供")
        shift_result = shift_detector.detect(product.ball_result, product.size_result)
        if shift_result.error_code != 0:
            return _fail(f"Shift检测失败: {shift_result.error_msg}")
        product.shift_result = shift_result
        if not shift_result.is_valid:
            _mark_ng()
            defect_type.append("Shift")
            if early_return_on_ng:
                product.defect_type = defect_type
                product.product_image_result = image.copy()
                return True, "成功", product

    # 划痕检测
    if scratch_check_enable:
        scratch_detector = detectors.get("scratch_detector")
        if scratch_detector is None:
            return _fail("scratch_detector 未提供")
        scratch_result = scratch_detector.detect(image_for_detection)
        if scratch_result.error_code != 0:
            return _fail(f"Scratch检测失败: {scratch_result.error_msg}")
        product.scratch_result = scratch_result
        if not scratch_result.is_valid:
            _mark_ng()
            defect_type.append("Scratch")
            if early_return_on_ng:
                product.defect_type = defect_type
                product.product_image_result = image.copy()
                return True, "成功", product

    # 更新总体判定
    if len(defect_type) > 1:
        defect_type[0] = "NG"
    product.defect_type = defect_type
    product.product_image_result = image.copy()

    return True, "成功", product


def parse_product_bbox(product_position) -> Optional[Tuple[int, int, int, int]]:
    """解析 product_position [x, y, w, h]，无效时返回 None。"""
    if not product_position or len(product_position) < 4:
        return None
    x, y, w, h = (int(product_position[i]) for i in range(4))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def crop_product_context_region(frame, x, y, w, h):
    """在整帧上按 product 中心扩展 3w×3h 裁剪（越界 clip）。返回 (crop, prod_x, prod_y) 或 None。"""
    frame = ensure_bgr_u8(frame, copy=False)
    ih, iw = frame.shape[:2]
    sx0, sy0 = max(0, x - w), max(0, y - h)
    sx1, sy1 = min(iw, x + 2 * w), min(ih, y + 2 * h)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    return frame[sy0:sy1, sx0:sx1].copy(), x - sx0, y - sy0


def overlay_bgr_patch(base, patch, origin_x, origin_y):
    """将 patch 贴到 base 的 (origin_x, origin_y)，返回新图像。"""
    if patch is None:
        return base
    display = base.copy()
    patch = ensure_bgr_u8(patch, copy=False)
    rh, rw = patch.shape[:2]
    if rh <= 0 or rw <= 0:
        return display
    dh, dw = display.shape[:2]
    x1, y1 = max(0, origin_x), max(0, origin_y)
    x2, y2 = min(dw, origin_x + rw), min(dh, origin_y + rh)
    if x2 <= x1 or y2 <= y1:
        return display
    sx, sy = x1 - origin_x, y1 - origin_y
    display[y1:y2, x1:x2] = patch[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
    return display


def resolve_product_overlay_patch(product) -> Optional[np.ndarray]:
    """取 product_image_result；缺失时按检测结果现场绘制。"""
    if product.product_image_result is not None:
        return product.product_image_result
    if product.product_image is None:
        return None
    mark_color = "green" if product.defect_type == ["OK"] else "red"
    _, _, patch = draw_detection_results(
        ensure_bgr_u8(product.product_image, copy=True),
        product,
        mark_color=mark_color,
    )
    return patch


# ==================== 满盘检测深度学习模型相关函数 ====================

class MobileNetV2ClassifierFulltray(nn.Module):
    """MobileNetV2满盘分类器"""

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def fulltray_get_transforms(is_train=False, input_size=150):
    """满盘检测数据变换"""
    return torch_transforms.Compose([
        torch_transforms.Resize((input_size, input_size)),
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    ])


def fulltray_load_model(model_path, device='cpu'):
    """
    加载满盘检测模型

    Args:
        model_path: 模型文件路径
        device: 设备 ('cpu' 或 'cuda')

    Returns:
        加载后的模型
    """
    model = MobileNetV2ClassifierFulltray(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    return model


def fulltray_predict_single_image(model, image_input, device='cpu', input_size=150):
    """
    满盘检测单张图像预测

    Args:
        model: 加载的模型
        image_input: 图像路径（str）或图像数组（np.ndarray）
        device: 设备 ('cpu' 或 'cuda')
        input_size: 输入图像尺寸

    Returns:
        tuple: (prediction, confidence)
            - prediction: 0=无产品, 1=有产品
            - confidence: 置信度 (0-1)
    """
    if isinstance(image_input, str):
        img = cv.imread(image_input, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = ensure_gray_u8(image_input, copy=True)
    else:
        raise ValueError(f"不支持的图像输入类型: {type(image_input)}")

    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    transform = fulltray_get_transforms(is_train=False, input_size=input_size)
    img_pil = PILImage.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item()


_SECTOR_REMAP_ORDER = ("sector_1", "sector_2", "sector_3")
_SECTOR_TARGET_CODE = {"sector_1": 3, "sector_2": 4, "sector_3": 5}
_SECTOR_MASK_BITS = 15
# 掩码 bit0/1/2 不参与重映射（类型编号 0、1、2 保持不变）
_SECTOR_SKIP_MASK_BITS = frozenset((0, 1, 2))


def map_product_type_to_sector(
    product_list: Union[Sequence[int], np.ndarray],
    ng_sector_dict: Dict[str, int],
) -> Union[List[int], np.ndarray]:
    """
    按 PLC 分选区掩码将一维产品类型编号重映射为分选目标码。

    Args:
        product_list: 一维序列（或 ``np.ndarray``），元素为整型类型编号；**bit k 与类型编号 k 对应**（k=0..14）。
            不修改传入对象，返回新序列。
        ng_sector_dict: 例如 ``{"sector_1": int, "sector_2": int, "sector_3": int}``，值为 PLC 读出的整型。

    Returns:
        ``list`` 或 ``np.ndarray``：对每个 sector，将 ``value & 0x7FFF`` 作为 15 位掩码；
        若 **bit k 为 1**（且 **k 不为 0、1、2**），则将所有当前值等于 **k** 的元素改为该 sector 的目标码：
        ``sector_1 -> 3``，``sector_2 -> 4``，``sector_3 -> 5``。
        类型编号 **0、1、2** 不因掩码参与重映射。
        按 ``sector_1`` → ``sector_2`` → ``sector_3`` 顺序应用。
    """
    if ng_sector_dict is None:
        ng_sector_dict = {}

    if isinstance(product_list, np.ndarray):
        out = np.copy(np.asarray(product_list))
        flat = out.reshape(-1)
        for name in _SECTOR_REMAP_ORDER:
            if name not in ng_sector_dict:
                continue
            target = _SECTOR_TARGET_CODE.get(name)
            if target is None:
                continue
            mask_val = int(ng_sector_dict[name]) & ((1 << _SECTOR_MASK_BITS) - 1)
            for k in range(_SECTOR_MASK_BITS):
                if k in _SECTOR_SKIP_MASK_BITS:
                    continue
                if (mask_val >> k) & 1:
                    flat[flat == k] = target
        return out

    out_list = [int(x) for x in product_list]
    for name in _SECTOR_REMAP_ORDER:
        if name not in ng_sector_dict:
            continue
        target = _SECTOR_TARGET_CODE.get(name)
        if target is None:
            continue
        mask_val = int(ng_sector_dict[name]) & ((1 << _SECTOR_MASK_BITS) - 1)
        for k in range(_SECTOR_MASK_BITS):
            if k in _SECTOR_SKIP_MASK_BITS:
                continue
            if (mask_val >> k) & 1:
                out_list = [target if x == k else x for x in out_list]
    return out_list

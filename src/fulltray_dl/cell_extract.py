"""从满盘 ROI 区域切分 cell 图像。"""

import os
from pathlib import Path

import cv2 as cv
import numpy as np
import time

from src.support.support_funs import ensure_gray_u8

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def normalize_search_roi_to_image(roi, img_w: int, img_h: int):
    """与 main_window._normalize_search_roi_to_image 等价。"""
    if img_w <= 0 or img_h <= 0:
        return None
    if not isinstance(roi, (list, tuple)) or len(roi) < 4:
        return [0, 0, img_w, img_h]
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    if w <= 0 or h <= 0:
        return [0, 0, img_w, img_h]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]


def extract_cells_from_roi(
    image: np.ndarray,
    roi,
    rows: int,
    cols: int,
    input_size: int,
    output_dir: str,
    image_stem: str = "fulltray",
) -> int:
    """
    按网格切分 ROI 内 cell，resize 后保存为 bmp。

    Returns:
        成功保存的 cell 数量。
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("图像无效，无法切分 cell")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows/cols 必须为正整数，当前 rows={rows}, cols={cols}")
    if input_size <= 0:
        raise ValueError(f"input_size 必须为正整数，当前 input_size={input_size}")

    gray = ensure_gray_u8(image, copy=True)
    img_h, img_w = gray.shape[:2]
    norm_roi = normalize_search_roi_to_image(roi, img_w, img_h)
    if norm_roi is None:
        raise ValueError("无法根据图像尺寸规范化 ROI")
    x, y, w, h = norm_roi
    roi_image = gray[y : y + h, x : x + w]
    roi_h, roi_w = roi_image.shape
    if roi_h <= 0 or roi_w <= 0:
        raise ValueError("ROI 区域为空")

    os.makedirs(output_dir, exist_ok=True)
    cell_h = roi_h // rows
    cell_w = roi_w // cols
    if cell_h <= 0 or cell_w <= 0:
        raise ValueError("网格过密，无法切分 cell")

    extracted = 0
    for i in range(rows):
        for j in range(cols):
            y1 = i * cell_h
            x1 = j * cell_w
            y2 = (i + 1) * cell_h if i < rows - 1 else roi_h
            x2 = (j + 1) * cell_w if j < cols - 1 else roi_w
            cell = roi_image[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            cell_resized = cv.resize(
                cell, (input_size, input_size), interpolation=cv.INTER_AREA
            )
            cell_filename = f"cell_{time.strftime('%Y%m%d%H%M%S')}_{extracted}.bmp"
            cell_path = os.path.join(output_dir, cell_filename)
            cv.imwrite(cell_path, cell_resized)
            extracted += 1
    return extracted

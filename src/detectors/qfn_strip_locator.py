# -*- coding: utf-8 -*-
"""整版 QFN 定位。

在整条料的图像上找出每一颗的塑封亮核，按行列网格给出中心坐标。
亮核在下采样图上做连通域，行距和列距取间距的主峰，靠得过近的碎块合并。
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from src.support.support_funs import ensure_gray_u8


class QfnStripLocator:
    def __init__(self, params: dict | None = None):
        self.params = {
            "scale": 4,
            "body_threshold": 220,
            "min_rows": 1,
        }
        if params:
            self.params.update(params)
        self.last_result = None

    def locate(self, image):
        gray = ensure_gray_u8(image, copy=False)
        centers = _body_centers(
            gray,
            int(self.params.get("scale", 4)),
            int(self.params.get("body_threshold", 220)),
        )
        if len(centers) == 0:
            self.last_result = _empty("整版上找不到塑封亮核")
            return self.last_result

        rows = _cluster_rows(centers)
        pitch_x = _pitch_from_rows(rows)
        if pitch_x is None:
            self.last_result = _empty("整版列距不稳定")
            return self.last_result
        rows = [_suppress_close(row, pitch_x * 0.65) for row in rows]
        rows = [row for row in rows if len(row) > 0]
        pitch_x = _pitch_from_rows(rows) or pitch_x
        pitch_y = _pitch_across_rows(rows) or pitch_x
        placed = _place_on_grid(rows, pitch_x, pitch_y)
        self.last_result = placed
        return placed


def _empty(message):
    return {
        "ok": False,
        "message": message,
        "rows": 0,
        "cols": 0,
        "pitch_x": 0.0,
        "pitch_y": 0.0,
        "found": 0,
        "missing": [],
        "units": [],
    }


def _body_centers(gray, scale, threshold):
    scale = max(2, int(scale))
    small = cv.resize(
        gray,
        (max(1, gray.shape[1] // scale), max(1, gray.shape[0] // scale)),
        interpolation=cv.INTER_AREA,
    )
    mask = (small >= threshold).astype(np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _count, _labels, stats, centroids = cv.connectedComponentsWithStats(mask, 8)
    if len(stats) <= 1:
        return np.empty((0, 2), dtype=np.float64)
    stats = stats[1:]
    centroids = centroids[1:]
    width = stats[:, cv.CC_STAT_WIDTH].astype(np.float64)
    height = stats[:, cv.CC_STAT_HEIGHT].astype(np.float64)
    area = stats[:, cv.CC_STAT_AREA].astype(np.float64)
    largest = area >= np.percentile(area, 97)
    if int(np.count_nonzero(largest)) < 3:
        return np.empty((0, 2), dtype=np.float64)
    height_ref = float(np.median(height[largest]))
    width_ref = float(np.median(width[largest]))
    keep = (
        (np.abs(height - height_ref) <= 0.28 * height_ref)
        & (width >= width_ref * 0.6)
        & (width <= max(width_ref * 1.6, height_ref))
    )
    if int(np.count_nonzero(keep)) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return centroids[keep] * float(scale)


def _cluster_rows(centers):
    order = np.argsort(centers[:, 1])
    points = centers[order]
    gaps = np.diff(points[:, 1])
    if len(gaps) == 0:
        return [points]
    split_at = max(40.0, float(np.percentile(gaps, 99)) * 0.45)
    rows = []
    start = 0
    for index in np.flatnonzero(gaps > split_at):
        rows.append(points[start : index + 1])
        start = index + 1
    rows.append(points[start:])
    return rows


def _pitch_from_rows(rows):
    gaps = [np.diff(np.sort(row[:, 0])) for row in rows if len(row) >= 3]
    if not gaps:
        return None
    gaps = np.concatenate(gaps)
    gaps = gaps[gaps > 1.0]
    if len(gaps) == 0:
        return None
    upper = gaps[gaps >= np.percentile(gaps, 55)]
    pitch = float(np.median(upper))
    if pitch < 8.0:
        return None
    return pitch


def _pitch_across_rows(rows):
    if len(rows) < 2:
        return None
    mids = np.array([float(np.median(row[:, 1])) for row in rows])
    gaps = np.diff(mids)
    gaps = gaps[gaps > 8.0]
    if len(gaps) == 0:
        return None
    return float(np.median(gaps))


def _suppress_close(points, minimum):
    points = points[np.argsort(points[:, 0])]
    kept = [points[0]]
    for point in points[1:]:
        if point[0] - kept[-1][0] < minimum:
            kept[-1] = (kept[-1] + point) * 0.5
            continue
        kept.append(point)
    return np.asarray(kept, dtype=np.float64)


def _phase(values, pitch):
    fraction = np.mod(values, pitch) / pitch
    angle = fraction * (2.0 * np.pi)
    mean = np.arctan2(float(np.mean(np.sin(angle))), float(np.mean(np.cos(angle))))
    if mean < 0.0:
        mean += 2.0 * np.pi
    return mean / (2.0 * np.pi) * pitch


def _place_on_grid(rows, pitch_x, pitch_y):
    row_y = np.array([float(np.median(row[:, 1])) for row in rows])
    origin_y = _phase(row_y, pitch_y)
    origin_x = _phase(np.concatenate([row[:, 0] for row in rows]), pitch_x)
    cells = {}
    for row in rows:
        row_index = int(np.rint((float(np.median(row[:, 1])) - origin_y) / pitch_y))
        for point in row:
            col_index = int(np.rint((float(point[0]) - origin_x) / pitch_x))
            previous = cells.get((row_index, col_index))
            center = (float(point[0]), float(point[1]))
            if previous is None:
                cells[(row_index, col_index)] = center
                continue
            target_x = origin_x + col_index * pitch_x
            target_y = origin_y + row_index * pitch_y
            old_dist = (previous[0] - target_x) ** 2 + (previous[1] - target_y) ** 2
            new_dist = (center[0] - target_x) ** 2 + (center[1] - target_y) ** 2
            if new_dist < old_dist:
                cells[(row_index, col_index)] = center

    row_ids = sorted({key[0] for key in cells})
    col_ids = sorted({key[1] for key in cells})
    row_map = {value: index for index, value in enumerate(row_ids)}
    col_map = {value: index for index, value in enumerate(col_ids)}
    units = []
    for (row_index, col_index), (x, y) in cells.items():
        units.append(
            {
                "row": row_map[row_index],
                "col": col_map[col_index],
                "x": round(x, 2),
                "y": round(y, 2),
            }
        )
    units.sort(key=lambda item: (item["row"], item["col"]))
    found = {(item["row"], item["col"]) for item in units}
    missing = [
        {"row": row, "col": col}
        for row in range(len(row_ids))
        for col in range(len(col_ids))
        if (row, col) not in found
    ]
    return {
        "ok": True,
        "message": "",
        "rows": len(row_ids),
        "cols": len(col_ids),
        "pitch_x": round(float(pitch_x), 2),
        "pitch_y": round(float(pitch_y), 2),
        "found": len(units),
        "missing": missing,
        "units": units,
    }

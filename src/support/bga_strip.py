from __future__ import annotations

from datetime import datetime as dt

import numpy as np

from src.support import data_structure
from src.support.support_funs import (
    calculate_cpk,
    create_alternating_array,
    calculate_write_positions,
    get_bga_animation_grid_layout,
)

_PRIORITY = {"Ball Count": 1, "Size": 2, "Ball_Area": 3, "Mark": 4, "Scratch": 5, "Shift": 6}
_NG_KEYS = ("Size", "Ball_Area", "Ball Count", "Mark", "Scratch", "Shift")
_DETECTION_LABELS = {
    "size_check_enable": "尺寸检测",
    "ball_check_enable": "锡球检测",
    "mark_check_enable": "Mark检测",
    "scratch_check_enable": "划痕检测",
    "shift_check_enable": "偏移检测",
}

# 状态码（Modbus 上报与动画统一口径，保持与旧实现一致）
_STATUS_BLANK = 0      # 非产品格（棋盘空白）
_STATUS_UNCHECKED = 99  # 合法格但未检
_DEFECT_CODE = {"Mark": 1, "Size": 3, "Ball Count": 4, "Ball_Area": 5, "Shift": 6, "Scratch": 7}
_STATUS_OK = 2
_STATUS_NG_DEFAULT = 8

# 状态码 → 动画颜色（BGR），与旧 get_full_animation 一致
_STATUS_COLOR = {
    2: (0, 255, 0),      # OK - 绿
    1: (0, 0, 255),      # Mark - 红
    3: (128, 0, 128),    # Size - 紫
    4: (0, 165, 255),    # BallCount - 橙
    5: (0, 255, 255),    # Ball_Area - 黄
    6: (42, 42, 165),    # Shift - 棕
    7: (255, 0, 0),      # Scratch - 蓝
    8: (0, 0, 255),      # NG - 红
    99: (255, 255, 255),  # 未检 - 白
    0: (0, 0, 0),        # 空白 - 黑
}


def _defect_state(product: data_structure.Product) -> tuple:
    defect_type = product.defect_type or ["OK"]
    first = defect_type[0] if defect_type else "OK"
    is_empty = first == "Empty"
    is_ng = not is_empty and first != "OK"
    return is_empty, is_ng, defect_type


def _primary_ng_type(defect_type: list):
    candidates = [t for t in defect_type if t in _PRIORITY]
    return min(candidates, key=lambda t: _PRIORITY[t]) if candidates else None


def _status_code(product: data_structure.Product) -> int:
    """由 Product.defect_type 推导状态码（单一数据源）。"""
    defect_type = product.defect_type or ["OK"]
    first = defect_type[0] if defect_type else "OK"
    if first == "Empty":
        return _STATUS_UNCHECKED
    if first == "OK":
        return _STATUS_OK
    for name, code in _DEFECT_CODE.items():
        if name in defect_type:
            return code
    return _STATUS_OK if "OK" in defect_type else _STATUS_NG_DEFAULT


def _summarize_defects(products: list) -> tuple:
    ng_total = empty_count = 0
    defect_counts = {k: 0 for k in _NG_KEYS}
    for product in products:
        is_empty, is_ng, defect_type = _defect_state(product)
        if is_empty:
            empty_count += 1
            continue
        if is_ng:
            ng_total += 1
            primary = _primary_ng_type(defect_type)
            if primary:
                defect_counts[primary] += 1
    return ng_total, empty_count, defect_counts


def _product_log_row(product: data_structure.Product, index: int) -> dict:
    is_empty, is_ng, defect_type = _defect_state(product)
    size = product.size_result
    shift = product.shift_result
    shift_x = shift.shift_x or 0.0
    shift_y = shift.shift_y or 0.0
    ng_types = list(dict.fromkeys(t for t in defect_type if t != "OK"))
    return {
        "product_index": index,
        "width": f"{size.width:.4f}" if size.width else "",
        "height": f"{size.height:.4f}" if size.height else "",
        "has_mark": "是" if product.mark_result.is_valid else "否",
        "ng_ball_count": str(product.ball_result.ng_ball_count or 0),
        "ng_ball_info": product.ball_result.ng_ball_info_text(),
        "shift_x": f"{shift_x:.4f}" if shift_x != 0.0 else "0.0000",
        "shift_y": f"{shift_y:.4f}" if shift_y != 0.0 else "0.0000",
        "is_ng": "否" if is_empty else ("是" if is_ng else "否"),
        "ng_types": "Empty" if is_empty else ";".join(ng_types),
    }


class BGA_STRIP:
    def __init__(self, station, strip_side, strip_lot, strip_sn, strip_create_time, params):
        self.params = params
        rows = int(params.get("total_rows", 0) or 0)
        cols = int(params.get("total_cols", 0) or 0)
        wrows = int(params.get("current_row", 0) or 0)
        wcols = int(params.get("current_col", 0) or 0)

        start_element = 0 if strip_side == "front" else 1
        if params.get("front_back_reverse"):
            start_element = 1 - start_element
        base = create_alternating_array(rows, cols, start_element, (0, 99))
        # 合法格（棋盘奇偶，由 side 决定）
        self.valid_mask = (base == 99) if rows > 0 and cols > 0 else np.zeros((rows, cols), dtype=bool)

        if rows > 0 and cols > 0 and wrows > 0 and wcols > 0:
            window_base = create_alternating_array(wrows, wcols, start_element, (0, 99))
            self.position_list = calculate_write_positions(base, window_base)
        else:
            self.position_list = []

        self.window_rows = wrows
        self.window_cols = wcols
        self.image_dict = {}
        self.count = 0  # 蛇形写入游标

        self.bga_strip: data_structure.Bga_Strip = data_structure.Bga_Strip(
            station=station,
            strip_lot=strip_lot,
            strip_sn=strip_sn,
            strip_create_time=strip_create_time,
            strip_side=strip_side,
            strip_cols=cols,
            strip_rows=rows,
            window_rows=wrows,
            window_cols=wcols,
            product_array=np.full((rows, cols), None, dtype=object),
            strip_image=None,
            animation_image=None,
        )

    # ---- 便捷属性（线程/UI 访问）----
    @property
    def strip_rows(self) -> int:
        return self.bga_strip.strip_rows

    @property
    def strip_cols(self) -> int:
        return self.bga_strip.strip_cols

    @property
    def strip_side(self) -> str:
        return self.bga_strip.strip_side

    @property
    def strip_lot(self) -> str:
        return self.bga_strip.strip_lot

    @property
    def strip_sn(self) -> str:
        return self.bga_strip.strip_sn

    @property
    def station(self) -> str:
        return self.bga_strip.station

    @property
    def product_array(self) -> np.ndarray:
        return self.bga_strip.product_array

    # ---- 写入 ----
    def pop_next_position(self):
        """取下一个蛇形写入位置并推进游标。返回 (start_point, window_size) 或 None。"""
        if self.count >= len(self.position_list):
            return None
        pos = self.position_list[self.count]
        self.count += 1
        return (pos[0], pos[1]), (self.window_rows, self.window_cols)

    def write(self, start_point, window_size, slot, current_image):
        """
        将窗口内 slot 的 Product 写入 product_array（每拍即写，单一数据源）。
        仅写入 valid_mask 合法且尚未写入的格；slot 对应格为 None 时写 Empty 占位 Product。

        参数:
            start_point: (row, col) 窗口左上角在 product_array 中的位置
            window_size: (rows, cols) 窗口尺寸（= current_row × current_col）
            slot: List[List[Optional[Product]]]，形状为 window_size
            current_image: 当前整帧图像
        """
        if start_point is None:
            return
        r0, c0 = int(start_point[0]), int(start_point[1])
        self.image_dict[(r0, c0)] = current_image

        if slot is None:
            return

        wr, wc = int(window_size[0]), int(window_size[1])
        pa = self.bga_strip.product_array
        rows, cols = self.bga_strip.strip_rows, self.bga_strip.strip_cols

        for lr in range(wr):
            for lc in range(wc):
                gr, gc = r0 + lr, c0 + lc
                if gr < 0 or gr >= rows or gc < 0 or gc >= cols:
                    continue
                if not self.valid_mask[gr, gc]:
                    continue
                if pa[gr, gc] is not None:
                    continue
                cell = None
                if lr < len(slot) and slot[lr] is not None and lc < len(slot[lr]):
                    cell = slot[lr][lc]
                if cell is None:
                    cell = data_structure.Product(defect_type=["Empty"], product_status="Empty")
                pa[gr, gc] = cell

    # ---- 派生视图 ----
    def get_status_array(self) -> np.ndarray:
        rows, cols = self.bga_strip.strip_rows, self.bga_strip.strip_cols
        arr = np.zeros((rows, cols), dtype=int)
        pa = self.bga_strip.product_array
        for r in range(rows):
            for c in range(cols):
                if not self.valid_mask[r, c]:
                    arr[r, c] = _STATUS_BLANK
                elif pa[r, c] is None:
                    arr[r, c] = _STATUS_UNCHECKED
                else:
                    arr[r, c] = _status_code(pa[r, c])
        return arr

    # 兼容旧的 .full_value 直接访问
    @property
    def full_value(self) -> np.ndarray:
        return self.get_status_array()

    def get_modbus_data(self) -> np.ndarray:
        return self.get_status_array()

    def get_full_animation(self) -> np.ndarray:
        array = self.get_status_array()
        h, w = array.shape if array.ndim == 2 else (0, 0)
        h = max(1, h)
        w = max(1, w)
        margin, block_height, block_width, img_h, img_w = get_bga_animation_grid_layout(
            h, w, margin=2, canvas_h=480, canvas_w=150
        )
        img = np.full((img_h, img_w, 3), 40, dtype=np.uint8)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                y1 = margin + i * (block_height + margin)
                x1 = margin + j * (block_width + margin)
                y2 = y1 + block_height
                x2 = x1 + block_width
                img[y1:y2, x1:x2] = _STATUS_COLOR.get(int(array[i, j]), (255, 255, 255))
        return img

    def set_strip_image(self, strip_image: np.ndarray):
        self.bga_strip.strip_image = strip_image

    def get_strip_image(self):
        return self.bga_strip.strip_image

    def get_pos_image(self, start_point):
        return self.image_dict.get((int(start_point[0]), int(start_point[1])), None)

    def get_frame_image_for_cell(self, row: int, col: int):
        """返回覆盖 (row, col) 的最后一次 write 整帧原图。"""
        row, col = int(row), int(col)
        wr, wc = self.window_rows, self.window_cols
        for i in range(min(self.count, len(self.position_list)) - 1, -1, -1):
            r0, c0 = int(self.position_list[i][0]), int(self.position_list[i][1])
            if r0 <= row < r0 + wr and c0 <= col < c0 + wc:
                return self.image_dict.get((r0, c0))
        return None

    def get_product(self, row: int, col: int):
        rows, cols = self.bga_strip.strip_rows, self.bga_strip.strip_cols
        if 0 <= row < rows and 0 <= col < cols:
            return self.bga_strip.product_array[row, col]
        return None

    def _iter_products(self) -> list:
        """按蛇形写入序遍历已写入的 Product（保证 product_index 与旧实现一致）。"""
        rows, cols = self.bga_strip.strip_rows, self.bga_strip.strip_cols
        pa = self.bga_strip.product_array
        products = []
        seen = set()
        if self.position_list:
            for pos in self.position_list:
                r0, c0 = pos[0], pos[1]
                for lr in range(self.window_rows):
                    for lc in range(self.window_cols):
                        gr, gc = r0 + lr, c0 + lc
                        if gr < 0 or gr >= rows or gc < 0 or gc >= cols:
                            continue
                        if (gr, gc) in seen:
                            continue
                        product = pa[gr, gc]
                        if product is not None:
                            seen.add((gr, gc))
                            products.append(product)
        else:
            for r in range(rows):
                for c in range(cols):
                    product = pa[r, c]
                    if product is not None:
                        products.append(product)
        return products

    def get_log_info(self):
        try:
            products = self._iter_products()
            ng_total, _, defect_counts = _summarize_defects(products)

            if self.bga_strip.strip_create_time:
                try:
                    start_time = dt.strptime(self.bga_strip.strip_create_time, "%Y%m%d%H%M%S")
                except ValueError:
                    start_time = dt.now()
            else:
                start_time = dt.now()
            end_time = dt.now()

            widths = [p.size_result.width for p in products if p.size_result.width]
            heights = [p.size_result.height for p in products if p.size_result.height]
            ball_radii = [r for p in products for r in p.ball_result.all_ball_radius_mm()]
            shift_x_list = [
                p.shift_result.shift_x for p in products
                if p.shift_result.shift_x is not None
            ]
            shift_y_list = [
                p.shift_result.shift_y for p in products
                if p.shift_result.shift_y is not None
            ]

            shift_x_cpk = shift_y_cpk = None
            if shift_x_list and self.params.get("shift_check_enable", False):
                tol = self.params.get("shift_x_tolerance", 0.5)
                shift_x_cpk = calculate_cpk(shift_x_list, tol, -tol)
            if shift_y_list and self.params.get("shift_check_enable", False):
                tol = self.params.get("shift_y_tolerance", 0.5)
                shift_y_cpk = calculate_cpk(shift_y_list, tol, -tol)

            return {
                "lot_id": self.bga_strip.strip_lot,
                "sn_id": self.bga_strip.strip_sn,
                "process_info": {
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": (end_time - start_time).total_seconds(),
                    "total_products": len(products),
                    "ng_total": ng_total,
                    "enabled_detections": [
                        label for key, label in _DETECTION_LABELS.items()
                        if self.params.get(key, True)
                    ],
                },
                "statistics": {
                    "width_range": max(widths) - min(widths) if widths else None,
                    "height_range": max(heights) - min(heights) if heights else None,
                    "avg_width": sum(widths) / len(widths) if widths else None,
                    "avg_height": sum(heights) / len(heights) if heights else None,
                    "avg_ball_radius": sum(ball_radii) / len(ball_radii) if ball_radii else None,
                    "shift_x_max": max(shift_x_list) if shift_x_list else None,
                    "shift_x_min": min(shift_x_list) if shift_x_list else None,
                    "shift_y_max": max(shift_y_list) if shift_y_list else None,
                    "shift_y_min": min(shift_y_list) if shift_y_list else None,
                    "shift_x_cpk": shift_x_cpk,
                    "shift_y_cpk": shift_y_cpk,
                },
                "defect_statistics": {k: defect_counts[k] for k in _NG_KEYS},
                "product_list": [
                    _product_log_row(product, index + 1)
                    for index, product in enumerate(products)
                ],
            }
        except Exception as e:
            print(f"整理日志信息错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_statistics_info(self) -> dict:
        try:
            products = self._iter_products()
            ng_total, empty_count, defect_counts = _summarize_defects(products)
            total_count = len(products)
            yield_rate = (
                (total_count - ng_total) / total_count * 100 if total_count > 0 else 0.0
            )
            station = self.bga_strip.station
            return {
                "station": "干燥台" if station == "dry" else "移栽台",
                "lot_id": self.bga_strip.strip_lot,
                "total_count": total_count,
                "empty_count": empty_count,
                "ng_count": ng_total,
                "yield_rate": yield_rate,
                "defect_counts": {
                    "Mark": defect_counts["Mark"],
                    "Size": defect_counts["Size"],
                    "Ball_Area": defect_counts["Ball_Area"],
                    "Ball Count": defect_counts["Ball Count"],
                    "Scratch": defect_counts["Scratch"],
                    "Shift": defect_counts["Shift"],
                },
            }
        except Exception as e:
            print(f"获取统计信息错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                "station": "干燥台" if self.bga_strip.station == "dry" else "移栽台",
                "lot_id": "",
                "total_count": 0,
                "empty_count": 0,
                "ng_count": 0,
                "yield_rate": 0.0,
                "defect_counts": {k: 0 for k in ("Mark", "Size", "Ball_Area", "Ball Count", "Scratch", "Shift")},
            }

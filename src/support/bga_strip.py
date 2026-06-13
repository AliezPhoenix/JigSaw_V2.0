from __future__ import annotations

from datetime import datetime as dt

from src.support import data_structure
import numpy as np

from src.support.support_funs import calculate_cpk

_PRIORITY = {"Ball Count": 1, "Size": 2, "Ball_Area": 3, "Mark": 4, "Scratch": 5, "Shift": 6}
_NG_KEYS = ("Size", "Ball_Area", "Ball Count", "Mark", "Scratch", "Shift")
_DETECTION_LABELS = {
    "size_check_enable": "尺寸检测",
    "ball_check_enable": "锡球检测",
    "mark_check_enable": "Mark检测",
    "scratch_check_enable": "划痕检测",
    "shift_check_enable": "偏移检测",
}


def _defect_state(product: data_structure.Product) -> tuple[bool, bool, list[str]]:
    defect_type = product.defect_type or ["OK"]
    first = defect_type[0] if defect_type else "OK"
    is_empty = first == "Empty"
    is_ng = not is_empty and first != "OK"
    return is_empty, is_ng, defect_type


def _primary_ng_type(defect_type: list[str]) -> str | None:
    candidates = [t for t in defect_type if t in _PRIORITY]
    return min(candidates, key=lambda t: _PRIORITY[t]) if candidates else None


def _summarize_defects(products: list[data_structure.Product]) -> tuple[int, int, dict[str, int]]:
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
    def __init__(self, params: dict):
        self.params = params
        self.bga_strip: data_structure.Bga_Strip = data_structure.Bga_Strip(
            station=params.get("station", ""),
            strip_lot="",
            strip_sn="",
            strip_create_time="",
            strip_side=params.get("strip_side", "front"),
            strip_cols=params.get("total_cols", 0),
            strip_rows=params.get("total_rows", 0),
            product_array=np.full(
                (params.get("total_rows", 0), params.get("total_cols", 0)),
                None,
                dtype=object,
            ),
            strip_image=None,
            animation_image=None,
        )

    def _iter_products(self) -> list[data_structure.Product]:
        rows, cols = self.bga_strip.strip_rows, self.bga_strip.strip_cols
        products: list[data_structure.Product] = []
        for r in range(rows):
            for c in range(cols):
                product = self.bga_strip.product_array[r, c]
                if product is not None:
                    products.append(product)
        return products

    def write(
        self,
        start_point: tuple[int, int],
        window_size: tuple[int, int],
        product: data_structure.Product,
    ):
        for row in range(start_point[0], start_point[0] + window_size[0]):
            for col in range(start_point[1], start_point[1] + window_size[1]):
                self.bga_strip.product_array[row, col] = product

    def set_strip_image(self, strip_image: np.ndarray):
        self.bga_strip.strip_image = strip_image

    def set_animation_image(self, animation_image: np.ndarray):
        self.bga_strip.animation_image = animation_image

    def get_strip_image(self):
        return self.bga_strip.strip_image

    def get_animation_image(self):
        return self.bga_strip.animation_image

    def render_animation(self):
        animation_map = np.zeros((self.bga_strip.strip_rows, self.bga_strip.strip_cols))
        for row in range(self.bga_strip.strip_rows):
            for col in range(self.bga_strip.strip_cols):
                current_product = self.bga_strip.product_array[row, col]
                if current_product is not None:
                    animation_map[row, col] = current_product.animation_color
                else:
                    animation_map[row, col] = 0

        h, w = animation_map.shape
        h = max(1, h)
        w = max(1, w)
        margin = 2
        block_height = max(1, 480 // h)
        block_width = max(1, 150 // w)
        animation_image = np.full(
            (h * block_height + (h + 1) * margin, w * block_width + (w + 1) * margin, 3),
            40,
            dtype=np.uint8,
        )
        for i in range(h):
            for j in range(w):
                y1 = margin + i * (block_height + margin)
                x1 = margin + j * (block_width + margin)
                y2 = y1 + block_height
                x2 = x1 + block_width
                animation_image[y1:y2, x1:x2] = animation_map[i, j]

        self.set_animation_image(animation_image)

    def get_modbus_data(self):
        modbus_data = np.zeros((self.bga_strip.strip_rows, self.bga_strip.strip_cols))
        code_map = {
            "OK": 2,
            "Mark": 1,
            "Size": 3,
            "Ball Count": 4,
            "Ball_Area": 5,
            "Shift": 6,
            "Scratch": 7,
        }
        for row in range(self.bga_strip.strip_rows):
            for col in range(self.bga_strip.strip_cols):
                current_product = self.bga_strip.product_array[row, col]
                if current_product is None:
                    modbus_data[row, col] = 0
                    continue
                defect_type = (current_product.defect_type or ["OK"])[0]
                modbus_data[row, col] = code_map.get(defect_type, 0)
        return modbus_data

    def get_log_info(self) -> dict | None:
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

    def get_product(self, row: int, col: int) -> data_structure.Product:
        return self.bga_strip.product_array[row, col]

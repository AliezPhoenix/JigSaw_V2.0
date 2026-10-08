# -*- coding: utf-8 -*-
"""清理 size/log 目录中的产品检测结果日志。

删除高度小于 14.975 mm 的明细行；高度为空或无法解析的行保留。
按剩余行重算统计信息、总产品数、NG 总数和各项不良数。
序号不改。原文件不改，结果写入 size/log_cleaned/。
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from openpyxl import load_workbook

HEIGHT_LIMIT = 14.970

STAT_ORDER = (
    "宽度极值(mm)",
    "高度极值(mm)",
    "平均尺寸(mm)",
    "平均球半径(mm)",
    "偏移X极值(mm)",
    "偏移Y极值(mm)",
    "偏移X CPK",
    "偏移Y CPK",
)

SHIFT_LABELS = frozenset(STAT_ORDER[4:])

DEFECT_LABEL_TO_KEY = {
    "尺寸检测不良数": "Size",
    "锡球面积检测不良数": "Ball_Area",
    "锡球数量检测不良数": "Ball Count",
    "Mark检测不良数": "Mark",
    "划痕检测不良数": "Scratch",
    "偏移检测不良数": "Shift",
}

# 与 src/support/bga_strip.py 的主不良类型优先级一致
_DEFECT_PRIORITY = {
    "Ball Count": 1,
    "Size": 2,
    "Ball_Area": 3,
    "Mark": 4,
    "Scratch": 5,
    "Shift": 6,
}


def calculate_cpk(data_list, usl, lsl):
    """与 src/support/support_funs.calculate_cpk 相同。"""
    if not data_list or len(data_list) < 2:
        return None
    mean = statistics.fmean(data_list)
    std = statistics.stdev(data_list)
    if std == 0:
        return None
    cpu = (usl - mean) / (3 * std) if usl is not None else None
    cpl = (mean - lsl) / (3 * std) if lsl is not None else None
    if cpu is not None and cpl is not None:
        return max(cpu, cpl)
    if cpu is not None:
        return cpu
    return cpl


def infer_shift_tolerance(values, cpk):
    """由清洗前的偏移样本和已写入的 CPK 反推规格容差。

    双侧限为 +tol / -tol 时，calculate_cpk 的结果等于 (tol + |均值|) / (3σ)。
    """
    if cpk is None or len(values) < 2:
        return None
    std = statistics.stdev(values)
    if std == 0:
        return None
    tol = float(cpk) * 3.0 * std - abs(statistics.fmean(values))
    if tol <= 0:
        return None
    return tol


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ball_radii(text):
    if text is None:
        return []
    radii = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        radius_text = part.split(",")[0].strip()
        number = parse_number(radius_text)
        if number is not None:
            radii.append(number)
    return radii


def primary_defect(ng_types):
    if ng_types is None:
        return None
    text = str(ng_types).strip()
    if not text or text == "Empty":
        return None
    parts = [part.strip() for part in text.replace("；", ";").split(";") if part.strip()]
    candidates = [part for part in parts if part in _DEFECT_PRIORITY]
    if not candidates:
        return None
    return min(candidates, key=lambda name: _DEFECT_PRIORITY[name])


def fmt4(value):
    return f"{value:.4f}"


def range_text(values):
    if not values:
        return None
    high = max(values)
    low = min(values)
    return f"{fmt4(high - low)} (最大:{fmt4(high)} - 最小:{fmt4(low)})"


def extreme_text(values):
    if not values:
        return None
    return f"最大:{fmt4(max(values))}, 最小:{fmt4(min(values))}"


def average_size_text(widths, heights):
    parts = []
    if widths:
        parts.append(f"宽度:{fmt4(statistics.fmean(widths))}")
    if heights:
        parts.append(f"高度:{fmt4(statistics.fmean(heights))}")
    if not parts:
        return None
    return ", ".join(parts)


def index_labels(ws):
    found = {}
    for row in range(1, ws.max_row + 1):
        label = ws.cell(row=row, column=1).value
        if not isinstance(label, str):
            continue
        key = label.strip()
        if key and key not in found:
            found[key] = row
    return found


def find_row(labels, needle):
    for key, row in labels.items():
        if needle in key:
            return row
    return None


def find_header_row(ws):
    for row in range(1, ws.max_row + 1):
        first = ws.cell(row=row, column=1).value
        second = ws.cell(row=row, column=2).value
        if str(first).strip() == "序号" and str(second).strip() == "宽度(mm)":
            return row
    return None


def row_values(ws, row):
    return [ws.cell(row=row, column=col).value for col in range(1, 11)]


def is_empty_row(values):
    return all(value is None or (isinstance(value, str) and not value.strip()) for value in values)


def collect_products(ws, header_row):
    products = []
    for row in range(header_row + 1, ws.max_row + 1):
        values = row_values(ws, row)
        if is_empty_row(values):
            continue
        products.append((row, values))
    return products


def should_drop(values):
    height = parse_number(values[2])
    if height is None:
        return False
    return round(height, 4) < HEIGHT_LIMIT


def build_statistics(kept_rows, tolerance_x, tolerance_y):
    widths = []
    heights = []
    shift_x = []
    shift_y = []
    radii = []
    ng_total = 0
    defect_counts = {key: 0 for key in DEFECT_LABEL_TO_KEY.values()}

    for values in kept_rows:
        width = parse_number(values[1])
        height = parse_number(values[2])
        if width is not None:
            widths.append(width)
        if height is not None:
            heights.append(height)
        sx = parse_number(values[6])
        sy = parse_number(values[7])
        if sx is not None:
            shift_x.append(sx)
        if sy is not None:
            shift_y.append(sy)
        radii.extend(parse_ball_radii(values[5]))
        if str(values[8]).strip() == "是":
            ng_total += 1
            primary = primary_defect(values[9])
            if primary:
                defect_counts[primary] += 1

    cpk_x = calculate_cpk(shift_x, tolerance_x, -tolerance_x) if tolerance_x is not None else None
    cpk_y = calculate_cpk(shift_y, tolerance_y, -tolerance_y) if tolerance_y is not None else None
    avg_radius = statistics.fmean(radii) if radii else None

    stats = {
        "宽度极值(mm)": range_text(widths),
        "高度极值(mm)": range_text(heights),
        "平均尺寸(mm)": average_size_text(widths, heights),
        "平均球半径(mm)": fmt4(avg_radius) if avg_radius is not None else None,
        "偏移X极值(mm)": extreme_text(shift_x),
        "偏移Y极值(mm)": extreme_text(shift_y),
        "偏移X CPK": fmt4(cpk_x) if cpk_x is not None else None,
        "偏移Y CPK": fmt4(cpk_y) if cpk_y is not None else None,
    }
    return stats, ng_total, defect_counts


def original_series(products, column):
    series = []
    for _, values in products:
        number = parse_number(values[column])
        if number is not None:
            series.append(number)
    return series


def shift_was_measured(products):
    """明细里的 0 可能只是空偏移的占位。出现非 0，或原表已有偏移统计时才补写。"""
    for _, values in products:
        for column in (6, 7):
            number = parse_number(values[column])
            if number is not None and number != 0:
                return True
    return False


def apply_statistics(ws, labels, stats, write_shift):
    defect_row = find_row(labels, "各检测项目不良统计")
    anchor = defect_row
    if anchor and anchor > 1 and is_empty_row(row_values(ws, anchor - 1)):
        anchor -= 1

    missing = []
    for label in STAT_ORDER:
        value = stats[label]
        row = labels.get(label)
        if row is not None:
            ws.cell(row=row, column=2).value = value
        elif value is not None and (label not in SHIFT_LABELS or write_shift):
            missing.append(label)

    if missing and anchor is not None:
        ws.insert_rows(anchor, amount=len(missing))
        for offset, label in enumerate(missing):
            ws.cell(row=anchor + offset, column=1).value = label
            ws.cell(row=anchor + offset, column=2).value = stats[label]


def apply_counts(ws, labels, total_products, ng_total, defect_counts):
    total_row = labels.get("总产品数")
    if total_row is not None:
        ws.cell(row=total_row, column=2).value = total_products
    ng_row = labels.get("NG总数")
    if ng_row is not None:
        ws.cell(row=ng_row, column=2).value = ng_total
    for label, key in DEFECT_LABEL_TO_KEY.items():
        row = labels.get(label)
        if row is not None:
            ws.cell(row=row, column=2).value = defect_counts[key]


def read_cpk(ws, labels, needle):
    row = find_row(labels, needle)
    if row is None:
        return None
    return parse_number(ws.cell(row=row, column=2).value)


def process_workbook(ws):
    header_row = find_header_row(ws)
    if header_row is None:
        return None

    labels = index_labels(ws)
    products = collect_products(ws, header_row)
    kept_rows = []
    drop_rows = []
    for row, values in products:
        if should_drop(values):
            drop_rows.append(row)
        else:
            kept_rows.append(values)

    tolerance_x = infer_shift_tolerance(original_series(products, 6), read_cpk(ws, labels, "偏移X CPK"))
    tolerance_y = infer_shift_tolerance(original_series(products, 7), read_cpk(ws, labels, "偏移Y CPK"))
    stats, ng_total, defect_counts = build_statistics(kept_rows, tolerance_x, tolerance_y)

    for row in reversed(drop_rows):
        ws.delete_rows(row, 1)

    write_shift = shift_was_measured(products) or any(find_row(labels, label) for label in SHIFT_LABELS)
    apply_counts(ws, labels, len(kept_rows), ng_total, defect_counts)
    apply_statistics(ws, labels, stats, write_shift)
    return {
        "deleted": len(drop_rows),
        "kept": len(kept_rows),
        "tolerance_x": tolerance_x,
        "tolerance_y": tolerance_y,
        "stats": stats,
    }


def clean_log_dir(log_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in log_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if not files:
        print(f"未找到批次日志: {log_dir}")
        return 0

    written = 0
    for path in files:
        wb = load_workbook(path)
        try:
            result = process_workbook(wb.active)
            if result is None:
                print(f"跳过（不是检测结果日志）: {path.name}")
                continue
            target = out_dir / path.name
            wb.save(target)
            written += 1
            tol_x = result["tolerance_x"]
            tol_y = result["tolerance_y"]
            tol_text = ""
            if tol_x is not None or tol_y is not None:
                tol_text = (
                    f"，偏移容差 X={fmt4(tol_x) if tol_x is not None else '无'}"
                    f" / Y={fmt4(tol_y) if tol_y is not None else '无'}"
                )
            print(f"{path.name}: 删除 {result['deleted']} 行，保留 {result['kept']} 行{tol_text}")
        finally:
            wb.close()
    print(f"已写入 {written} 个文件到 {out_dir}")
    return written


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent
    log_dir = root / "size" / "log"
    if not log_dir.is_dir():
        print(f"日志目录不存在: {log_dir}")
        return 1
    clean_log_dir(log_dir, root / "size" / "log_cleaned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

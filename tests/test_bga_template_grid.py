"""
BGA 窗口网格逻辑测试：assign_matches_to_grid 与 Bga_Strip.write（合成 params，无相机）。

含 3×6 窗口：空网格、单点随机（48 组 seed）、多点随机与置信度合并（24 组 seed）；
Bga_Strip.write 含 (6,12,3,6)、(9,18,3,6) 整盘参数。

运行（仓库根目录）:
  pip install -r requirements-dev.txt
  pytest tests/test_bga_template_grid.py -v
"""

import numpy as np
import pytest

from src.support.support_funs import (
    Bga_Strip,
    assign_matches_to_grid,
    _is_vacant_log_entry,
)


# --- assign_matches_to_grid ---


def test_assign_single_match_bottom_right_2x2():
    """100×100 ROI，2×2 网格，单点中心落在右下格。"""
    tw, th = 10, 10
    roi = [0, 0, 100, 100]
    # 右下格 cell: x in [50,100), y in [50,100)；中心 (65,65) -> 左上角 (60,60)
    matches = [(60, 60, 0.95)]
    out = assign_matches_to_grid(matches, tw, th, roi, (100, 100), 2, 2)
    assert out[0][0] is None and out[0][1] is None
    assert out[1][0] is None
    assert out[1][1] == (60, 60)


def test_assign_two_matches_different_cells_2x2():
    tw, th = 10, 10
    roi = [0, 0, 100, 100]
    matches = [(5, 5, 0.8), (60, 60, 0.85)]
    out = assign_matches_to_grid(matches, tw, th, roi, (100, 100), 2, 2)
    assert out[0][0] == (5, 5)
    assert out[1][1] == (60, 60)


def test_assign_same_cell_keeps_higher_confidence():
    tw, th = 10, 10
    roi = [0, 0, 100, 100]
    matches = [(5, 5, 0.5), (8, 8, 0.99)]
    out = assign_matches_to_grid(matches, tw, th, roi, (100, 100), 2, 2)
    assert out[0][0] == (8, 8)


def test_assign_invalid_grid_returns_placeholder_shape():
    out = assign_matches_to_grid([], 10, 10, None, (100, 100), 0, 0)
    assert len(out) >= 1 and len(out[0]) >= 1


@pytest.mark.parametrize(
    "grid_r,grid_c",
    [(2, 2), (2, 3), (3, 2), (1, 1), (3, 6)],
)
def test_assign_empty_matches_all_none(grid_r, grid_c):
    out = assign_matches_to_grid([], 10, 10, [0, 0, 200, 200], (200, 200), grid_r, grid_c)
    assert len(out) == grid_r and len(out[0]) == grid_c
    for r in range(grid_r):
        for c in range(grid_c):
            assert out[r][c] is None


def _expected_cell_xy(
    x: int,
    y: int,
    tw: int,
    th: int,
    roi: list,
    img_hw: tuple,
    grid_r: int,
    grid_c: int,
) -> tuple:
    """与 assign_matches_to_grid 分桶一致，返回 (gr, gc)。"""
    img_h, img_w = img_hw
    if roi and len(roi) >= 4:
        rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    else:
        rx, ry, rw, rh = 0, 0, img_w, img_h
    rw = max(1, rw)
    rh = max(1, rh)
    cell_w = rw / float(grid_c)
    cell_h = rh / float(grid_r)
    cx = float(x) + tw / 2.0
    cy = float(y) + th / 2.0
    gc = int((cx - rx) / cell_w) if cell_w > 0 else 0
    gr = int((cy - ry) / cell_h) if cell_h > 0 else 0
    gc = max(0, min(grid_c - 1, gc))
    gr = max(0, min(grid_r - 1, gr))
    return gr, gc


def _merge_matches_expected(matches, tw, th, roi, img_hw, grid_r, grid_c):
    """同格多点时保留最高 conf，返回 dict[(gr,gc)] -> (x, y, conf)。"""
    best = {}
    for x, y, conf in matches:
        gr, gc = _expected_cell_xy(x, y, tw, th, roi, img_hw, grid_r, grid_c)
        key = (gr, gc)
        if key not in best or conf > best[key][2]:
            best[key] = (x, y, float(conf))
    return best


@pytest.mark.parametrize("seed", list(range(48)))
def test_assign_random_single_match_3x6_expected_cell(seed: int):
    """3×6 窗口，单随机匹配点，落格与独立计算的分桶一致。"""
    rng = np.random.default_rng(seed)
    tw, th = 8, 8
    img_w, img_h = 240, 180
    roi = [0, 0, img_w, img_h]
    grid_r, grid_c = 3, 6
    x = int(rng.integers(0, max(1, img_w - tw + 1)))
    y = int(rng.integers(0, max(1, img_h - th + 1)))
    conf = float(rng.random())
    matches = [(x, y, conf)]
    out = assign_matches_to_grid(matches, tw, th, roi, (img_h, img_w), grid_r, grid_c)
    gr, gc = _expected_cell_xy(x, y, tw, th, roi, (img_h, img_w), grid_r, grid_c)
    assert out[gr][gc] == (x, y)
    for r in range(grid_r):
        for c in range(grid_c):
            if (r, c) != (gr, gc):
                assert out[r][c] is None


@pytest.mark.parametrize("seed", list(range(24)))
def test_assign_random_many_matches_3x6_conf_merge(seed: int):
    """3×6 窗口，多随机点，同格保留最高置信度，与参考合并结果一致。"""
    rng = np.random.default_rng(seed + 10_000)
    tw, th = 6, 6
    img_w, img_h = 300, 150
    roi = [0, 0, img_w, img_h]
    grid_r, grid_c = 3, 6
    n_matches = 50
    matches = [
        (
            int(rng.integers(0, max(1, img_w - tw + 1))),
            int(rng.integers(0, max(1, img_h - th + 1))),
            float(rng.random()),
        )
        for _ in range(n_matches)
    ]
    out = assign_matches_to_grid(matches, tw, th, roi, (img_h, img_w), grid_r, grid_c)
    expected = _merge_matches_expected(matches, tw, th, roi, (img_h, img_w), grid_r, grid_c)
    for gr in range(grid_r):
        for gc in range(grid_c):
            key = (gr, gc)
            if key in expected:
                ex, ey, _ = expected[key]
                assert out[gr][gc] == (ex, ey)
            else:
                assert out[gr][gc] is None


# --- Bga_Strip.write ---


def _make_strip(total_rows, total_cols, current_row, current_col, strip_side="front"):
    params = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "current_row": current_row,
        "current_col": current_col,
    }
    return Bga_Strip(
        station="dry",
        strip_side=strip_side,
        strip_lot="LOT",
        strip_sn="SN",
        strip_create_time="20260101120000",
        params=params,
    )


def _build_slot_from_window_value(bga):
    """按 window_value 中 99 格填 OK，其余 None。"""
    wv = np.array(bga.window_value)
    slot = []
    for gr in range(bga.window_rows):
        row = []
        for gc in range(bga.window_cols):
            if wv[gr, gc] == 99:
                row.append({"defect_type": ["OK"]})
            else:
                row.append(None)
        slot.append(row)
    return slot


def _first_window_targets(bga):
    """与 write 内一致：第一个 position 的切片中值为 99 的局部坐标。"""
    pos = bga.position_list[0]
    row_start, col_start, _, _, actual_row_end, actual_col_end, _, _ = pos
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_end = min(actual_row_end, bga.full_value_rows)
    col_end = min(actual_col_end, bga.full_value_cols)
    full_slice = bga.full_value[row_start:row_end, col_start:col_end]
    targets = [
        (r, c)
        for r in range(full_slice.shape[0])
        for c in range(full_slice.shape[1])
        if full_slice[r, c] == 99
    ]
    return row_start, col_start, targets


@pytest.mark.parametrize(
    "total_rows,total_cols,win_r,win_c",
    [
        (4, 4, 2, 2),
        (6, 6, 2, 3),
        (6, 6, 3, 2),
        (2, 2, 1, 1),
        (4, 4, 1, 1),
        (6, 12, 3, 6),
        (9, 18, 3, 6),
    ],
)
def test_write_ok_slot_updates_full_value_and_logs_per_cell(
    total_rows, total_cols, win_r, win_c
):
    bga = _make_strip(total_rows, total_cols, win_r, win_c)
    assert len(bga.position_list) >= 1
    log_before = len(bga.accumulated_log_info)
    count_before = bga.count

    row_start, col_start, targets = _first_window_targets(bga)
    slot = _build_slot_from_window_value(bga)
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)

    bga.write(slot, dummy_img)

    assert bga.count == count_before + 1
    assert len(bga.accumulated_log_info) == log_before + win_r * win_c

    empty_logs = sum(1 for e in bga.accumulated_log_info[log_before:] if _is_vacant_log_entry(e))
    none_cells = sum(
        1
        for gr in range(win_r)
        for gc in range(win_c)
        if slot[gr][gc] is None
    )
    assert empty_logs == none_cells

    for r, c in targets:
        assert bga.full_value[row_start + r, col_start + c] == 2


def test_write_slot_none_appends_empty_logs_and_increments_count():
    bga = _make_strip(4, 4, 2, 2)
    log_before = len(bga.accumulated_log_info)
    count_before = bga.count
    wr, wc = bga.window_rows, bga.window_cols
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)

    bga.write(None, dummy_img)

    assert bga.count == count_before + 1
    added = bga.accumulated_log_info[log_before:]
    assert len(added) == wr * wc
    assert all(_is_vacant_log_entry(e) for e in added)
    for i, e in enumerate(added):
        assert e["defect_type"] == ["Empty"]

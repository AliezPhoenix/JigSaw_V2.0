from itertools import product
import re
from typing import Dict, List, Sequence, Union

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


def _is_bga_product_slot(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], list)
    )


def _is_vacant_log_entry(product: dict) -> bool:
    dt = product.get("defect_type", [])
    return isinstance(dt, list) and len(dt) > 0 and dt[0] == "Empty"


def _vacant_log_entry(product_index: int, gr: int, gc: int) -> dict:
    """模板未匹配或无效格：写入 accumulated_log_info 的占位结构（与正常条目共字段）。"""
    return {
        "product_index": product_index,
        "size": [None, None],
        "has_mark": False,
        "ok_balls": [],
        "ng_balls": [],
        "shift_x": None,
        "shift_y": None,
        "defect_type": ["Empty"],
        "grid_row": gr,
        "grid_col": gc,
    }


class Bga_Strip():
    def __init__(self,station,strip_side:str, strip_lot:str, strip_sn, strip_create_time:str, params:dict):
        # 从params中获取尺寸参数
        self.strip_cols = params.get("total_cols", 0)
        self.strip_rows = params.get("total_rows", 0)
        self.window_rows = params.get("current_row", 0)
        self.window_cols = params.get("current_col", 0)
        
        self.strip_side = strip_side
        self.strip_lot = strip_lot
        self.strip_sn = strip_sn
        self.strip_create_time = strip_create_time
        
        # 保存params
        self.params = params
        
        if strip_side == "front":
            start_element = 0
        else:
            start_element = 1
        self.full_value = create_alternating_array(self.strip_rows, self.strip_cols, start_element, (0,99))
        self.window_value = create_alternating_array(self.window_rows, self.window_cols, start_element, (0,99))
        self.position_list = calculate_write_positions(self.full_value, self.window_value)
        
        # 添加bga_strip_log相关属性
        self.image_dict = {}
        self.full_value_cols = self.strip_cols
        self.full_value_rows = self.strip_rows
        self.window_value_cols = self.window_cols
        self.window_value_rows = self.window_rows
        self.side = strip_side
        self.station = station
        # 初始化图像字典
        for pos in self.position_list:
            self.image_dict[pos[0:2]] = np.zeros((100,100))
        
        self.count = 0
        self.log_file = {
            "Lot_ID": "",
            "Strip_ID": "",
            "Date": "",
            "Time": "",
            "Activate_Defect_Type": "",
            "Result": ""
        }
        # 添加最大历史记录数量限制（用于内存管理）
        self.max_history_size = 500  # 默认保留最近500条记录
        # 添加日志数据存储
        self.accumulated_log_info = []  # 存储所有产品检测信息
    
    def write(self, slot, current_image):
        """
        将二维 slot 中每格的产品结果写入 full_value 当前窗口切片内值为 99 的格；
        slot[行][列] 与窗口棋盘一致，None 表示该格未检出，保持 99。
        accumulated_log_info 仅对本切片内值为 99 的格各记一条（与写入棋盘的目标格一致），不遍历整窗。

        参数:
            slot: List[List[Optional[dict]]]，形状为 window_rows × window_cols，与 params 中 current_row/current_col 一致
            current_image: 当前图像
        """
        # 检查count是否超出范围
        if self.count >= len(self.position_list):
            print(f"错误: count ({self.count}) 超出position_list范围 ({len(self.position_list)})")
            return

        if slot is None:
            print("警告: slot 为 None，本帧棋盘不写值，按格记日志为 Empty")
            return

        if not _is_bga_product_slot(slot):
            print("错误: write 需要二维 slot [行][列]（与 current_row × current_col 一致），按格记日志为 Empty")
            self.image_dict[self.position_list[self.count][0:2]] = current_image
            self.count += 1
            return

        if ( len(slot) != self.window_rows or any(len(row) != self.window_cols for row in slot)):
            print(
                f"错误: slot 维度 {len(slot)}×{len(slot[0]) if slot else 0} "
                f"与窗口 {self.window_rows}×{self.window_cols} 不符，本帧不写值，按格记日志为 Empty")
            self.image_dict[self.position_list[self.count][0:2]] = current_image
            self.count += 1
            return

        # 获取当前位置信息
        pos = self.position_list[self.count]
        row_start, col_start, _, _, actual_row_end, actual_col_end, _, _ = pos

        # 调整切片范围到最大可用范围
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(actual_row_end, self.full_value_rows)
        col_end = min(actual_col_end, self.full_value_cols)

        # 验证调整后的范围有效性
        if row_end <= row_start or col_end <= col_start:
            print(f"错误: 调整后的切片范围无效: ({row_start}, {col_start}) 到 ({row_end}, {col_end})")
            self.image_dict[pos[0:2]] = current_image
            self.count += 1
            return

        # 获取切片并找到所有值为99的位置（99代表未检测到产品，需要写入检测结果）
        full_slice = self.full_value[row_start:row_end, col_start:col_end].copy()
        target_positions = [
            (row, col)
            for row in range(full_slice.shape[0])
            for col in range(full_slice.shape[1])
            if full_slice[row, col] == 99
        ]

        if not target_positions:
            print("警告: 切片中没有值为99的位置，跳过棋盘写入，本帧不写日志条目")
            self.image_dict[pos[0:2]] = current_image
            self.count += 1
            return

        for row, col in target_positions:
            product_info = slot[row][col]
            if product_info is None:
                continue
            defect_type = product_info["defect_type"]
            # 根据defect_type列表中的缺陷类型设置对应的值
            if not isinstance(defect_type, list) or len(defect_type) == 0 or defect_type[0] == "OK":
                full_slice[row, col] = 2  # OK - 绿色
            else:
                # 根据不同的NG类型设置不同的值（按优先级）
                # 优先级：Mark > Size > Ball Count > Ball_Area > Shift > 其他
                if "Mark" in defect_type:
                    full_slice[row, col] = 1  # Mark - 红色
                elif "Size" in defect_type:
                    full_slice[row, col] = 3  # Size - 紫色
                elif "Ball Count" in defect_type:
                    full_slice[row, col] = 4  # BallCount - 橙色
                elif "Ball_Area" in defect_type:
                    full_slice[row, col] = 5  # Ball_Area - 黄色
                elif "Shift" in defect_type:
                    full_slice[row, col] = 6  # Shift - 棕色
                elif "Scratch" in defect_type:
                    full_slice[row, col] = 7  #Scratch - 蓝色
                else:
                    full_slice[row, col] = 8  # 默认NG（如Scratch等）- 红色

        # 将修改后的切片写回full_value
        self.full_value[row_start:row_end, col_start:col_end] = full_slice

        # 将产品检测信息转换为日志格式并存储（仅对与 full_value 中 99 格对应的格写入；缺检记 Empty）
        for row, col in target_positions:
            product_info = slot[row][col]
            if product_info is None:
                self.accumulated_log_info.append(
                    _vacant_log_entry(len(self.accumulated_log_info) + 1, row, col)
                )
                continue
            log_entry = {}

            # 产品序号
            log_entry["product_index"] = len(self.accumulated_log_info) + 1

            # 尺寸信息
            size_result = product_info.get("size_result")
            if size_result and len(size_result) > 2:
                size_data = size_result[2]
                width_mm = size_data.get("width", None)
                height_mm = size_data.get("height", None)
                log_entry["size"] = (
                    [width_mm, height_mm]
                    if width_mm is not None and height_mm is not None
                    else [None, None]
                )
            else:
                log_entry["size"] = [None, None]

            # Mark信息
            mark_result = product_info.get("mark_result")
            if mark_result and len(mark_result) > 2:
                mark_data = mark_result[2]
                # 如果mark_result存在且is_valid为True，则表示检测到Mark（有Mark）
                log_entry["has_mark"] = mark_data.get("is_valid", False)
            else:
                log_entry["has_mark"] = False

            # 球信息
            ball_result = product_info.get("ball_result")
            ok_balls = []
            ng_balls = []
            if ball_result and len(ball_result) > 2:
                ball_data = ball_result[2]
                ok_details = ball_data.get("ok_details", [])
                ng_details = ball_data.get("ng_details", [])

                # 提取合格球信息
                for ball in ok_details:
                    ok_balls.append({
                        "radius_mm": ball.get("radius_mm"),
                        "area_mm2": ball.get("area_mm", 0.0)  # 使用area_mm作为area_mm2
                    })

                # 提取不合格球信息
                for ball in ng_details:
                    ng_balls.append({
                        "radius_mm": ball.get("radius_mm"),
                        "area_mm2": ball.get("area_mm", 0.0)  # 使用area_mm作为area_mm2
                    })

            log_entry["ok_balls"] = ok_balls
            log_entry["ng_balls"] = ng_balls

            # 偏移量信息
            shift_result = product_info.get("shift_result")
            if shift_result and len(shift_result) > 2:
                shift_data = shift_result[2]
                log_entry["shift_x"] = shift_data.get("shift_x", None)
                log_entry["shift_y"] = shift_data.get("shift_y", None)
            else:
                log_entry["shift_x"] = None
                log_entry["shift_y"] = None

            # 缺陷类型
            log_entry["defect_type"] = product_info.get("defect_type", ["OK"])

            # 添加到累积日志信息
            self.accumulated_log_info.append(log_entry)
        
        # 保存图像并更新计数
        self.image_dict[pos[0:2]] = current_image
        self.count += 1
        
        # 如果图像数量超过限制，清理最旧的图像（保留最近的数据）
        if len(self.image_dict) > self.max_history_size:
            # 简化处理：如果超过限制，删除最旧的一半
            num_to_remove = len(self.image_dict) - self.max_history_size // 2
            keys_to_remove = list(self.image_dict.keys())[:num_to_remove]
            for key in keys_to_remove:
                if key in self.image_dict:
                    del self.image_dict[key]
    
    def get_pos_image(self, pos):
        """获取指定位置的图像"""
        return self.image_dict.get(pos[0:2], None)
    
    def cleanup_old_data(self, max_size=None):
        """
        清理旧数据，保留最近的数据
        
        参数:
            max_size: 最大保留数量，如果为None则使用self.max_history_size
        """
        if max_size is None:
            max_size = self.max_history_size
        
        if len(self.image_dict) <= max_size:
            return
        
        # 获取所有位置，按count排序，删除最旧的
        # 简化处理：如果超过限制，删除最旧的一半
        num_to_remove = len(self.image_dict) - max_size // 2
        keys_to_remove = list(self.image_dict.keys())[:num_to_remove]
        for key in keys_to_remove:
            if key in self.image_dict:
                del self.image_dict[key]
    
    def get_full_animation(self):
        """生成完整的动画图像"""
        array = np.array(self.full_value)
        h, w = array.shape
        # 根据整体图像比例计算方块尺寸
        # 整体图像比例为 y:480, x:150
        # 例如：30行5列 -> 方块尺寸 16*30 (480/30=16, 150/5=30)
        margin = 2       # 小方块间隔
        block_height = 480 // h  # 每个小方块的高度（像素）
        block_width = 150 // w    # 每个小方块的宽度（像素）
        img_h = h * block_height + (h + 1) * margin
        img_w = w * block_width + (w + 1) * margin
        # 初始化底图（灰色背景）
        img = np.full((img_h, img_w, 3), 40, dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                y1 = margin + i * (block_height + margin)
                x1 = margin + j * (block_width + margin)
                y2 = y1 + block_height
                x2 = x1 + block_width
                if array[i, j] == 2:
                    color = (0, 255, 0)  # OK - 绿色
                elif array[i, j] == 1:
                    color = (0, 0, 255)  # Mark - 红色
                elif array[i, j] == 3:
                    color = (128, 0, 128)  # Size - 紫色
                elif array[i, j] == 4:
                    color = (0, 165, 255)  # BallCount - 橙色
                elif array[i, j] == 5:
                    color = (0, 255, 255)  # Ball_Area - 黄色
                elif array[i, j] == 6:
                    color = (42, 42, 165)  # Shift - 棕色
                elif array[i,j] == 7:
                    color = (255,0,0)    # Scartch - 蓝色
                elif array[i, j] == 8:
                    color = (0, 0, 255)  # NG - 红色
                elif array[i, j] == 99:
                    color = (255, 255, 255)  # 未检测到产品 - 白色
                elif array[i, j] == 0:
                    color = (0, 0, 0)  # 空白
                else:
                    color = (255, 255, 255)  # 其他 - 白色
                img[y1:y2, x1:x2] = color
        
        return img
    
    def get_log_info(self):
        """
        整理并返回检测日志数据
        
        Returns:
            dict: 包含检测流程信息、统计信息和产品详细信息的结构化字典
        """
        try:
            # 转换numpy对象
            accumulated_log_info = convert_numpy_obj(self.accumulated_log_info)
            
            # 转换开始时间为datetime对象
            start_time = None
            if self.strip_create_time:
                try:
                    start_time = dt.strptime(self.strip_create_time, "%Y%m%d%H%M%S")
                except:
                    start_time = dt.now()
            else:
                start_time = dt.now()
            
            # 计算结束时间和持续时间
            end_time = dt.now()
            duration_seconds = (end_time - start_time).total_seconds() if start_time else None
            
            # 统计NG总数和各类型不良数
            priority_map = {"Ball Count": 1, "Size": 2, "Ball_Area": 3, "Mark": 4, "Scratch": 5, "Shift": 6}
            ng_count_by_type = {"Size": 0, "Ball_Area": 0, "Ball Count": 0, "Mark": 0, "Scratch": 0, "Shift": 0}
            ng_total_count = 0
            
            # 收集统计数据
            width_list, height_list, all_ball_radii = [], [], []
            shift_x_list, shift_y_list = [], []
            
            for product in accumulated_log_info:
                defect_type = product.get("defect_type", ["OK"])
                # Empty 不计入 NG 统计
                is_vacant = _is_vacant_log_entry(product)
                is_ng = (
                    not is_vacant
                    and isinstance(defect_type, list)
                    and len(defect_type) > 0
                    and defect_type[0] != "OK"
                )
                
                if is_ng:
                    ng_total_count += 1
                    ng_types = [t for t in defect_type if t in ng_count_by_type]
                    if ng_types:
                        # 按优先级统计，只统计优先级最高的类型
                        primary_type = min(ng_types, key=lambda x: priority_map.get(x, 999))
                        ng_count_by_type[primary_type] += 1
                
                # 收集尺寸数据
                size = product.get("size", [None, None])
                if size and size[0] is not None and size[1] is not None:
                    width_list.append(float(size[0]))
                    height_list.append(float(size[1]))
                
                # 收集球半径数据
                for ball in product.get("ok_balls", []) + product.get("ng_balls", []):
                    radius = ball.get("radius_mm")
                    if radius is not None:
                        all_ball_radii.append(float(radius))
                
                # 收集偏移数据
                shift_x = product.get("shift_x", None)
                shift_y = product.get("shift_y", None)
                if shift_x is not None and isinstance(shift_x, (int, float)):
                    shift_x_list.append(float(shift_x))
                if shift_y is not None and isinstance(shift_y, (int, float)):
                    shift_y_list.append(float(shift_y))
            
            # 计算统计信息
            width_range = max(width_list) - min(width_list) if width_list else None
            height_range = max(height_list) - min(height_list) if height_list else None
            avg_width = sum(width_list) / len(width_list) if width_list else None
            avg_height = sum(height_list) / len(height_list) if height_list else None
            avg_ball_radius = sum(all_ball_radii) / len(all_ball_radii) if all_ball_radii else None
            
            shift_x_max = max(shift_x_list) if shift_x_list else None
            shift_x_min = min(shift_x_list) if shift_x_list else None
            shift_y_max = max(shift_y_list) if shift_y_list else None
            shift_y_min = min(shift_y_list) if shift_y_list else None
            
            # 计算CPK
            shift_x_cpk = None
            shift_y_cpk = None
            if shift_x_list and self.params.get('shift_check_enable', False):
                shift_x_tolerance = self.params.get('shift_x_tolerance', 0.5)
                shift_x_cpk = calculate_cpk(shift_x_list, shift_x_tolerance, -shift_x_tolerance)
            if shift_y_list and self.params.get('shift_check_enable', False):
                shift_y_tolerance = self.params.get('shift_y_tolerance', 0.5)
                shift_y_cpk = calculate_cpk(shift_y_list, shift_y_tolerance, -shift_y_tolerance)
            
            # 构建已启用检测项目列表
            detection_map = {
                'size_check_enable': '尺寸检测',
                'ball_check_enable': '锡球检测',
                'mark_check_enable': 'Mark检测',
                'scratch_check_enable': '划痕检测',
                'shift_check_enable': '偏移检测'
            }
            enabled_detections = [name for key, name in detection_map.items() if self.params.get(key, True)]
            
            # 构建产品列表
            product_list = []
            for product in accumulated_log_info:
                size = product.get("size", [None, None])
                defect_type = product.get("defect_type", ["OK"])
                ng_balls = product.get("ng_balls", [])
                is_vacant = _is_vacant_log_entry(product)
                # 如果defect_type[0]不是"OK"，则产品是NG（Empty 单独标记）
                is_ng_product = (
                    not is_vacant
                    and isinstance(defect_type, list)
                    and len(defect_type) > 0
                    and defect_type[0] != "OK"
                )
                # 提取所有NG类型（defect_type中的所有元素都是NG类型）
                ng_type_list = list(dict.fromkeys([t for t in defect_type if t != "OK"])) if isinstance(defect_type, list) else []
                
                # 构建NG球信息字符串
                ng_info_list = [
                    f"{b.get('radius_mm', 0):.4f},{b.get('area_mm2', 0):.4f}"
                    for b in ng_balls if b.get("radius_mm") is not None
                ]
                
                shift_x = product.get("shift_x", 0.0)
                shift_y = product.get("shift_y", 0.0)
                
                # 处理 None 值：如果 shift_x/shift_y 为 None，使用默认值 0.0
                if shift_x is None:
                    shift_x = 0.0
                if shift_y is None:
                    shift_y = 0.0
                
                row_pl = {
                    "product_index": product.get("product_index", ""),
                    "width": f"{size[0]:.4f}" if size and size[0] is not None else "",
                    "height": f"{size[1]:.4f}" if size and size[1] is not None else "",
                    "has_mark": "是" if product.get("has_mark", False) else "否",
                    "ng_ball_count": str(len(ng_balls)),
                    "ng_ball_info": ";".join(ng_info_list),
                    "shift_x": f"{shift_x:.4f}" if shift_x != 0.0 else "0.0000",
                    "shift_y": f"{shift_y:.4f}" if shift_y != 0.0 else "0.0000",
                    "is_ng": "否" if is_vacant else ("是" if is_ng_product else "否"),
                    "ng_types": "Empty" if is_vacant else ";".join(ng_type_list),
                }
                product_list.append(row_pl)
            
            # 构建返回字典
            # 计算总产品数：使用实际记录的产品数量
            total_products = len(accumulated_log_info)
            
            result = {
                "lot_id": self.strip_lot,
                "sn_id": self.strip_sn,
                "process_info": {
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else "",
                    "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "duration_seconds": duration_seconds,
                    "total_products": total_products,
                    "ng_total": ng_total_count,
                    "enabled_detections": enabled_detections
                },
                "statistics": {
                    "width_range": width_range,
                    "height_range": height_range,
                    "avg_width": avg_width,
                    "avg_height": avg_height,
                    "avg_ball_radius": avg_ball_radius,
                    "shift_x_max": shift_x_max,
                    "shift_x_min": shift_x_min,
                    "shift_y_max": shift_y_max,
                    "shift_y_min": shift_y_min,
                    "shift_x_cpk": shift_x_cpk,
                    "shift_y_cpk": shift_y_cpk
                },
                "defect_statistics": {
                    "Size": ng_count_by_type.get("Size", 0),
                    "Ball_Area": ng_count_by_type.get("Ball_Area", 0),
                    "Ball Count": ng_count_by_type.get("Ball Count", 0),
                    "Mark": ng_count_by_type.get("Mark", 0),
                    "Scratch": ng_count_by_type.get("Scratch", 0),
                    "Shift": ng_count_by_type.get("Shift", 0)
                },
                "product_list": product_list
            }
            
            return result
            
        except Exception as e:
            print(f"整理日志信息错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_statistics_info(self):
        """
        获取实时统计信息（轻量级版本，用于实时更新）
        
        Returns:
            dict: 包含统计信息的字典
                - station: 工位名称 ("干燥台")
                - lot_id: Lot ID
                - total_count: 总检测数（含 Empty 占位条）
                - empty_count: 缺检/未检出占位条数（defect_type[0]=="Empty"，见 write 中 _vacant_log_entry）
                - ng_count: NG总数（不含 Empty）
                - yield_rate: 良率 (%)
                - defect_counts: 各缺陷类型统计字典（不含 Empty；Empty 用 empty_count）
        """
        try:
            # 转换numpy对象
            accumulated_log_info = convert_numpy_obj(self.accumulated_log_info)
            
            # 统计NG总数和各类型不良数
            priority_map = {"Ball Count": 1, "Size": 2, "Ball_Area": 3, "Mark": 4, "Scratch": 5, "Shift": 6}
            ng_count_by_type = {"Size": 0, "Ball_Area": 0, "Ball Count": 0, "Mark": 0, "Scratch": 0, "Shift": 0}
            ng_total_count = 0
            total_count = len(accumulated_log_info)
            empty_count = 0
            
            for product in accumulated_log_info:
                defect_type = product.get("defect_type", ["OK"])
                is_vacant = _is_vacant_log_entry(product)
                if is_vacant:
                    empty_count += 1
                is_ng = (
                    not is_vacant
                    and isinstance(defect_type, list)
                    and len(defect_type) > 0
                    and defect_type[0] != "OK"
                )
                
                if is_ng:
                    ng_total_count += 1
                    ng_types = [t for t in defect_type if t in ng_count_by_type]
                    if ng_types:
                        primary_type = min(ng_types, key=lambda x: priority_map.get(x, 999))
                        ng_count_by_type[primary_type] += 1
            
            # 计算良率（Empty 不计入 NG，仍计入 total_count）
            yield_rate = ((total_count - ng_total_count) / total_count * 100) if total_count > 0 else 0.0
            
            return {
                "station": "干燥台" if self.station == "dry" else "移栽台",
                "lot_id": self.strip_lot if hasattr(self, 'strip_lot') else "",
                "total_count": total_count,
                "empty_count": empty_count,
                "ng_count": ng_total_count,
                "yield_rate": yield_rate,
                "defect_counts": {
                    "Mark": ng_count_by_type.get("Mark", 0),
                    "Size": ng_count_by_type.get("Size", 0),
                    "Ball_Area": ng_count_by_type.get("Ball_Area", 0),
                    "Ball Count": ng_count_by_type.get("Ball Count", 0),
                    "Scratch": ng_count_by_type.get("Scratch", 0),
                    "Shift": ng_count_by_type.get("Shift", 0)
                }
            }
            
        except Exception as e:
            print(f"获取统计信息错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "station": "干燥台" if self.station == "dry" else "移栽台",
                "lot_id": "",
                "total_count": 0,
                "empty_count": 0,
                "ng_count": 0,
                "yield_rate": 0.0,
                "defect_counts": {
                    "Mark": 0,
                    "Size": 0,
                    "Ball_Area": 0,
                    "Ball Count": 0,
                    "Scratch": 0,
                    "Shift": 0
                }
            }


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


def draw_detection_results(image_result: np.ndarray, product_info: dict, 
                          mark_color: str = "red"):
    """
    在图像上绘制检测结果（通用方法）
    
    Args:
        image_result: 结果图像（BGR格式或灰度图）
        product_info: 产品信息字典，包含检测结果元组，格式为：
            {
                "size_result": tuple 或 None,  # (success, msg, result_dict)
                "ball_result": tuple 或 None,  # (success, msg, result_dict)
                "mark_result": tuple 或 None,  # (success, msg, result_dict)
                "scratch_result": tuple 或 None,  # (success, msg, result_dict)
                "shift_result": tuple 或 None,  # (success, msg, result_dict)
            }
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
    
    ####尺寸绘制####
    if product_info.get("size_result") is not None:
        try:
            size_result = product_info["size_result"]
            # 支持两种格式：tuple (success, msg, result_dict) 或直接是dict
            if isinstance(size_result, tuple) and len(size_result) >= 3:
                box_points = size_result[2].get("box_points", [])
                is_valid = size_result[2].get("is_valid", False)
            elif isinstance(size_result, dict):
                box_points = size_result.get("box_points", [])
                is_valid = size_result.get("is_valid", False)
            else:
                box_points = []
                is_valid = False
            
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
                            COLOR_GREEN if is_valid else COLOR_RED, 4)
        except Exception as e:
            error_messages.append(f"绘制尺寸结果错误: {e}")
    
    ####ball绘制####
    if product_info.get("ball_result") is not None:
        try:
            ball_result = product_info["ball_result"]
            # 支持两种格式
            if isinstance(ball_result, tuple) and len(ball_result) >= 3:
                ok_details = ball_result[2].get("ok_details", [])
                ng_details = ball_result[2].get("ng_details", [])
            elif isinstance(ball_result, dict):
                ok_details = ball_result.get("ok_details", [])
                ng_details = ball_result.get("ng_details", [])
            else:
                ok_details = []
                ng_details = []
            
            for detail in ok_details:
                try:
                    box = detail.get("box", [0, 0, 0, 0])
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    else:
                        x, y, w, h = 0, 0, 0, 0
                    if check_image_bounds(image_result, x, y, w, h):
                        cv.rectangle(image_result, (x, y), (x+w, y+h), COLOR_GREEN, 2)
                except Exception as e:
                    error_messages.append(f"绘制OK球结果错误: {e}")
            
            for detail in ng_details:
                try:
                    box = detail.get("box", [0, 0, 0, 0])
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    else:
                        x, y, w, h = 0, 0, 0, 0
                    if check_image_bounds(image_result, x, y, w, h):
                        cv.rectangle(image_result, (x, y), (x+w, y+h), COLOR_RED, 3)
                except Exception as e:
                    error_messages.append(f"绘制NG球结果错误: {e}")
        except Exception as e:
            error_messages.append(f"获取球检测结果错误: {e}")
    
    ####mark绘制####
    if product_info.get("mark_result") is not None:
        try:
            mark_result = product_info["mark_result"]
            # 支持两种格式
            if isinstance(mark_result, tuple) and len(mark_result) >= 3:
                mark_contour = mark_result[2].get("mark_contour", None)
            elif isinstance(mark_result, dict):
                mark_contour = mark_result.get("mark_contour", None)
            else:
                mark_contour = None
            
            if mark_contour is not None:
                # mark_contour可能是单个轮廓或轮廓列表
                if isinstance(mark_contour, list) and len(mark_contour) > 0:
                    # 检查第一个元素是否是轮廓（numpy数组）
                    if isinstance(mark_contour[0], np.ndarray):
                        # 是轮廓列表
                        for contour in mark_contour:
                            cv.drawContours(image_result, [contour], -1, mark_draw_color, 3)
                    else:
                        # 可能是单个轮廓
                        cv.drawContours(image_result, [mark_contour], -1, mark_draw_color, 3)
                elif isinstance(mark_contour, np.ndarray):
                    # 单个轮廓
                    cv.drawContours(image_result, [mark_contour], -1, mark_draw_color, 3)
            elif isinstance(mark_result, tuple) and len(mark_result) >= 3:
                per_roi = mark_result[2].get("per_roi") or []
                for pr in per_roi:
                    c = pr.get("mark_contour") if isinstance(pr, dict) else None
                    if c is not None:
                        cv.drawContours(image_result, [c], -1, mark_draw_color, 3)
        except Exception as e:
            error_messages.append(f"绘制Mark结果错误: {e}")
    
    ####scratch绘制####
    if product_info.get("scratch_result") is not None:
        try:
            scratch_result = product_info["scratch_result"]
            # 支持两种格式
            if isinstance(scratch_result, tuple) and len(scratch_result) >= 3:
                scratch_contours = scratch_result[2].get("ng_scratch_contours", [])
            elif isinstance(scratch_result, dict):
                scratch_contours = scratch_result.get("ng_scratch_contours", [])
            else:
                scratch_contours = []
            
            if scratch_contours:
                for contour in scratch_contours:
                    cv.drawContours(image_result, [contour], -1, COLOR_RED, 2)
        except Exception as e:
            error_messages.append(f"绘制划痕结果错误: {e}")
    
    ####shift绘制####
    if product_info.get("shift_result") is not None:
        try:
            shift_result = product_info["shift_result"]
            # 支持两种格式
            if isinstance(shift_result, tuple) and len(shift_result) >= 3:
                shift_dict = shift_result[2]
            elif isinstance(shift_result, dict):
                shift_dict = shift_result
            else:
                shift_dict = {}
            
            is_valid_shift = shift_dict.get("is_valid", False)
            ball_center = shift_dict.get("ball_center", None)
            size_center = shift_dict.get("size_center", None)
            
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
                                 COLOR_GREEN if is_valid_shift else COLOR_RED, 3, tipLength=10)
                    cv.line(image_result, (size_x, size_y), (ball_x, ball_y), COLOR_GREEN if is_valid_shift else COLOR_RED, 3,cv.LINE_8)
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
        tuple: (success: bool, msg: str, product_info: dict)
            - success: bool 是否成功执行（False表示检测失败）
            - msg: str 消息（错误信息或成功信息）
            - product_info: dict 产品信息字典，包含以下键：
                - "defect_type": list 缺陷类型列表，如 ["OK"] 或 ["NG", "Mark"]
                - "product_image_result": np.ndarray 产品图像结果（原始图像副本）
                - "size_result": tuple 或 None 尺寸检测结果 (success, msg, result_dict)
                - "ball_result": tuple 或 None 锡球检测结果 (success, msg, result_dict)
                - "mark_result": tuple 或 None Mark检测结果 (success, msg, result_dict)
                - "shift_result": tuple 或 None 偏移检测结果 (success, msg, result_dict)
                - "scratch_result": tuple 或 None 划痕检测结果 (success, msg, result_dict)
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
    
    # 初始化结果字典
    product_info = {
        "defect_type": ["OK"],
        "product_image_result": None,
        "size_result": None,
        "ball_result": None,
        "mark_result": None,
        "shift_result": None,
        "scratch_result": None,
    }

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
            error_msg = "mark_detector 未提供"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        mark_detect_result = mark_detector.detect(image_for_detection)
        if not mark_detect_result[0]:  # 检测失败
            error_msg = f"Mark检测失败: {mark_detect_result[1]}"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        product_info["mark_result"] = mark_detect_result
        
        # 获取allow_mark参数（默认False，保持向后兼容）
        allow_mark = params.get("allow_mark", False)
        is_valid = mark_detect_result[2]["is_valid"]
        
        # 根据allow_mark参数决定判定逻辑
        if allow_mark:
            # allow_mark == True: 检测到Mark判定为OK，未检测到Mark判定为NG
            if not is_valid:
                # 未检测到Mark，判定为NG
                if "OK" in product_info["defect_type"]:
                    product_info["defect_type"].remove("OK")
                product_info["defect_type"].append("Mark")
                
                # 如果启用提前返回，检测到NG时立即返回
                if early_return_on_ng:
                    product_info["product_image_result"] = image.copy()
                    return True, "成功", product_info
            # 检测到Mark，判定为OK，继续后续检测
        else:
            # allow_mark == False（默认）: 检测到Mark判定为NG，未检测到Mark判定为OK
            if is_valid:
                # 检测到Mark，判定为NG
                if "OK" in product_info["defect_type"]:
                    product_info["defect_type"].remove("OK")
                product_info["defect_type"].append("Mark")
                
                # 如果启用提前返回，检测到NG时立即返回
                if early_return_on_ng:
                    product_info["product_image_result"] = image.copy()
                    return True, "成功", product_info
            # 未检测到Mark，判定为OK，继续后续检测
    
    # 尺寸检测
    if size_check_enable:
        size_detector = detectors.get("size_detector")
        if size_detector is None:
            error_msg = "size_detector 未提供"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        size_detect_result = size_detector.detect(image_for_detection)
        if not size_detect_result[0]:  # 检测失败
            error_msg = f"Size检测失败: {size_detect_result[1]}"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        product_info["size_result"] = size_detect_result
        if not size_detect_result[2]["is_valid"]:
            # 尺寸不合格，判定为NG
            if "OK" in product_info["defect_type"]:
                product_info["defect_type"].remove("OK")
            product_info["defect_type"].append("Size")
            
            # 如果启用提前返回，检测到NG时立即返回
            if early_return_on_ng:
                product_info["product_image_result"] = image.copy()
                return True, "成功", product_info
    
    # 锡球检测
    if ball_check_enable:
        ball_detector = detectors.get("ball_detector")
        if ball_detector is None:
            error_msg = "ball_detector 未提供"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        ball_detect_result = ball_detector.detect(image_for_detection)
        if not ball_detect_result[0]:  # 检测失败
            error_msg = f"Ball检测失败: {ball_detect_result[1]}"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        product_info["ball_result"] = ball_detect_result
        if not ball_detect_result[2]["is_valid"]:
            # 锡球不合格，判定为NG
            if "OK" in product_info["defect_type"]:
                product_info["defect_type"].remove("OK")
            
            # 判断是数量问题还是质量问题
            # 从ball_detector的参数中获取期望数量
            expected_count = ball_detector.params.get("expected_ball_count", 0)
            if ball_detect_result[2].get("ball_count", 0) != expected_count:
                product_info["defect_type"].append("Ball Count")
            else:
                product_info["defect_type"].append("Ball_Area")
            
            # 如果启用提前返回，检测到NG时立即返回
            if early_return_on_ng:
                product_info["product_image_result"] = image.copy()
                return True, "成功", product_info
    
    # 偏移检测（需要ball_result和size_result）
    if shift_check_enable and product_info["ball_result"] is not None and product_info["size_result"] is not None:
        shift_detector = detectors.get("shift_detector")
        if shift_detector is None:
            error_msg = "shift_detector 未提供"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        # shift_detector.detect()期望接收dict参数，从tuple中提取result_dict
        ball_result_dict = product_info["ball_result"][2]
        size_result_dict = product_info["size_result"][2]
        shift_detect_result = shift_detector.detect(ball_result_dict, size_result_dict)
        
        if not shift_detect_result[0]:  # 检测失败
            error_msg = f"Shift检测失败: {shift_detect_result[1]}"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        product_info["shift_result"] = shift_detect_result
        if not shift_detect_result[2]["is_valid"]:
            # 偏移不合格，判定为NG
            if "OK" in product_info["defect_type"]:
                product_info["defect_type"].remove("OK")
            product_info["defect_type"].append("Shift")
            
            # 如果启用提前返回，检测到NG时立即返回
            if early_return_on_ng:
                product_info["product_image_result"] = image.copy()
                return True, "成功", product_info
    
    # 划痕检测
    if scratch_check_enable:
        scratch_detector = detectors.get("scratch_detector")
        if scratch_detector is None:
            error_msg = "scratch_detector 未提供"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        scratch_detect_result = scratch_detector.detect(image_for_detection)
        if not scratch_detect_result[0]:  # 检测失败
            error_msg = f"Scratch检测失败: {scratch_detect_result[1]}"
            if error_callback:
                error_callback(error_msg)
            return False, error_msg, product_info
        
        product_info["scratch_result"] = scratch_detect_result
        if not scratch_detect_result[2]["is_valid"]:
            # 划痕不合格，判定为NG
            if "OK" in product_info["defect_type"]:
                product_info["defect_type"].remove("OK")
            product_info["defect_type"].append("Scratch")
            
            # 如果启用提前返回，检测到NG时立即返回
            if early_return_on_ng:
                product_info["product_image_result"] = image.copy()
                return True, "成功", product_info
    
    # 更新总体判定
    if len(product_info["defect_type"]) > 1:
        product_info["defect_type"][0] = "NG"
    
    # 生成产品图像结果（用于显示）
    product_info["product_image_result"] = image.copy()
    
    return True, "成功", product_info


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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Size_Result:
    width: float = 0.0
    height: float = 0.0
    box_points: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # [x, y, w, h] 像素

    is_valid: bool = False
    error_code: int = 0  # 0=检测成功（is_valid 表合格/NG），非0=检测失败/边界无效
    error_msg: Optional[str] = None


@dataclass
class Mark_Result:
    mark_contour: list = field(default_factory=list)  # list[np.ndarray]，全局坐标
    mark_count: int = 0
    mark_position: list = field(default_factory=list)  # list[Tuple[int,int]]
    mark_area: list = field(default_factory=list)  # list[float] 各 ROI 面积（像素）

    is_valid: bool = False
    error_code: int = 0
    error_msg: Optional[str] = None


@dataclass
class Ball_Result:
    ball_contour: list = field(default_factory=list)
    product_center_based_on_balls: Tuple[float, float] = (0.0, 0.0)
    ball_position: list = field(default_factory=list)
    ball_count: int = 0
    ball_area: list = field(default_factory=list)
    ball_area_mm: list = field(default_factory=list)
    ball_radius: list = field(default_factory=list)
    ball_radius_mm: list = field(default_factory=list)
    ball_average_area: float = 0.0
    ball_average_area_mm: float = 0.0
    ball_average_radius: float = 0.0
    ball_average_radius_mm: float = 0.0

    ng_ball_contour: list = field(default_factory=list)
    ng_ball_position: list = field(default_factory=list)
    ng_ball_count: int = 0
    ng_ball_area: list = field(default_factory=list)
    ng_ball_area_mm: list = field(default_factory=list)
    ng_ball_radius: list = field(default_factory=list)
    ng_ball_radius_mm: list = field(default_factory=list)

    is_valid: bool = False
    error_code: int = 0
    error_msg: Optional[str] = None

    def ng_ball_info_text(self) -> str:
        return ";".join(
            f"{r:.4f},{a:.4f}"
            for r, a in zip(self.ng_ball_radius_mm, self.ng_ball_area_mm)
            if r is not None
        )

    def all_ball_radius_mm(self) -> list:
        return [r for r in self.ball_radius_mm + self.ng_ball_radius_mm if r]


@dataclass
class Shift_Result:
    shift_x: float = 0.0
    shift_y: float = 0.0
    shift_x_mm: float = 0.0
    shift_y_mm: float = 0.0
    ball_center: Tuple[float, float] = (0.0, 0.0)  # 用于绘制偏移向量
    size_center: Tuple[float, float] = (0.0, 0.0)

    is_valid: bool = False
    error_code: int = 0
    error_msg: Optional[str] = None


@dataclass
class Scratch_Result:
    scratch_contour: list = field(default_factory=list)  # NG 划痕轮廓（全局坐标）
    scratch_position: list = field(default_factory=list)
    scratch_count: int = 0
    scratch_area: float = 0.0
    scratch_area_mm: float = 0.0

    is_valid: bool = True  # True=无划痕缺陷
    error_code: int = 0
    error_msg: Optional[str] = None


@dataclass
class Product:
    product_type: str = ""
    product_position: list = field(default_factory=list)  # [x, y, w, h] 整帧像素包围盒
    product_status: str = ""
    product_image: Optional[np.ndarray] = None  # 该产品原图裁剪
    product_image_result: Optional[np.ndarray] = None  # 绘制检测框后的结果图

    defect_type: list = field(default_factory=lambda: ["OK"])
    std_width: float = 0.0
    std_height: float = 0.0
    size_tolerance_x: float = 0.0
    size_tolerance_y: float = 0.0
    std_ball_radius: float = 0.0
    ball_radius_tolerance: float = 0.0

    size_result: Size_Result = field(default_factory=Size_Result)
    mark_result: Mark_Result = field(default_factory=Mark_Result)
    ball_result: Ball_Result = field(default_factory=Ball_Result)
    shift_result: Shift_Result = field(default_factory=Shift_Result)
    scratch_result: Scratch_Result = field(default_factory=Scratch_Result)
    animation_color: int = 0  # 状态码 0/2/1/3/4/5/6/7（与 Modbus/动画一致）

    is_valid: bool = True


@dataclass
class Bga_Strip:
    station: str = ""
    strip_lot: str = ""
    strip_sn: str = ""
    strip_create_time: str = ""
    strip_side: str = "front"
    strip_cols: int = 0
    strip_rows: int = 0
    window_rows: int = 0
    window_cols: int = 0
    product_array: Optional[np.ndarray] = None
    strip_image: Optional[np.ndarray] = None
    animation_image: Optional[np.ndarray] = None

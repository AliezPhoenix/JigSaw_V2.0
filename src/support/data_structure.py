from dataclasses import dataclass
from typing import Tuple

import numpy as np

@dataclass
class Size_Result:
    width: float
    height: float

    is_valid:bool

@dataclass
class Mark_Result:
    mark_contour: list[np.ndarray]
    mark_count: int
    mark_position: list[Tuple[int,int]]
    mark_area: list[float]

    is_valid: bool

@dataclass
class Ball_Result:
    ball_contour: list[np.ndarray]
    product_center_based_on_balls: Tuple[int, int]
    ball_position: list[Tuple[int, int]]
    ball_count: int
    ball_area: list[float]
    ball_area_mm: list[float]
    ball_radius: list[float]
    ball_radius_mm: list[float]
    ball_average_area: float
    ball_average_area_mm: float
    ball_average_radius: float
    ball_average_radius_mm: float

    ng_ball_contour: list[np.ndarray]
    ng_ball_position: list[Tuple[int, int]]
    ng_ball_count: int
    ng_ball_area: list[float]
    ng_ball_area_mm: list[float]
    ng_ball_radius: list[float]
    ng_ball_radius_mm: list[float]

    is_valid: bool

    def ng_ball_info_text(self) -> str:
        return ";".join(
            f"{r:.4f},{a:.4f}"
            for r, a in zip(self.ng_ball_radius_mm, self.ng_ball_area_mm)
            if r is not None
        )

    def all_ball_radius_mm(self) -> list[float]:
        return [r for r in self.ball_radius_mm + self.ng_ball_radius_mm if r]

@dataclass
class Shift_Result:
    shift_x: float
    shift_y: float
    shift_x_mm: float
    shift_y_mm: float


    is_valid: bool

@dataclass
class Scratch_Result:
    scratch_contour: list[np.ndarray]
    scratch_position: list[Tuple[int,int]]
    scratch_count: int
    scratch_area: float
    scratch_area_mm: float
    is_valid: bool

@dataclass
class Product:
    product_type: str
    product_position: list[int]
    product_status: str
    product_image: np.ndarray
    
    
    defect_type: list[str]
    std_width: float
    std_height: float
    size_tolerance_x: float
    size_tolerance_y: float
    std_ball_radius: float
    ball_radius_tolerance: float


    size_result: Size_Result
    mark_result: Mark_Result
    ball_result: Ball_Result
    shift_result: Shift_Result
    scratch_result: Scratch_Result
    animation_color:tuple[int,int,int]
    
    
    is_valid: bool

@dataclass
class Bga_Strip:
    station: str
    strip_lot: str
    strip_sn: str
    strip_create_time: str
    strip_side: str
    strip_cols: int
    strip_rows: int
    product_array: np.ndarray
    strip_image: np.ndarray
    animation_image: np.ndarray
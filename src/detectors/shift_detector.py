from . import *
from src.support.data_structure import Ball_Result, Shift_Result, Size_Result


# ==================== ShiftDetector 类 ====================

class ShiftDetector:
    def __init__(self, params: dict = None):
        """
        初始化 Shift 检测器
        
        Args:
            params: 参数字典，包含以下键：
                - pixel_size: float 像素尺寸（mm/pixel），默认0.001
                - error_correction_factor: float 误差修正系数，默认0.7，用于消除误差
        """

        # 默认参数
        self.params = {
            "pixel_size": 0.001,
            "error_correction_factor": 0.7,
            "allow_tolerance_x": 0.0,
            "allow_tolerance_y": 0.0
        }
        if params:
            self.params.update(params)
        
    def detect(self, ball_detection_result: Ball_Result, size_detection_result: Size_Result):
        """
        执行 Shift 偏移检测
        
        算法：
        1. 从 BallDetector 检测结果中提取所有球的中心点
        2. 计算所有球中心的平均位置作为产品中心
        3. 从 SizeDetector 检测结果中提取尺寸中心位置
        4. 偏移量 = 球中心平均位置 - 尺寸中心位置
        
        Args:
            ball_detection_result: BallDetector.detect() 的返回结果字典，包含：
                - ok_details: list 合格球的详细信息列表，每个元素包含 "center" 键
                - ng_details: list 不合格球的详细信息列表，每个元素包含 "center" 键
            size_detection_result: SizeDetector.detect() 的返回结果字典，包含：
                - box_points: list 边界框坐标 [x, y, w, h]，其中 x, y 为左上角坐标，w, h 为宽度和高度（像素）
        
        Returns:
            tuple: (success, msg, result_dict)
                - success: bool 是否检测成功
                - msg: str 消息
                - result_dict: dict 结果字典，包含以下键：
                    - is_valid: bool 产品是否合格
                    - shift_x: float X方向的偏移量（mm）
                    - shift_y: float Y方向的偏移量（mm）

        """
        try:
            # 获取参数
            pixel_size = self.params.get('pixel_size', 0.001)
            error_correction_factor = self.params.get('error_correction_factor', 0.7)
            allow_tolerance_x = self.params.get('allow_tolerance_x', 0.0)
            allow_tolerance_y = self.params.get('allow_tolerance_y', 0.0)
                        
            # 从 BallDetector 结果中提取所有球的中心点（合格 + 不合格）
            ball_centers = []
            for center in list(ball_detection_result.ball_position) + list(ball_detection_result.ng_ball_position):
                ball_centers.append((float(center[0]), float(center[1])))

            # 从 SizeDetector 结果中提取尺寸中心
            box_points = size_detection_result.box_points

            # 从 [x, y, w, h] 格式计算尺寸中心
            if box_points is None or len(box_points) < 4:
                raise ValueError("box_points 格式无效")
            
            x_min_bound = float(box_points[0])
            y_min_bound = float(box_points[1])
            x_max_bound = x_min_bound + float(box_points[2])
            y_max_bound = y_min_bound + float(box_points[3])
            size_center_x = (x_min_bound + x_max_bound) / 2.0
            size_center_y = (y_min_bound + y_max_bound) / 2.0
            
            # 检查是否有球中心点
            if len(ball_centers) == 0:
                raise ValueError("未检测到任何球的中心点")
            
            # 将球中心转换为numpy数组便于处理
            ball_centers_array = np.array(ball_centers)
            
            # 计算所有球中心的平均位置（产品中心）
            ball_center_x = float(np.mean(ball_centers_array[:, 0]))
            ball_center_y = float(np.mean(ball_centers_array[:, 1]))
            
            # 计算偏移量（像素单位）
            shift_x_pixel = ball_center_x - size_center_x
            shift_y_pixel = ball_center_y - size_center_y
            
            # 转换为实际单位（mm）并应用误差修正系数
            shift_x = shift_x_pixel * pixel_size * error_correction_factor
            shift_y = shift_y_pixel * pixel_size * error_correction_factor
            
            is_valid = not (abs(shift_x) > allow_tolerance_x or abs(shift_y) > allow_tolerance_y)
            return Shift_Result(
                shift_x=shift_x,
                shift_y=shift_y,
                shift_x_mm=shift_x,
                shift_y_mm=shift_y,
                ball_center=(ball_center_x, ball_center_y),
                size_center=(size_center_x, size_center_y),
                is_valid=is_valid,
            )

        except Exception as e:
            return Shift_Result(
                error_code=2, error_msg=f"偏移检测异常: {str(e)}", is_valid=False
            )
    
    def update_params(self, params: dict):
        """
        更新检测器参数
        Args:
            params: dict 要更新的参数字典，可以包含以下键：
                - pixel_size: float 像素尺寸（mm/pixel）
                - error_correction_factor: float 误差修正系数
                - allow_tolerance_x: float X方向容差（mm）
                - allow_tolerance_y: float Y方向容差（mm）
        
        Returns:
            bool: 更新是否成功
        """
        
        # 更新参数
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
        return True
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            dict: 当前参数字典的副本
        """
        return self.params.copy()


from . import *
from src.support.support_funs import ensure_gray_u8


# ==================== BallDetector 类 ====================

class BallDetector:
    def __init__(self, params: dict = None):
        self.image = None
        
        # 默认参数
        self.params = {
            "min_threshold": 0,
            "max_threshold": 255,
            "ball_area_min_threshold": 0,
            "ball_area_max_threshold": 10000,
            "ball_radius_tolerance": 0.0,  # mm
            "std_radius": 0.0,  # mm
            "pixel_size": 0.001,  # mm/pixel
            "expected_ball_count": 0,
            "ball_search_roi": [0, 0, 0, 0]  # (x, y, w, h)
        }
        if params:
            self.params.update(params)
        
    def detect(self, image):
        """
        执行锡球检测
        
        Args:
            image: 输入的灰度图像或BGR图像
        
        Returns:
            dict: 检测结果字典，包含以下键：
                - error_code: int 错误码
                    0: OK（合格）
                    1: NG（不合格）
                    2: 检测失败
                - is_valid: bool 是否合格（所有球都合格且数量正确）
                - ball_count: int 检测到的球数量
                - ok_details: list 合格球的详细信息列表
                - ng_details: list 不合格球的详细信息列表
                - avg_radius: float 平均半径（像素）
        """
        image_gray = ensure_gray_u8(image, copy=True)
        self.image = image_gray
        
        # 获取参数
        min_threshold = self.params.get('min_threshold', 0)
        max_threshold = self.params.get('max_threshold', 255)
        ball_area_min_threshold = self.params.get('ball_area_min_threshold', 0)
        ball_area_max_threshold = self.params.get('ball_area_max_threshold', 10000)
        ball_radius_tolerance_mm = self.params.get('ball_radius_tolerance', 0.0)
        std_radius_mm = self.params.get('std_radius', 0.0)
        pixel_size = self.params.get('pixel_size', 0.001)
        expected_ball_count = self.params.get('expected_ball_count', 0)
        ball_search_roi = self.params.get('ball_search_roi', [0, 0, 0, 0])
        
        # 验证参数
        if len(ball_search_roi) < 4:
            return False, "锡球搜索区域无效", {'is_valid': False, 'ball_count': 0, 'ok_details': [], 'ng_details': [], 'avg_radius': 0.0}
        
        # 将mm单位转换为像素单位
        ball_radius_tolerance_pixel = ball_radius_tolerance_mm / pixel_size if pixel_size > 0 else 0.0
        std_radius_pixel = std_radius_mm / pixel_size if pixel_size > 0 else 0.0
        
        # 图像预处理：中值滤波
        image_gray_blur = cv.medianBlur(image_gray, 3)
        
        # 二值化
        image_binary_ball = cv.inRange(image_gray_blur, min_threshold, max_threshold)
        
        # 查找轮廓
        contours, hierarchy = cv.findContours(image_binary_ball, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        # 初始化变量
        filtered_ball_image = []
        contour_count = 0
        total_area = 0
        
        # 预计算产品边界范围（留5像素边距），用于判断锡球是否在产品范围内
        x_min_bound = ball_search_roi[0] + 5
        y_min_bound = ball_search_roi[1] + 5
        x_max_bound = ball_search_roi[0] + ball_search_roi[2] - 5
        y_max_bound = ball_search_roi[1] + ball_search_roi[3] - 5
        
        # 遍历所有轮廓，筛选出有效的锡球
        for contour in contours:
            area = cv.contourArea(contour)
            
            # 提前进行面积检查，避免不必要的moments计算
            if not (ball_area_min_threshold <= area <= ball_area_max_threshold):
                continue
            
            # 计算轮廓的质心坐标
            try:
                contour_center = cv.moments(contour)
                m00 = contour_center['m00']
                if m00 == 0:
                    continue
                contour_center_x = int(contour_center['m10'] / m00)
                contour_center_y = int(contour_center['m01'] / m00)
            except (ZeroDivisionError, KeyError):
                continue
            
            # 判断轮廓中心是否在产品范围内（留5像素边距）
            if (contour_center_x > x_min_bound and contour_center_y > y_min_bound and
                contour_center_x < x_max_bound and contour_center_y < y_max_bound):
                # 获取轮廓的边界框，并保存相关信息
                box = cv.boundingRect(contour)
                filtered_ball_image.append((
                    image_gray[box[1]:box[1]+box[3], box[0]:box[0]+box[2]],
                    contour,
                    box,
                    area
                ))
                contour_count += 1
                total_area += area
        
        # 计算平均面积（避免除零错误）
        avg_area = total_area / contour_count if contour_count > 0 else 0
        
        # 初始化结果列表
        ok_details = []
        ng_details = []
        radius_list = []
        all_ball_details = []  # 临时存储所有球的详细信息
        
        # 检测每个锡球
        for item in filtered_ball_image:
            ball, contour, box, area = item
            
            # 计算最小外接圆半径
            (center_x, center_y), radius = cv.minEnclosingCircle(contour)
            radius_list.append(radius)
            
            # 计算面积偏差
            area_diff_pixel = abs(area - avg_area)
            
            # 计算半径偏差（像素单位）
            radius_diff_pixel = abs(radius - std_radius_pixel)
            
            # 转换为mm单位
            radius_mm = radius * pixel_size
            area_mm = area * pixel_size * pixel_size
            radius_diff_mm = radius_diff_pixel * pixel_size
            area_diff_mm = area_diff_pixel * pixel_size * pixel_size
            
            # 创建详细信息字典（先不包含avg_radius_mm）
            ball_detail = {
                "box": box,  # (x, y, w, h)
                "center": (int(center_x), int(center_y)),
                "radius_pixel": float(radius),
                "radius_mm": radius_mm,
                "area_pixel": float(area),
                "area_mm": area_mm,
                "radius_diff_mm": radius_diff_mm,
                "area_diff_mm": area_diff_mm,
                "is_ok": radius_diff_pixel <= ball_radius_tolerance_pixel  # 临时标记是否合格
            }
            
            all_ball_details.append(ball_detail)
        
        # 计算最终平均半径（像素和mm）
        avg_radius = np.mean(radius_list) if len(radius_list) > 0 else 0.0
        avg_radius_mm = avg_radius * pixel_size
        
        # 为所有球的详细信息添加avg_radius_mm，并分类到ok_details和ng_details
        for ball_detail in all_ball_details:
            ball_detail["avg_radius_mm"] = avg_radius_mm
            # 移除临时标记
            is_ok = ball_detail.pop("is_ok")
            if is_ok:
                ok_details.append(ball_detail)
            else:
                ng_details.append(ball_detail)
        
        # 判断是否合格
        # 条件1: 所有球都合格
        # 条件2: 检测到的球数量等于期望数量（如果期望数量>0）
        is_valid = False
        
        if contour_count == 0:
            # 没有检测到球
            is_valid = False
        elif len(ng_details) == 0:
            # 所有球都合格
            if expected_ball_count > 0:
                # 需要检查数量
                if contour_count == expected_ball_count:
                    is_valid = True
                else:
                    is_valid = False
            else:
                # 不检查数量，所有球合格即可
                is_valid = True
        else:
            # 有不合格的球
            is_valid = False
        
        # 返回结果
        return True, "锡球检测完成", {
            'is_valid': is_valid,
            'ball_count': contour_count,
            'ok_details': ok_details, 
            'ng_details': ng_details,
            'avg_radius': float(avg_radius)
        } 
    
    def update_params(self, params: dict):
        """
        更新检测器参数
        
        Args:
            params: dict 要更新的参数字典，可以包含以下键：
                - min_threshold: int 二值化下阈值（0-255）
                - max_threshold: int 二值化上阈值（0-255）
                - ball_area_min_threshold: int 最小面积（像素）
                - ball_area_max_threshold: int 最大面积（像素）
                - ball_radius_tolerance: float 半径容差（mm）
                - std_radius: float 标准半径（mm）
                - pixel_size: float 像素尺寸（mm/pixel）
                - expected_ball_count: int 期望球数量
                - ball_search_roi: list[int, int, int, int] 搜索区域 [x, y, w, h]
        
        Returns:
            bool: 更新是否成功
        
        Example:
            >>> detector.update_params({
            ...     'min_threshold': 100,
            ...     'max_threshold': 200,
            ...     'ball_area_min_threshold': 1000,
            ...     'ball_area_max_threshold': 2000,
            ...     'ball_radius_tolerance': 0.05,
            ...     'std_radius': 0.5,
            ...     'expected_ball_count': 100,
            ...     'ball_search_roi': [0, 0, 100, 100]
            ... })
        """
        # 验证参数有效性
        valid_keys = {
            'min_threshold', 'max_threshold', 'ball_area_min_threshold', 'ball_area_max_threshold',
            'ball_radius_tolerance', 'std_radius', 'pixel_size', 
            'expected_ball_count', 'ball_search_roi'
        }
        
        invalid_keys = []
        for key in params.keys():
            if key not in valid_keys:
                invalid_keys.append(key)
        
        # 验证数值参数的有效性
        validation_errors = []
        
        if 'min_threshold' in params:
            val = params['min_threshold']
            if not isinstance(val, (int, float)) or val < 0 or val > 255:
                validation_errors.append(f"min_threshold 必须在 [0, 255] 范围内，当前值: {val}")
        
        if 'max_threshold' in params:
            val = params['max_threshold']
            if not isinstance(val, (int, float)) or val < 0 or val > 255:
                validation_errors.append(f"max_threshold 必须在 [0, 255] 范围内，当前值: {val}")
        
        if 'min_threshold' in params and 'max_threshold' in params:
            if params['min_threshold'] > params['max_threshold']:
                validation_errors.append(f"min_threshold ({params['min_threshold']}) 不能大于 max_threshold ({params['max_threshold']})")
        
        if 'ball_area_min_threshold' in params:
            val = params['ball_area_min_threshold']
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(f"ball_area_min_threshold 必须是非负数，当前值: {val}")
        
        if 'ball_area_max_threshold' in params:
            val = params['ball_area_max_threshold']
            if not isinstance(val, (int, float)) or val <= 0:
                validation_errors.append(f"ball_area_max_threshold 必须是正数，当前值: {val}")
        
        if 'ball_area_min_threshold' in params and 'ball_area_max_threshold' in params:
            if params['ball_area_min_threshold'] > params['ball_area_max_threshold']:
                validation_errors.append(f"ball_area_min_threshold ({params['ball_area_min_threshold']}) 不能大于 ball_area_max_threshold ({params['ball_area_max_threshold']})")
        
        if 'ball_radius_tolerance' in params:
            val = params['ball_radius_tolerance']
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(f"ball_radius_tolerance 必须是非负数，当前值: {val}")
        
        if 'std_radius' in params:
            val = params['std_radius']
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(f"std_radius 必须是非负数，当前值: {val}")
        
        if 'pixel_size' in params:
            val = params['pixel_size']
            if not isinstance(val, (int, float)) or val <= 0:
                validation_errors.append(f"pixel_size 必须是正数，当前值: {val}")
        
        if 'expected_ball_count' in params:
            val = params['expected_ball_count']
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(f"expected_ball_count 必须是非负整数，当前值: {val}")
            else:
                params['expected_ball_count'] = int(val)
        
        if 'ball_search_roi' in params:
            val = params['ball_search_roi']
            if not isinstance(val, (tuple, list)) or len(val) < 4:
                validation_errors.append(f"ball_search_roi 必须是长度为4的元组或列表，当前值: {val}")
            elif not all(isinstance(v, (int, float)) and v >= 0 for v in val[:4]):
                validation_errors.append(f"ball_search_roi 的元素必须是非负数，当前值: {val}")
        
        # 如果有验证错误，记录并返回失败
        if validation_errors:
            error_msg = "参数验证失败:\n" + "\n".join(f"  - {err}" for err in validation_errors)
            return False
        
        # 更新参数（只更新有效键）
        updated_keys = []
        for key in valid_keys:
            if key in params:
                self.params[key] = params[key]
                updated_keys.append(key)
        
        return True
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            dict: 当前参数字典的副本
        """
        return self.params.copy()


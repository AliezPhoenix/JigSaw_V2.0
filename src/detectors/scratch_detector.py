from . import *
from src.support.support_funs import mask_roi_regions, ensure_gray_u8
from src.support.data_structure import Scratch_Result


class ScratchDetector:
    def __init__(self, params: dict = None):
        """
        初始化 ScratchDetector
        
        Args:
            params: 参数字典，包含以下键：
                - min_threshold: int 二值化处理的最小阈值
                - max_threshold: int 二值化处理的最大阈值
                - scratch_length_threshold: float 划痕长度阈值（mm），超过此长度的划痕将被判定为不良
                - pixel_size: float 像素尺寸（mm/pixel）
                - scratch_roi: list 划痕检测区域列表，每个元素为 (x, y, w, h) 格式，默认空列表
                - roi_blocks: list 需要屏蔽的区域列表，每个元素为 (x, y, w, h) 格式，默认None
                - aspect_ratio_threshold: float 长宽比阈值，默认3.0
                - min_contour_area: int 最小轮廓面积（像素），用于过滤太小的轮廓，默认10
        """
        # 默认参数
        self.params = {
            "min_threshold": 0,
            "max_threshold": 255,
            "scratch_length_threshold": 0.0,  # mm
            "pixel_size": 0.001,  # mm/pixel
            "scratch_roi": [],  # 划痕检测区域列表
            "roi_blocks": None,  # 需要屏蔽的区域列表
            "aspect_ratio_threshold": 3.0,  # 长宽比阈值
            "min_contour_area": 10  # 最小轮廓面积（像素）
        }
        if params:
            self.params.update(params)
    
    def detect(self, image):
        """
        执行划痕检测
        
        Args:
            image: 输入的灰度图像或BGR图像
        
        Returns:
            tuple: (success: bool, msg: str, result_dict: dict)
                - success: bool 是否检测成功
                - msg: str 消息
                - result_dict: dict 结果字典，包含以下键：
                    - is_valid: bool 是否检测到划痕缺陷（False表示有缺陷）
                    - ng_scratch_contours: list NG的划痕轮廓列表，每个元素为numpy数组，用于绘制
        """
        try:
            image_gray = ensure_gray_u8(image, copy=True)
            
            # 获取参数
            min_threshold = self.params.get('min_threshold', 0)
            max_threshold = self.params.get('max_threshold', 255)
            scratch_length_threshold = self.params.get('scratch_length_threshold', 0.0)
            pixel_size = self.params.get('pixel_size', 0.001)
            scratch_roi = self.params.get('scratch_roi', [])
            roi_blocks = self.params.get('roi_blocks', None)
            aspect_ratio_threshold = self.params.get('aspect_ratio_threshold', 3.0)
            min_contour_area = self.params.get('min_contour_area', 10)
            
            # 屏蔽ROI区域（如果提供了roi_blocks）
            if roi_blocks is not None and isinstance(roi_blocks, list) and len(roi_blocks) > 0:
                image_gray = mask_roi_regions(image_gray, roi_blocks)
            
            # 检查是否有划痕检测区域
            if not isinstance(scratch_roi, list) or len(scratch_roi) == 0:
                return Scratch_Result(is_valid=True)
            
            # 存储检测到的NG划痕轮廓
            ng_scratch_contours = []
            
                
            x, y, w, h = int(scratch_roi[0]), int(scratch_roi[1]), int(scratch_roi[2]), int(scratch_roi[3])
            
            # 边界检查
            h_img, w_img = image_gray.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            
            # 提取ROI区域
            roi_image = image_gray[y:y+h, x:x+w]
            
            # 使用阈值进行二值化处理
            # 划痕通常是暗色的，所以使用inRange检测低灰度值区域
            roi_binary = cv.inRange(roi_image, min_threshold, max_threshold)
            
            # 形态学操作：去除小的噪声点
            # 使用开运算（先腐蚀后膨胀）去除小点
            kernel_small = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            roi_binary = cv.morphologyEx(roi_binary, cv.MORPH_OPEN, kernel_small)
            
            # 使用闭运算（先膨胀后腐蚀）连接断开的划痕
            kernel_line = cv.getStructuringElement(cv.MORPH_RECT, (5, 2))  # 水平方向的线形核
            roi_binary = cv.morphologyEx(roi_binary, cv.MORPH_CLOSE, kernel_line)
            kernel_line_vertical = cv.getStructuringElement(cv.MORPH_RECT, (2, 5))  # 垂直方向的线形核
            roi_binary = cv.morphologyEx(roi_binary, cv.MORPH_CLOSE, kernel_line_vertical)
            
            # 查找轮廓
            contours, hierarchy = cv.findContours(roi_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # 筛选划痕轮廓
            for contour in contours:
                # 计算轮廓面积
                area = cv.contourArea(contour)
                if area < min_contour_area:  # 过滤太小的轮廓
                    continue
                
                # 计算轮廓的最小外接矩形
                rect = cv.minAreaRect(contour)
                width, height = rect[1]
                if width < height:
                    width, height = height, width  # 确保width是长边
                
                # 计算长宽比，划痕通常是细长的
                aspect_ratio = width / height if height > 0 else 0
                
                # 使用最小外接矩形的长度作为划痕长度
                scratch_length_pixels = max(width, height)
                scratch_length_mm = scratch_length_pixels * pixel_size
                
                # 判断是否为划痕：长宽比大于阈值 或 长度超过阈值
                # 划痕特征：细长（长宽比大）或长度超过阈值
                if aspect_ratio > aspect_ratio_threshold or scratch_length_mm > scratch_length_threshold:
                    # 如果长度超过阈值，判定为不良并保存轮廓
                    if scratch_length_mm > scratch_length_threshold:
                        # 将轮廓坐标转换回原图坐标系
                        contour_global = contour + np.array([x, y], dtype=np.int32)
                        ng_scratch_contours.append(contour_global)
            
            # 判断结果
            is_valid = len(ng_scratch_contours) == 0
            total_area = float(sum(cv.contourArea(c) for c in ng_scratch_contours))
            return Scratch_Result(
                scratch_contour=ng_scratch_contours,
                scratch_count=len(ng_scratch_contours),
                scratch_area=total_area,
                scratch_area_mm=total_area * pixel_size * pixel_size,
                is_valid=is_valid,
            )

        except Exception as e:
            return Scratch_Result(
                error_code=2, error_msg=f"划痕检测异常: {str(e)}", is_valid=False
            )
    
    def update_params(self, params: dict):
        """
        更新检测器参数
        
        Args:
            params: dict 要更新的参数字典，可以包含以下键：
                - min_threshold: int 二值化下阈值（0-255）
                - max_threshold: int 二值化上阈值（0-255）
                - scratch_length_threshold: float 划痕长度阈值（mm）
                - pixel_size: float 像素尺寸（mm/pixel）
                - scratch_roi: list 划痕检测区域列表
                - roi_blocks: list 需要屏蔽的区域列表
                - aspect_ratio_threshold: float 长宽比阈值
                - min_contour_area: int 最小轮廓面积（像素）
        
        Returns:
            bool: 更新是否成功
        """
        if not isinstance(params, dict):
            return False
        
        # 验证参数有效性
        if 'min_threshold' in params:
            val = params['min_threshold']
            if not isinstance(val, (int, float)) or val < 0 or val > 255:
                return False
        
        if 'max_threshold' in params:
            val = params['max_threshold']
            if not isinstance(val, (int, float)) or val < 0 or val > 255:
                return False
        
        if 'min_threshold' in params and 'max_threshold' in params:
            if params['min_threshold'] > params['max_threshold']:
                return False
        
        if 'scratch_length_threshold' in params:
            val = params['scratch_length_threshold']
            if not isinstance(val, (int, float)) or val < 0:
                return False
        
        if 'pixel_size' in params:
            val = params['pixel_size']
            if not isinstance(val, (int, float)) or val <= 0:
                return False
        
        if 'aspect_ratio_threshold' in params:
            val = params['aspect_ratio_threshold']
            if not isinstance(val, (int, float)) or val <= 0:
                return False
        
        if 'min_contour_area' in params:
            val = params['min_contour_area']
            if not isinstance(val, (int, float)) or val < 0:
                return False
        
        # 更新参数
        self.params.update(params)
        return True
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            dict: 当前参数字典的副本
        """
        return self.params.copy()

from . import *


# ==================== MarkDetector 类 ====================

class MarkDetector:
    def __init__(self, params: dict = None):
        """
        初始化 Mark 检测器
        
        Args:
            params: 参数字典，包含以下键：
                - min_threshold: int 二值化处理的最小阈值（手动模式）
                - max_threshold: int 二值化处理的最大阈值
                - min_mark_area: int Mark区域的最小面积（像素）
                - auto_threshold_factor: float 自动阈值因子，默认1.05
                - pixel_size: float 像素尺寸（mm/pixel），默认0.001
                - mark_detect_mode: "auto"或"manual"，默认"manual"
        """

        self.image = None
        
        # 默认参数
        self.params = {
            "min_threshold": 0,
            "max_threshold": 255,
            "min_mark_area": 0,
            "auto_threshold_factor": 1.05,
            "pixel_size": 0.001,
            "mark_detect_mode": "manual",  # "auto"=自动模式, "manual"=手动模式
            "mark_roi": []
        }
        if params:
            self.params.update(params)
        
    def detect(self, image):
        """
        执行 Mark 标记检测
        
        Args:
            image: 输入的灰度图像或BGR图像
        
        Returns:
            tuple: (success, msg, result_dict)
                - success: bool 是否检测成功
                - msg: str 消息
                - result_dict: dict 结果字典，包含以下键：
                    - is_valid: bool 是否检测到Mark
                    - mark_contour: np.ndarray 标记区域轮廓，如果未检测到则为None
                    - mark_area: float 标记区域面积（像素）
                    - mark_area_mm: float 标记区域面积（mm²）
        """
        if len(image.shape) == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        
        self.image = image_gray

        # 获取第一个ROI区域
        roi = self.params.get('mark_roi', [])
        
        # 边界检查
        h, w = image_gray.shape
        x, y, roi_w, roi_h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = max(1, min(roi_w, w - x))
        roi_h = max(1, min(roi_h, h - y))
        
        # 提取Mark区域
        try:
            image_gray_mark = image_gray[y:y+roi_h, x:x+roi_w]

        except Exception as e:
            return False,"提取Mark ROI区域失败: {e}",{'is_valid': False,"mark_contour":None,"mark_area":0.0,"mark_area_mm":0.0}

        
        # 获取参数
        min_threshold = self.params.get('min_threshold', 0)
        max_threshold = self.params.get('max_threshold', 255)
        min_mark_area = self.params.get('min_mark_area', 0)
        auto_threshold_factor = self.params.get('auto_threshold_factor', 1.05)
        mark_detect_mode = self.params.get('mark_detect_mode', "manual")
        pixel_size = self.params.get('pixel_size', 0.001)
        
        # 根据模式选择阈值
        if mark_detect_mode == "auto":
            # 自动模式：使用平均值的倍数作为阈值下限
            mean_val = int(cv.mean(image_gray_mark)[0] * auto_threshold_factor)
            image_binary_mark = cv.inRange(image_gray_mark, mean_val, max_threshold)
        else:
            # 手动模式：使用指定的阈值
            image_binary_mark = cv.inRange(image_gray_mark, min_threshold, max_threshold)
        
        # 去除杂点：使用形态学开运算去除细小的杂点（先腐蚀后膨胀）
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        image_binary_mark = cv.morphologyEx(image_binary_mark, cv.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, hierarchy = cv.findContours(image_binary_mark, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        # 查找满足面积要求的轮廓
        mark_contour = None
        mark_area = 0.0
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area > min_mark_area:
                mark_contour = contour
                mark_area = area
                break  # 找到第一个满足条件的轮廓即可
        
        # 判断结果
        if mark_contour is not None and mark_area > 0:
            # 将轮廓坐标从ROI坐标系映射回原图坐标系
            # 轮廓坐标是相对于ROI的，需要加上ROI的偏移量(x, y)
            mark_contour_global = mark_contour + np.array([x, y], dtype=np.int32)
            
            # 检测到Mark
            mark_area_mm = mark_area * pixel_size * pixel_size
            return True,"检测到Mark",{'is_valid': True,"mark_contour":mark_contour_global,"mark_area":float(mark_area),"mark_area_mm":float(mark_area_mm)}
        else:
            # 未检测到Mark
            return True,"未检测到Mark",{'is_valid': False,"mark_contour":None,"mark_area":0.0,"mark_area_mm":0.0}
    
    def update_params(self, params: dict):
        """
        更新检测器参数
        
        Args:
            params: dict 要更新的参数字典，可以包含以下键：
                - min_threshold: int 二值化下阈值（手动模式）
                - max_threshold: int 二值化上阈值
                - min_mark_area: int Mark区域的最小面积（像素）
                - auto_threshold_factor: float 自动阈值因子
                - pixel_size: float 像素尺寸（mm/pixel）
                - mark_detect_mode: "auto"或"manual"
        
        Returns:
            tuple: (success, msg)
                - success: bool 是否更新成功
                - msg: str 消息
        """
        # 更新参数
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
        
        return True,"参数更新成功"
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            dict: 当前参数字典的副本
        """
        return self.params.copy()


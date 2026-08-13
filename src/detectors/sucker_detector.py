import cv2 as cv
import numpy as np
from src.support.support_funs import ensure_gray_u8


class SuckerDetector:
    def __init__(self, params: dict = None):
        """
        初始化 SuckerDetector
        
        Args:
            params: 参数字典，包含以下键：
                - min_threshold_sucker: int 吸嘴检测二值化下阈值
                - max_threshold_sucker: int 吸嘴检测二值化上阈值
                - min_threshold_product: int 产品检测二值化下阈值
                - max_threshold_product: int 产品检测二值化上阈值
                - min_area_sucker: int 吸嘴最小面积（像素）
                - min_area_product: int 产品最小面积（像素）
                - pixel_size: float 像素尺寸（mm/pixel）
        """
        # 默认参数
        self.params = {
            "min_threshold_sucker": 240,
            "max_threshold_sucker": 255,
            "min_threshold_product": 150,
            "max_threshold_product": 255,
            "min_area_sucker": 10000,
            "min_area_product": 30000,
            "pixel_size": 0.001,  # mm/pixel
        }
        if params:
            self.params.update(params)
    
    def detect(self, image, mode):
        """
        执行吸嘴或产品检测
        
        Args:
            image: 输入的灰度图像或BGR图像
            mode: str 检测模式，"sucker" 或 "product"
        
        Returns:
            tuple: (success: bool, msg: str, result_dict: dict)
                - success: bool 是否检测成功
                - msg: str 消息
                - result_dict: dict 结果字典，包含以下键：
                    - is_valid: bool 是否检测到目标
                    - dis_x: int X方向偏移量（微米）
                    - dis_y: int Y方向偏移量（微米）
                    - angle: int 角度（毫度）
                    - sucker_box_point: tuple 吸嘴边界框 (x, y, w, h) 或 None
                    - product_box_point: tuple 产品边界框 (x, y, w, h) 或 None
        """
        image_gray = ensure_gray_u8(image, copy=True)
        
        if mode == "sucker":
            return self._detect_sucker(image_gray)
        elif mode == "product":
            return self._detect_product(image_gray)
        else:
            return False, f"未知的检测模式: {mode}", {
                'is_valid': False,
                'dis_x': 0,
                'dis_y': 0,
                'angle': 0,
                'sucker_box_point': None,
                'product_box_point': None
            }
    
    def _detect_sucker(self, image_gray):
        """
        Args:
            image_gray: 灰度图像
        
        Returns:
            tuple: (success: bool, msg: str, result_dict: dict)
        """
        try:
            # 二值化
            _, image_thresh = cv.threshold(
                image_gray,
                self.params["min_threshold_sucker"],
                self.params["max_threshold_sucker"],
                cv.THRESH_BINARY
            )
            
            # 形态学开运算
            image_open = cv.morphologyEx(
                image_thresh,
                cv.MORPH_OPEN,
                cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
            )
            
            # 查找轮廓
            image_contours, _ = cv.findContours(
                image_open,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )
            
            # 图像中心
            image_center = (int(image_gray.shape[1] / 2), int(image_gray.shape[0] / 2))
            
            # 遍历轮廓查找吸嘴
            for i in range(len(image_contours)):
                sucker_rect = cv.minAreaRect(image_contours[i])
                sucker_area = cv.contourArea(image_contours[i])
                distance = abs(sucker_rect[0][1] - image_center[1])
                
                # 判断是否为吸嘴：距离图像中心Y方向<100像素且面积>最小面积
                if distance < 100 and sucker_area > self.params["min_area_sucker"]:
                    bounding_box = cv.boundingRect(image_contours[i])
                    sucker_center = (int(sucker_rect[0][0]), int(sucker_rect[0][1]))
                    
                    # 计算偏移量（微米）
                    dis_x = int((sucker_center[0] - image_center[0]) * self.params["pixel_size"] * 1000)
                    dis_y = int((sucker_center[1] - image_center[1]) * self.params["pixel_size"] * 1000)
                    
                    # 计算角度（毫度）
                    angle = sucker_rect[2]
                    if angle > 45:
                        angle = angle - 90
                    angle = int(angle * 1000)
                    
                    return True, "吸嘴检测成功", {
                        'is_valid': True,
                        'dis_x': dis_x,
                        'dis_y': dis_y,
                        'angle': angle,
                        'sucker_box_point': bounding_box,  # (x, y, w, h)
                        'product_box_point': None
                    }
            
            # 未检测到吸嘴
            return True, "未检测到吸嘴", {
                'is_valid': False,
                'dis_x': 0,
                'dis_y': 0,
                'angle': 0,
                'sucker_box_point': None,
                'product_box_point': None
            }
            
        except Exception as e:
            return False, f"吸嘴检测异常: {str(e)}", {
                'is_valid': False,
                'dis_x': 0,
                'dis_y': 0,
                'angle': 0,
                'sucker_box_point': None,
                'product_box_point': None
            }
    
    def _detect_product(self, image_gray):
        """
        Args:
            image_gray: 灰度图像
        
        Returns:
            tuple: (success: bool, msg: str, result_dict: dict)
        """
        try:
            # 第一步：检测吸嘴区域
            _, image_thresh = cv.threshold(
                image_gray,
                self.params["min_threshold_sucker"],
                self.params["max_threshold_sucker"],
                cv.THRESH_BINARY
            )
            
            # 形态学开运算
            image_open = cv.morphologyEx(
                image_thresh,
                cv.MORPH_OPEN,
                cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
            )
            
            # 查找轮廓
            image_contours, _ = cv.findContours(
                image_open,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )
            
            # 图像中心
            image_center = (int(image_gray.shape[1] / 2), int(image_gray.shape[0] / 2))
            
            # 查找吸嘴区域并裁剪产品图像
            image_product = None
            sucker_box = None
            x, y, w, h = 0, 0, 0, 0
            
            for i in range(len(image_contours)):
                sucker_rect = cv.minAreaRect(image_contours[i])
                sucker_area = cv.contourArea(image_contours[i])
                distance = abs(sucker_rect[0][1] - image_center[1])
                
                # 判断是否为吸嘴区域
                if distance < 100 and sucker_area > self.params["min_area_sucker"]:
                    x, y, w, h = cv.boundingRect(image_contours[i])
                    sucker_box = (x, y, w, h)
                    # 裁剪产品图像
                    image_product = image_gray[y:y+h, x:x+w].copy()
                    break
            
            # 如果未找到吸嘴区域，返回失败
            if image_product is None or image_product.size == 0:
                return True, "未检测到吸嘴区域，无法检测产品", {
                    'is_valid': False,
                    'dis_x': 0,
                    'dis_y': 0,
                    'angle': 0,
                    'sucker_box_point': sucker_box,
                    'product_box_point': None
                }
            
            # 第二步：对产品图像进行二值化（反转）
            _, image_product_thresh = cv.threshold(
                image_product,
                self.params["min_threshold_product"],
                self.params["max_threshold_product"],
                cv.THRESH_BINARY_INV
            )
            
            # 确保数据类型为 uint8
            if image_product_thresh.dtype != np.uint8:
                image_product_thresh = image_product_thresh.astype(np.uint8)
            
            # 查找产品轮廓
            image_product_contours, _ = cv.findContours(
                image_product_thresh,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )
            
            # 找到最大面积的轮廓作为产品
            max_product_area = 0
            max_index = -1
            
            for i in range(len(image_product_contours)):
                product_area = cv.contourArea(image_product_contours[i])
                if product_area > max_product_area:
                    max_product_area = product_area
                    max_index = i
            
            # 如果未找到产品轮廓，返回失败
            if max_index == -1:
                return True, "未检测到产品轮廓", {
                    'is_valid': False,
                    'dis_x': 0,
                    'dis_y': 0,
                    'angle': 0,
                    'sucker_box_point': sucker_box,
                    'product_box_point': None
                }
            
            # 计算产品边界框和中心
            product_rect = cv.minAreaRect(image_product_contours[max_index])
            bounding_box_local = cv.boundingRect(image_product_contours[max_index])
            
            # 转换为全局坐标
            product_box = (
                bounding_box_local[0] + x,
                bounding_box_local[1] + y,
                bounding_box_local[2],
                bounding_box_local[3]
            )
            
            # 产品中心（局部坐标）
            product_center = (int(product_rect[0][0]), int(product_rect[0][1]))
            
            # 产品图像中心（局部坐标）
            image_product_center = (
                int(image_product.shape[1] / 2),
                int(image_product.shape[0] / 2)
            )
            
            # 计算偏移量（微米）
            # 注意：Defect_3 中是 image_product_center - product_center
            dis_x = int((image_product_center[0] - product_center[0]) * self.params["pixel_size"] * 1000)
            dis_y = int((image_product_center[1] - product_center[1]) * self.params["pixel_size"] * 1000)
            
            # 计算角度（毫度）
            angle = product_rect[2]
            if angle > 45:
                angle = angle - 90
            angle = int(angle * 1000)
            
            return True, "产品检测成功", {
                'is_valid': True,
                'dis_x': dis_x,
                'dis_y': dis_y,
                'angle': angle,
                'sucker_box_point': sucker_box,
                'product_box_point': product_box  # (x, y, w, h)
            }
            
        except Exception as e:
            return False, f"产品检测异常: {str(e)}", {
                'is_valid': False,
                'dis_x': 0,
                'dis_y': 0,
                'angle': 0,
                'sucker_box_point': None,
                'product_box_point': None
            }
    
    def update_params(self, params: dict):
        """
        更新检测器参数
        
        Args:
            params: dict 要更新的参数字典
        
        Returns:
            bool: 更新是否成功
        """
        if not isinstance(params, dict):
            return False
        
        self.params.update(params)
        return True
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            dict: 当前参数字典的副本
        """
        return self.params.copy()

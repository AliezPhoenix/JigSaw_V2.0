from math import prod
import sys
from pathlib import Path

from sympy import product

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.support import support_funs
import os 
import cv2 as cv
from src.support.support_funs import ensure_gray_u8
import numpy as np


# ==================== 辅助函数 ====================

def smooth_gradient(grad, smooth_window=3):
    """平滑梯度曲线，过滤频繁剧烈变化的区域"""
    if len(grad) == 0 or smooth_window <= 1:
        return grad
    
    # 确保窗口大小是奇数
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    if len(grad) <= smooth_window:
        return grad
    
    # 使用移动平均进行平滑
    smoothed = np.zeros_like(grad)
    half_window = smooth_window // 2
    
    # 中间部分：使用完整的移动平均
    for i in range(half_window, len(grad) - half_window):
        smoothed[i] = np.mean(grad[i - half_window:i + half_window + 1])
    
    # 边界部分：使用逐渐增大/减小的窗口
    for i in range(half_window):
        smoothed[i] = np.mean(grad[:i + half_window + 1])
    for i in range(len(grad) - half_window, len(grad)):
        smoothed[i] = np.mean(grad[i - half_window:])
    
    return smoothed

def calculate_subpixel_offset(grad_values, min_idx, method='parabola'):
    """
    计算亚像素偏移量
    
    Args:
        grad_values: 梯度值数组
        min_idx: 最小值索引
        method: 插值方法，'parabola'（抛物线拟合，精度高）或'linear'（线性插值，速度快）
    
    Returns:
        亚像素偏移量（-1到1之间）
    """
    if min_idx <= 0 or min_idx >= len(grad_values) - 1:
        return 0.0
    
    g0, g1, g2 = grad_values[min_idx - 1], grad_values[min_idx], grad_values[min_idx + 1]
    
    if method == 'parabola':
        # 抛物线拟合: g(x) = ax^2 + bx + c，求极值点位置
        a = (g2 - 2*g1 + g0) / 2.0
        b = (g2 - g0) / 2.0
        if abs(a) > 1e-6:
            offset = -b / (2 * a)
            return np.clip(offset, -1.0, 1.0)
        # a接近0时回退到线性插值
        method = 'linear'
    
    if method == 'linear':
        # 线性插值
        if abs(g2 - g0) > 1e-6:
            offset = (g1 - g0) / (g2 - g0) - 0.5
            return np.clip(offset, -0.5, 0.5)
    
    return 0.0

def detect_boundary_subpixel(proj_curve, need_reverse=False, dx=1.0, smooth_window=3):
    """
    检测边界点（亚像素精度）
    检测逻辑：由产品（白）向背景（黑）进行检测，查找负梯度最小值
    """
    if len(proj_curve) == 0:
        return None
    
    # 计算并平滑梯度
    grad = np.gradient(proj_curve, dx)
    if smooth_window > 1:
        grad = smooth_gradient(grad, smooth_window)
    
    # 查找负梯度最小值位置（边界处应该是下降趋势）
    grad_negative = grad.copy()
    grad_negative[grad_negative > 0] = np.inf
    min_idx = np.argmin(grad_negative) if np.any(grad < 0) else np.argmax(np.abs(grad))
    
    # 计算亚像素偏移（优先使用抛物线拟合，失败则使用线性插值）
    offset = calculate_subpixel_offset(grad, min_idx, method='parabola')
    if abs(offset) > 0.8:
        offset = calculate_subpixel_offset(grad, min_idx, method='linear')
    
    # 计算最终边界位置
    boundary_pos = float(min_idx + offset)
    if need_reverse:
        boundary_pos = len(proj_curve) - 1 - boundary_pos
    
    return boundary_pos

def calculate_projection_curve(roi_image, direction='horizontal'):
    """计算投影曲线"""
    if direction == 'horizontal':
        return [np.sum(roi_image[h_idx, :]) / roi_image.shape[1] for h_idx in range(roi_image.shape[0])]
    else:  # vertical
        return [np.sum(roi_image[:, w_idx]) / roi_image.shape[0] for w_idx in range(roi_image.shape[1])]

def detect_peak_subpixel(curve, need_reverse=False, smooth_window=5):
    """
    在一维曲线中查找主峰位置（亚像素）
    适用于频域增强后的边界响应曲线。
    """
    if len(curve) == 0:
        return None

    curve_arr = np.asarray(curve, dtype=np.float32)
    if need_reverse:
        curve_arr = curve_arr[::-1]

    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        curve_arr = np.convolve(curve_arr, kernel, mode='same')

    peak_idx = int(np.argmax(curve_arr))
    offset = calculate_subpixel_offset(curve_arr, peak_idx, method='parabola')
    if abs(offset) > 0.8:
        offset = calculate_subpixel_offset(curve_arr, peak_idx, method='linear')

    boundary_pos = float(peak_idx + offset)
    if need_reverse:
        boundary_pos = len(curve) - 1 - boundary_pos
    return boundary_pos

def build_frequency_edge_maps(image_float, low_cut_ratio=0.03):
    """
    通过频域方向滤波生成边界响应图：
    - horizontal_edge_map: 强调水平边界（top/bottom）
    - vertical_edge_map: 强调垂直边界（left/right）
    """
    h, w = image_float.shape
    dft = cv.dft(image_float, flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])

    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    u = xx - cx
    v = yy - cy
    radius = np.sqrt(u * u + v * v)
    low_cut = max(2.0, min(h, w) * float(low_cut_ratio))

    high_pass = (radius >= low_cut).astype(np.float32)
    vertical_selector = (np.abs(u) >= np.abs(v)).astype(np.float32)
    horizontal_selector = (np.abs(v) >= np.abs(u)).astype(np.float32)

    def reconstruct(mask_2d):
        mask_2d = (high_pass * mask_2d).astype(np.float32)
        mask = np.repeat(mask_2d[:, :, np.newaxis], 2, axis=2)
        filtered_shift = dft_shift * mask
        filtered = np.fft.ifftshift(filtered_shift, axes=[0, 1])
        spatial = cv.idft(filtered, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        spatial = np.abs(spatial)
        spatial = cv.GaussianBlur(spatial, (5, 5), 0)
        return cv.normalize(spatial, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    vertical_edge_map = reconstruct(vertical_selector)
    horizontal_edge_map = reconstruct(horizontal_selector)

    dft_mag = cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    dft_mag = np.log1p(dft_mag)
    dft_mag = cv.normalize(dft_mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    return horizontal_edge_map, vertical_edge_map, dft_mag

# ==================== SizeDetector 类 ====================

class SizeDetector:
    def __init__(self, params: dict = None):
        self.image = None
        self.detection_result = None  # 存储检测结果
        
        # 默认参数
        self.params = {
            "min_threshold": 0,
            "max_threshold": 255,
            "allow_tolerance_x": 0.0,  
            "allow_tolerance_y": 0.0,
            "roi_width": 50,
            "std_size": (0.0, 0.0),  # 标准产品尺寸 (width, height) mm单位
            "pixel_size": 0.001  # 像素尺寸（mm/pixel）
        }
        if params:
            self.params.update(params)
        
    def detect(self, image):
        """
        执行尺寸检测
        
        Args:
            image: 输入的灰度图像或BGR图像
        Returns:
            dict: 包含以下键的字典
                - error_code: int 错误码
                    0: OK（合格）
                    1: NG（不合格）
                    2: 检测失败/边界无效
                - is_valid: bool 尺寸是否合格
                - box_points: list 边界框坐标 [x, y, w, h]，其中 x, y 为左上角坐标，w, h 为宽度和高度（像素）
                - width: float 产品宽度（mm）
                - height: float 产品高度（mm）
        """

        image_gray = ensure_gray_u8(image, copy=True)
        self.image = image_gray
        h, w = image_gray.shape
        # 频域方向滤波提取边界响应
        image_float = image_gray.astype(np.float32)
        horizontal_edge_map, vertical_edge_map, dft_mag = build_frequency_edge_maps(image_float)
        image_binary = cv.max(horizontal_edge_map, vertical_edge_map)

        cv.namedWindow("image_dft",cv.WINDOW_NORMAL)
        cv.namedWindow("image_binary_debug",cv.WINDOW_NORMAL)
        cv.imshow("image_binary_debug",image_binary)
        cv.imshow("image_dft", dft_mag)
        
        # 默认参数：差分步长0.7，平滑窗口5
        gradient_dx = 0.7
        smooth_window = 5
        
        # 自适应ROI宽度：根据图像尺寸调整
        roi_width = max(30, min(80, int(min(h, w) * 0.1)))
        if 'roi_width' in self.params and self.params['roi_width'] > 0:
            roi_width = self.params['roi_width']
        
        # 定义4个ROI区域
        rois = {
            'top': (0, 0, w, roi_width),
            'bottom': (0, max(0, h - roi_width), w, roi_width),
            'left': (0, 0, roi_width, h),
            'right': (max(0, w - roi_width), 0, roi_width, h)
        }
        
        # 对每个ROI进行边界检测
        boundaries = {}
        for roi_name, (roi_x, roi_y, roi_w, roi_h) in rois.items():
            is_horizontal = roi_name in ['top', 'bottom']
            is_reverse = roi_name in ['top', 'left']
            source_map = horizontal_edge_map if is_horizontal else vertical_edge_map
            roi_image = source_map[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
            
            # 计算投影曲线
            proj_curve = calculate_projection_curve(roi_image, 'horizontal' if is_horizontal else 'vertical')

            # 在频域增强曲线中做主峰亚像素定位
            boundary_offset = detect_peak_subpixel(
                proj_curve,
                need_reverse=is_reverse,
                smooth_window=smooth_window
            )
            
            # 转换为全局坐标
            if boundary_offset is not None:
                if is_horizontal:
                    boundaries[roi_name] = float(roi_y + boundary_offset)
                else:
                    boundaries[roi_name] = float(roi_x + boundary_offset)
            else:
                # 失败时使用ROI中间位置
                boundaries[roi_name] = float((roi_y + roi_h // 2) if is_horizontal else (roi_x + roi_w // 2))
        
        top_boundary = boundaries['top']
        bottom_boundary = boundaries['bottom']
        left_boundary = boundaries['left']
        right_boundary = boundaries['right']
        
        # 验证边界合理性：确保边界顺序正确且尺寸合理
        if right_boundary <= left_boundary or bottom_boundary <= top_boundary:
            self.detection_result = False, "尺寸边界无效", {'is_valid': False, 'box_points': [0.0, 0.0, 0.0, 0.0], 'width': 0.0, 'height': 0.0}
            return
        
        width_pixel = right_boundary - left_boundary
        height_pixel = bottom_boundary - top_boundary
        
        x_min, y_min = float(left_boundary), float(top_boundary)
        
        # 返回 x, y, w, h 格式
        box_points = [x_min, y_min, width_pixel, height_pixel]
        
        # 转换为实际尺寸（mm）：宽度可用 pixel_size_x，未设置时与 pixel_size 相同
        pixel_size = self.params.get('pixel_size', 0.001)
        ps_x = self.params.get('pixel_size_x')
        width_scale = pixel_size if ps_x is None else ps_x
        product_width_mm = width_pixel * width_scale
        product_height_mm = height_pixel * pixel_size
        
        # 判断尺寸是否合格
        is_valid = False
        
        if width_pixel > 0 and height_pixel > 0:
            std_size = self.params.get('std_size', (0.0, 0.0))
            std_width, std_height = std_size
            
            if std_width > 0 and std_height > 0:
                    
                tolerance_x = self.params.get('allow_tolerance_x', 0.0)
                tolerance_y = self.params.get('allow_tolerance_y', 0.0)

                # 判断是否在容差范围内
                width_diff = abs(std_width - product_width_mm)
                height_diff = abs(std_height - product_height_mm)
                if width_diff <= tolerance_x and height_diff <= tolerance_y:
                    is_valid = True
            else:
                # 没有标准尺寸，默认认为合格
                is_valid = True
        
        # 保存检测结果
        detection_result = True, "尺寸检测完成", {
            'is_valid': is_valid,
            'box_points': box_points,
            'width': product_width_mm,
            'height': product_height_mm
        }
            

        return detection_result

    def update_params(self, params: dict, clear_result: bool = True):
        """
        更新检测器参数
        
        Args:
            params: dict 要更新的参数字典，可以包含以下键：
                - min_threshold: int 二值化下阈值（灰度值区间下限）
                - max_threshold: int 二值化上阈值（灰度值区间上限）
                - allow_tolerance_x: float X方向尺寸容差（mm）
                - allow_tolerance_y: float Y方向尺寸容差（mm）
                - roi_width: int ROI区域宽度（像素）
                - std_size: tuple[float, float] 标准产品尺寸 (width, height) 像素单位
                - pixel_size: float 像素尺寸（mm/pixel）
            clear_result: bool 是否清除之前的检测结果，默认为True
        
        Returns:
            bool: 更新是否成功
        
        Example:
            >>> detector.update_params({
            ...     'min_threshold': 100,
            ...     'max_threshold': 200,
            ...     'allow_tolerance_x': 0.1,
            ...     'pixel_size': 0.001
            ... })
        """
        
        # 验证参数有效性
        valid_keys = {
            'min_threshold', 'max_threshold', 'allow_tolerance_x',
            'allow_tolerance_y', 'roi_width', 'std_size', 'pixel_size', 'pixel_size_x',
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
        
        if 'allow_tolerance_x' in params:
            val = params['allow_tolerance_x']
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(f"allow_tolerance_x 必须是非负数，当前值: {val}")
        
        if 'allow_tolerance_y' in params:
            val = params['allow_tolerance_y']
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(f"allow_tolerance_y 必须是非负数，当前值: {val}")
        
        if 'roi_width' in params:
            val = params['roi_width']
            if not isinstance(val, (int, float)) or val <= 0:
                validation_errors.append(f"roi_width 必须是正数，当前值: {val}")
        
        if 'pixel_size' in params:
            val = params['pixel_size']
            if not isinstance(val, (int, float)) or val <= 0:
                validation_errors.append(f"pixel_size 必须是正数，当前值: {val}")

        if 'pixel_size_x' in params and params['pixel_size_x'] is not None:
            val = params['pixel_size_x']
            if not isinstance(val, (int, float)) or val <= 0:
                validation_errors.append(f"pixel_size_x 必须是正数或省略，当前值: {val}")

        if 'std_size' in params:
            val = params['std_size']
            if not isinstance(val, (tuple, list)) or len(val) != 2:
                validation_errors.append(f"std_size 必须是长度为2的元组或列表，当前值: {val}")
            elif not all(isinstance(v, (int, float)) and v >= 0 for v in val):
                validation_errors.append(f"std_size 的元素必须是非负数，当前值: {val}")
        
        # 如果有验证错误，记录并返回失败
        if validation_errors:
            error_msg = "参数验证失败:\n" + "\n".join(f"  - {err}" for err in validation_errors)
            print(error_msg)
            return False
        
        # 更新参数（只更新有效键）
        updated_keys = []
        for key in valid_keys:
            if key in params:
                self.params[key] = params[key]
                updated_keys.append(key)
        
        # 清除之前的检测结果（如果参数已更改）
        if clear_result and updated_keys:
            self.detection_result = None
        return True
    
    def get_params(self):
        """
        获取当前参数
        
        Returns:
            dict: 当前参数字典的副本
        """
        return self.params.copy()



Size_D = SizeDetector()
Size_D.__init__()
detect_params = {
    "min_threshold":100,
    "max_threshold":200,
    "pixel_size":0.008841,
    "roi_width":150
}
product_info = {
    "size_result":()
}

Size_D.update_params(detect_params)
for i in os.listdir('Image\Data_Save_Dry'):
    image_ori = cv.imread("Image\Data_Save_Dry\\"+i)
    size_detect_result = Size_D.detect(image_ori)
    print(size_detect_result[2])
    print(size_detect_result[2])
    product_info["size_result"] = size_detect_result
    ret,msg,image_ori = support_funs.draw_detection_results(image_result= image_ori,product_info= product_info)
    cv.namedWindow("result",cv.WINDOW_NORMAL)
    cv.imshow("result",image_ori)
    cv.waitKey()
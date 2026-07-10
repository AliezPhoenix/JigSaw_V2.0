from . import *
from src.support.support_funs import ensure_gray_u8
from src.support.data_structure import Size_Result


# 负梯度候选：相对全局最负值的容差带（grad <= g_min + ratio*|g_min|）
_NEG_GRAD_CANDIDATE_RATIO = 0.2


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

def calculate_subpixel_offset(grad_values, min_idx, method='linear'):
    """
    计算亚像素偏移量
    
    Args:
        grad_values: 梯度值数组
        min_idx: 最小值索引
        method: 插值方法，'linear'（默认）或'parabola'
    
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


def _is_local_min(grad, i):
    """判断 grad[i] 是否为局部极小。"""
    n = len(grad)
    if n == 1:
        return True
    if i == 0:
        return grad[0] <= grad[1]
    if i == n - 1:
        return grad[-1] <= grad[-2]
    return grad[i] <= grad[i - 1] and grad[i] < grad[i + 1]


def _select_neg_grad_min_index(grad, candidate_ratio=_NEG_GRAD_CANDIDATE_RATIO):
    """
    沿检测方向（索引从小到大）选择负梯度边界索引。

    1. 以全曲线负梯度全局最小值 g_min 为参考幅值；
    2. 收集局部极小且 grad<0、且落在 g_min 容差带内的候选；
    3. 取检测方向上第一个候选；若无候选则回退到 g_min 所在位置。
    """
    grad = np.asarray(grad, dtype=np.float64)
    n = len(grad)
    if n == 0:
        return None

    neg_mask = grad < 0
    if not np.any(neg_mask):
        return int(np.argmax(np.abs(grad)))

    g_min = float(np.min(grad[neg_mask]))
    g_min_idx = int(np.argmin(np.where(neg_mask, grad, np.inf)))

    # 容差带：更负或接近 g_min（例如 g_min=-10, ratio=0.2 → 上限 -8）
    band_hi = g_min + float(candidate_ratio) * abs(g_min)

    candidates = [
        i for i in range(n)
        if grad[i] < 0 and _is_local_min(grad, i) and grad[i] <= band_hi
    ]
    if candidates:
        return int(candidates[0])
    return g_min_idx


def detect_boundary_subpixel(proj_curve, need_reverse=False, dx=1.0, smooth_window=3):
    """
    检测边界点（亚像素精度）。
    检测逻辑：由产品（白）向背景（黑）搜索；
    在负梯度局部极小中，取接近全局最负值的第一个候选（沿检测方向）。
    """
    if len(proj_curve) == 0:
        return None
    
    # 计算并平滑梯度
    grad = np.gradient(proj_curve, dx)
    if smooth_window > 1:
        grad = smooth_gradient(grad, smooth_window)

    min_idx = _select_neg_grad_min_index(grad)
    if min_idx is None:
        return None
    
    # 亚像素偏移：默认线性拟合
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

# ==================== SizeDetector 类 ====================

class SizeDetector:
    ROI_SIDES = ("top", "left", "bottom", "right")
    DEFAULT_ROI_STRIP = 80

    def __init__(self, params: dict = None):
        self.image = None
        self.detection_result = None  # 存储检测结果

        # 默认参数
        self.params = {
            "min_threshold": 0,
            "max_threshold": 255,
            "allow_tolerance_x": 0.0,
            "allow_tolerance_y": 0.0,
            "rois": {side: None for side in self.ROI_SIDES},
            "std_size": (0.0, 0.0),  # 标准产品尺寸 (width, height) mm单位
            "pixel_size": 0.001,  # 像素尺寸（mm/pixel）
        }
        if params:
            self.update_params(params)

    @staticmethod
    def default_rois(img_w: int, img_h: int, strip: int = None):
        """按图像尺寸生成默认四边条带 ROI（自由矩形初始值）。"""
        strip = int(strip if strip is not None else SizeDetector.DEFAULT_ROI_STRIP)
        w = max(1, int(img_w))
        h = max(1, int(img_h))
        tw = max(1, min(strip, h))
        lw = max(1, min(strip, w))
        return {
            "top": (0, 0, w, tw),
            "bottom": (0, max(0, h - tw), w, tw),
            "left": (0, 0, lw, h),
            "right": (max(0, w - lw), 0, lw, h),
        }

    @staticmethod
    def _normalize_roi(roi):
        if roi is None:
            return None
        if isinstance(roi, dict):
            try:
                x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            except (KeyError, TypeError, ValueError):
                return None
        else:
            try:
                if len(roi) < 4:
                    return None
                x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            except (TypeError, ValueError):
                return None
        if w < 1 or h < 1:
            return None
        return (x, y, w, h)

    @classmethod
    def _normalize_rois(cls, rois):
        if not isinstance(rois, dict):
            return {side: None for side in cls.ROI_SIDES}
        return {side: cls._normalize_roi(rois.get(side)) for side in cls.ROI_SIDES}

    @staticmethod
    def _clamp_roi_to_image(roi, img_w, img_h):
        if roi is None:
            return None
        x, y, w, h = roi
        img_w, img_h = int(img_w), int(img_h)
        if img_w < 1 or img_h < 1:
            return None
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        return (x, y, w, h)

    def _validate_rois(self, img_w, img_h):
        """四边 ROI 必须完整且可夹到图内；否则返回错误文案。"""
        rois = self.params.get("rois") or {}
        out = {}
        for side in self.ROI_SIDES:
            roi = self._normalize_roi(rois.get(side))
            if roi is None:
                return None, "未框选完整的四个 ROI"
            clamped = self._clamp_roi_to_image(roi, img_w, img_h)
            if clamped is None:
                return None, "未框选完整的四个 ROI"
            out[side] = clamped
        return out, None

    def detect(self, image):
        """
        执行尺寸检测

        Args:
            image: 输入的灰度图像或BGR图像
        Returns:
            Size_Result
        """
        image_gray = ensure_gray_u8(image, copy=True)
        self.image = image_gray
        h, w = image_gray.shape

        rois, roi_err = self._validate_rois(w, h)
        if roi_err is not None:
            self.detection_result = Size_Result(
                error_code=2, error_msg=roi_err, is_valid=False
            )
            return self.detection_result

        # 二值化
        image_binary = cv.inRange(
            image_gray, self.params["min_threshold"], self.params["max_threshold"]
        )

        # 默认参数：差分步长0.7，平滑窗口5
        gradient_dx = 0.7
        smooth_window = 5

        # 对每个ROI进行边界检测
        boundaries = {}
        for roi_name, (roi_x, roi_y, roi_w, roi_h) in rois.items():
            roi_image = image_binary[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
            if roi_image.size == 0:
                self.detection_result = Size_Result(
                    error_code=2, error_msg=f"{roi_name}边ROI无效", is_valid=False
                )
                return self.detection_result

            is_horizontal = roi_name in ["top", "bottom"]
            is_reverse = roi_name in ["top", "left"]

            # 计算投影曲线
            proj_curve = calculate_projection_curve(
                roi_image, "horizontal" if is_horizontal else "vertical"
            )

            # 根据搜索方向处理投影曲线
            proj_curve_for_detection = proj_curve[::-1] if is_reverse else proj_curve

            # 检测边界
            boundary_offset = detect_boundary_subpixel(
                proj_curve_for_detection,
                need_reverse=is_reverse,
                dx=gradient_dx,
                smooth_window=smooth_window,
            )

            # 转换为全局坐标
            if boundary_offset is not None:
                if is_horizontal:
                    boundaries[roi_name] = float(roi_y + boundary_offset)
                else:
                    boundaries[roi_name] = float(roi_x + boundary_offset)
            else:
                self.detection_result = Size_Result(
                    error_code=2,
                    error_msg=f"无法检测到{roi_name}边边界",
                    is_valid=False,
                )
                return self.detection_result

        top_boundary = boundaries["top"]
        bottom_boundary = boundaries["bottom"]
        left_boundary = boundaries["left"]
        right_boundary = boundaries["right"]

        # 验证边界合理性：确保边界顺序正确且尺寸合理
        if right_boundary <= left_boundary or bottom_boundary <= top_boundary:
            self.detection_result = Size_Result(
                error_code=2, error_msg="尺寸边界无效", is_valid=False
            )
            return self.detection_result

        width_pixel = right_boundary - left_boundary
        height_pixel = bottom_boundary - top_boundary

        x_min, y_min = float(left_boundary), float(top_boundary)

        # 返回 x, y, w, h 格式
        box_points = [x_min, y_min, width_pixel, height_pixel]

        # 转换为实际尺寸（mm）：宽度可用 pixel_size_x，未设置时与 pixel_size 相同
        pixel_size = self.params.get("pixel_size", 0.001)
        ps_x = self.params.get("pixel_size_x")
        width_scale = pixel_size if ps_x is None else ps_x
        product_width_mm = width_pixel * width_scale
        product_height_mm = height_pixel * pixel_size

        # 判断尺寸是否合格
        is_valid = False

        if width_pixel > 0 and height_pixel > 0:
            std_size = self.params.get("std_size", (0.0, 0.0))
            std_width, std_height = std_size

            if std_width > 0 and std_height > 0:
                tolerance_x = self.params.get("allow_tolerance_x", 0.0)
                tolerance_y = self.params.get("allow_tolerance_y", 0.0)

                # 判断是否在容差范围内
                width_diff = abs(std_width - product_width_mm)
                height_diff = abs(std_height - product_height_mm)
                if width_diff <= tolerance_x and height_diff <= tolerance_y:
                    is_valid = True
            else:
                # 没有标准尺寸，默认认为合格
                is_valid = True

        # 保存检测结果
        detection_result = Size_Result(
            width=product_width_mm,
            height=product_height_mm,
            box_points=[float(v) for v in box_points],
            is_valid=is_valid,
        )
        self.detection_result = detection_result

        return detection_result

    def update_params(self, params: dict, clear_result: bool = True):
        """
        更新检测器参数

        Args:
            params: dict 要更新的参数字典，可以包含以下键：
                - min_threshold / max_threshold
                - allow_tolerance_x / allow_tolerance_y
                - rois: {top/left/bottom/right: (x,y,w,h)|None}
                - std_size / pixel_size / pixel_size_x
                - roi_width: 旧键，忽略（兼容旧配方）
            clear_result: bool 是否清除之前的检测结果，默认为True

        Returns:
            bool: 更新是否成功
        """
        if not params:
            return True

        # 验证参数有效性（roi_width 仅兼容忽略，不参与检测）
        valid_keys = {
            "min_threshold",
            "max_threshold",
            "allow_tolerance_x",
            "allow_tolerance_y",
            "rois",
            "std_size",
            "pixel_size",
            "pixel_size_x",
        }

        validation_errors = []

        if "min_threshold" in params:
            val = params["min_threshold"]
            if not isinstance(val, (int, float)) or val < 0 or val > 255:
                validation_errors.append(
                    f"min_threshold 必须在 [0, 255] 范围内，当前值: {val}"
                )

        if "max_threshold" in params:
            val = params["max_threshold"]
            if not isinstance(val, (int, float)) or val < 0 or val > 255:
                validation_errors.append(
                    f"max_threshold 必须在 [0, 255] 范围内，当前值: {val}"
                )

        if "min_threshold" in params and "max_threshold" in params:
            if params["min_threshold"] > params["max_threshold"]:
                validation_errors.append(
                    f"min_threshold ({params['min_threshold']}) 不能大于 max_threshold ({params['max_threshold']})"
                )

        if "allow_tolerance_x" in params:
            val = params["allow_tolerance_x"]
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(
                    f"allow_tolerance_x 必须是非负数，当前值: {val}"
                )

        if "allow_tolerance_y" in params:
            val = params["allow_tolerance_y"]
            if not isinstance(val, (int, float)) or val < 0:
                validation_errors.append(
                    f"allow_tolerance_y 必须是非负数，当前值: {val}"
                )

        if "rois" in params and params["rois"] is not None and not isinstance(params["rois"], dict):
            validation_errors.append(f"rois 必须是字典，当前值: {params['rois']}")

        if "pixel_size" in params:
            val = params["pixel_size"]
            if not isinstance(val, (int, float)) or val <= 0:
                validation_errors.append(f"pixel_size 必须是正数，当前值: {val}")

        if "pixel_size_x" in params and params["pixel_size_x"] is not None:
            val = params["pixel_size_x"]
            if not isinstance(val, (int, float)) or val <= 0:
                validation_errors.append(
                    f"pixel_size_x 必须是正数或省略，当前值: {val}"
                )

        if "std_size" in params:
            val = params["std_size"]
            if not isinstance(val, (tuple, list)) or len(val) != 2:
                validation_errors.append(
                    f"std_size 必须是长度为2的元组或列表，当前值: {val}"
                )
            elif not all(isinstance(v, (int, float)) and v >= 0 for v in val):
                validation_errors.append(
                    f"std_size 的元素必须是非负数，当前值: {val}"
                )

        if validation_errors:
            error_msg = "参数验证失败:\n" + "\n".join(f"  - {err}" for err in validation_errors)
            print(error_msg)
            return False

        updated_keys = []
        for key in valid_keys:
            if key in params:
                if key == "rois":
                    self.params[key] = self._normalize_rois(params[key])
                else:
                    self.params[key] = params[key]
                updated_keys.append(key)

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

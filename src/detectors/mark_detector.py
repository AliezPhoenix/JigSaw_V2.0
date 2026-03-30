from . import *
from src.support.support_funs import ensure_gray_u8, normalize_mark_rois_in_params


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
                - mark_rois: list[list[int]] 每项 [x, y, w, h]
                - allow_mark: bool 聚合语义：False=干燥(OR)，True=移栽(AND)
        """

        self.image = None

        self.params = {
            "min_threshold": 0,
            "max_threshold": 255,
            "min_mark_area": 0,
            "auto_threshold_factor": 1.05,
            "pixel_size": 0.001,
            "mark_detect_mode": "manual",
            "mark_rois": [],
            "allow_mark": False,
        }
        if params:
            self.update_params(params)

    def _detect_single_roi_mark(self, image_gray: np.ndarray, roi_rect) -> dict:
        """对单个 ROI 执行二值化与轮廓检测。roi_rect: [x,y,w,h]。"""
        h_img, w_img = image_gray.shape[:2]
        try:
            x, y, roi_w, roi_h = (
                int(roi_rect[0]),
                int(roi_rect[1]),
                int(roi_rect[2]),
                int(roi_rect[3]),
            )
        except (TypeError, ValueError, IndexError):
            return {
                "is_valid": False,
                "mark_contour": None,
                "mark_area": 0.0,
                "mark_area_mm": 0.0,
            }

        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        roi_w = max(1, min(roi_w, w_img - x))
        roi_h = max(1, min(roi_h, h_img - y))

        try:
            image_gray_mark = image_gray[y : y + roi_h, x : x + roi_w]
        except Exception:
            return {
                "is_valid": False,
                "mark_contour": None,
                "mark_area": 0.0,
                "mark_area_mm": 0.0,
            }

        min_threshold = self.params.get("min_threshold", 0)
        max_threshold = self.params.get("max_threshold", 255)
        min_mark_area = self.params.get("min_mark_area", 0)
        auto_threshold_factor = self.params.get("auto_threshold_factor", 1.05)
        mark_detect_mode = self.params.get("mark_detect_mode", "manual")
        pixel_size = self.params.get("pixel_size", 0.001)

        if mark_detect_mode == "auto":
            mean_val = int(cv.mean(image_gray_mark)[0] * auto_threshold_factor)
            image_binary_mark = cv.inRange(image_gray_mark, mean_val, max_threshold)
        else:
            image_binary_mark = cv.inRange(
                image_gray_mark, min_threshold, max_threshold
            )

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        image_binary_mark = cv.morphologyEx(
            image_binary_mark, cv.MORPH_OPEN, kernel
        )

        contours, _ = cv.findContours(
            image_binary_mark, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )

        mark_contour = None
        mark_area = 0.0

        for contour in contours:
            area = cv.contourArea(contour)
            if area > min_mark_area:
                mark_contour = contour
                mark_area = area
                break

        if mark_contour is not None and mark_area > 0:
            mark_contour_global = mark_contour + np.array([x, y], dtype=np.int32)
            mark_area_mm = mark_area * pixel_size * pixel_size
            return {
                "is_valid": True,
                "mark_contour": mark_contour_global,
                "mark_area": float(mark_area),
                "mark_area_mm": float(mark_area_mm),
            }
        return {
            "is_valid": False,
            "mark_contour": None,
            "mark_area": 0.0,
            "mark_area_mm": 0.0,
        }

    def detect(self, image):
        """
        执行 Mark 标记检测（多 ROI 聚合）。

        Returns:
            tuple: (success, msg, result_dict)
        """
        image_gray = ensure_gray_u8(image, copy=True)
        self.image = image_gray

        mark_rois = self.params.get("mark_rois") or []
        allow_mark = bool(self.params.get("allow_mark", False))

        if not mark_rois:
            return True, "未配置Mark ROI", {
                "is_valid": False,
                "per_roi": [],
                "mark_contour": None,
                "mark_area": 0.0,
                "mark_area_mm": 0.0,
            }

        per_roi = []
        for i, roi_rect in enumerate(mark_rois):
            one = self._detect_single_roi_mark(image_gray, roi_rect)
            per_roi.append(
                {
                    "index": i,
                    "roi": list(roi_rect) if isinstance(roi_rect, (list, tuple)) else roi_rect,
                    "is_valid": one["is_valid"],
                    "mark_contour": one["mark_contour"],
                    "mark_area": one["mark_area"],
                    "mark_area_mm": one["mark_area_mm"],
                }
            )

        if allow_mark:
            is_valid = all(entry["is_valid"] for entry in per_roi)
        else:
            is_valid = any(entry["is_valid"] for entry in per_roi)

        contours_draw = [
            entry["mark_contour"]
            for entry in per_roi
            if entry.get("mark_contour") is not None
        ]
        if not contours_draw:
            mark_contour_out = None
        elif len(contours_draw) == 1:
            mark_contour_out = contours_draw[0]
        else:
            mark_contour_out = contours_draw

        total_area = sum(entry.get("mark_area", 0.0) for entry in per_roi)
        total_mm = sum(entry.get("mark_area_mm", 0.0) for entry in per_roi)

        msg = "检测到Mark" if is_valid else "未检测到Mark"
        return True, msg, {
            "is_valid": is_valid,
            "per_roi": per_roi,
            "mark_contour": mark_contour_out,
            "mark_area": float(total_area),
            "mark_area_mm": float(total_mm),
        }

    def update_params(self, params: dict):
        """更新检测器参数（仅接受已知键；支持 mark_rois / allow_mark）。"""
        for key, value in params.items():
            if key in self.params or key in ("mark_rois", "allow_mark"):
                self.params[key] = value
        normalize_mark_rois_in_params(self.params)
        return True, "参数更新成功"

    def get_params(self):
        return self.params.copy()

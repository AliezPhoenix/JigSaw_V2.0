from . import *
from src.support.support_funs import ensure_gray_u8

class TemplateDetector:
    def __init__(self):
        self.template = None
        self.image = None
        self.params = {
            "template_threshold": 0.6,
            "search_roi": None
        }
    def detect(self, template, image):
        """
        模板匹配
        """
        params = self.params
        threshold = params.get("template_threshold", 0.6)
        search_roi = params.get("search_roi", None)
        img_h, img_w = image.shape[:2]
        if search_roi and isinstance(search_roi, (list, tuple)) and len(search_roi) >= 4:
            x, y, w, h = int(search_roi[0]), int(search_roi[1]), int(search_roi[2]), int(search_roi[3])
        else:
            x, y, w, h = 0, 0, img_w, img_h
        roi_origin_x, roi_origin_y = x, y
        self.template = template
        self.image = image[y:y+h,x:x+w]
        match_points = []
        result_list = []
        template_width = self.template.shape[1]
        template_height = self.template.shape[0]

        self.template = ensure_gray_u8(self.template, copy=True)
        self.image = ensure_gray_u8(self.image, copy=True)

        self.template = cv.GaussianBlur(self.template, (5, 5), 0)
        self.image = cv.GaussianBlur(self.image, (5, 5), 0)


        result = cv.matchTemplate(self.image, self.template, cv.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            x,y = pt
            confidence = result[y,x]
            match_points.append((x,y,confidence))

        match_points.sort(key=lambda x: x[2], reverse=True)
        min_distence = min(template_height/2, template_width/2) ### 采取二分之一的模板宽度作为最小距离
  
        for x, y ,confidence in match_points:
            too_close = False
            for existing_x, existing_y in result_list:
                distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if distance < min_distence:
                    too_close = True
                    break
            if not too_close:
                result_list.append((x, y))

        y_tolerance = template_height/2
        result_list.sort(key=lambda pt: (int(pt[1] / y_tolerance) if y_tolerance > 0 else int(pt[1]), pt[0]))

        # 将坐标转换为原始图像坐标（加上裁剪区域左上角偏移；循环内会覆盖 x,y，故用 roi_origin_*）
        result_list = [(px + roi_origin_x, py + roi_origin_y) for px, py in result_list]

        return result_list

    def update_params(self, params:dict):
        self.params.update(params)
        return True,"参数更新成功"

    def get_params(self):
        return self.params
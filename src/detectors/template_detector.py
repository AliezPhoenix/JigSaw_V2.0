from . import *

from src.support.support_funs import ensure_gray_u8





class TemplateDetector:

    def __init__(self):

        self.template = None

        self.image = None

        self.params = {

            "template_threshold": 0.6,

            "search_roi": None,

            "use_pyramid": False,

            "pyramid_scale": 2,

            "pyramid_refine_margin": 0.2,

        }



    def detect(self, template, image):

        """

        模板匹配。



        Returns:

            list[tuple]: 每项为 (x, y, confidence)，全图坐标；confidence 为 TM_CCOEFF_NORMED 得分。

        """

        if self.params.get("use_pyramid", False):

            return self._detect_pyramid(template, image)

        return self._detect_standard(template, image)



    def _detect_standard(self, template, image):

        roi_origin_x, roi_origin_y, template_gray, image_gray, template_width, template_height = (

            self._prepare_match_inputs(template, image)

        )

        result_list = self._collect_matches(

            image_gray,

            template_gray,

            self.params.get("template_threshold", 0.6),

            template_width,

            template_height,

        )

        return [

            (px + roi_origin_x, py + roi_origin_y, float(conf))

            for px, py, conf in result_list

        ]



    def _detect_pyramid(self, template, image):

        """粗搜（下采样）+ 原图局部精搜，精搜窗口为模板宽高各扩展 20% margin。"""

        roi_origin_x, roi_origin_y, template_gray, image_gray, template_width, template_height = (

            self._prepare_match_inputs(template, image)

        )

        threshold = self.params.get("template_threshold", 0.6)

        scale = max(4, int(self.params.get("pyramid_scale", 2)))

        margin_ratio = float(self.params.get("pyramid_refine_margin", 0.2))



        img_h, img_w = image_gray.shape[:2]

        if (

            min(template_width, template_height) < scale * 4

            or max(img_w, img_h) < scale * 4

        ):

            return self._detect_standard(template, image)



        coarse_template = cv.resize(

            template_gray,

            (max(1, template_width // scale), max(1, template_height // scale)),

            interpolation=cv.INTER_AREA,

        )

        coarse_image = cv.resize(

            image_gray,

            (max(1, img_w // scale), max(1, img_h // scale)),

            interpolation=cv.INTER_AREA,

        )

        coarse_tw, coarse_th = coarse_template.shape[1], coarse_template.shape[0]

        coarse_matches = self._collect_matches(

            coarse_image,

            coarse_template,

            threshold,

            coarse_tw,

            coarse_th,

        )

        if not coarse_matches:

            return []



        margin_x = max(1, int(template_width * margin_ratio))

        margin_y = max(1, int(template_height * margin_ratio))

        refined_points = []

        for cx, cy, _ in coarse_matches:

            fx = int(cx * scale)

            fy = int(cy * scale)

            win_x = max(0, fx - margin_x)

            win_y = max(0, fy - margin_y)

            win_w = min(img_w - win_x, template_width + 2 * margin_x)

            win_h = min(img_h - win_y, template_height + 2 * margin_y)

            if win_w < template_width or win_h < template_height:

                continue



            patch = image_gray[win_y:win_y + win_h, win_x:win_x + win_w]

            fine_result = cv.matchTemplate(patch, template_gray, cv.TM_CCOEFF_NORMED)

            _, fine_conf, _, fine_loc = cv.minMaxLoc(fine_result)

            if fine_conf < threshold:

                continue

            refined_points.append((win_x + fine_loc[0], win_y + fine_loc[1], float(fine_conf)))



        if not refined_points:

            return []



        result_list = self._filter_and_sort_matches(

            refined_points,

            template_width,

            template_height,

        )

        return [

            (px + roi_origin_x, py + roi_origin_y, float(conf))

            for px, py, conf in result_list

        ]



    def _prepare_match_inputs(self, template, image):

        params = self.params

        search_roi = params.get("search_roi", None)

        img_h, img_w = image.shape[:2]

        if search_roi and isinstance(search_roi, (list, tuple)) and len(search_roi) >= 4:

            x, y, w, h = int(search_roi[0]), int(search_roi[1]), int(search_roi[2]), int(search_roi[3])

        else:

            x, y, w, h = 0, 0, img_w, img_h



        roi_origin_x, roi_origin_y = x, y

        self.template = template

        self.image = image[y:y + h, x:x + w]

        template_width = self.template.shape[1]

        template_height = self.template.shape[0]



        template_gray = ensure_gray_u8(self.template, copy=True)

        image_gray = ensure_gray_u8(self.image, copy=True)

        template_gray = cv.GaussianBlur(template_gray, (5, 5), 0)

        image_gray = cv.GaussianBlur(image_gray, (5, 5), 0)

        return roi_origin_x, roi_origin_y, template_gray, image_gray, template_width, template_height



    def _collect_matches(self, image_gray, template_gray, threshold, template_width, template_height):

        result = cv.matchTemplate(image_gray, template_gray, cv.TM_CCOEFF_NORMED)

        match_points = []

        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):

            px, py = pt

            match_points.append((px, py, float(result[py, px])))



        match_points.sort(key=lambda item: item[2], reverse=True)

        return self._filter_and_sort_matches(match_points, template_width, template_height)



    def _filter_and_sort_matches(self, match_points, template_width, template_height):

        result_list = []

        min_distence = max((template_height / 3) * 2, (template_width / 3) * 2)



        for x, y, confidence in match_points:

            too_close = False

            for existing_x, existing_y, _ in result_list:

                distance = np.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)

                if distance < min_distence:

                    too_close = True

                    break

            if not too_close:

                result_list.append((x, y, confidence))



        y_tolerance = template_height / 2

        result_list.sort(

            key=lambda pt: (int(pt[1] / y_tolerance) if y_tolerance > 0 else int(pt[1]), pt[0])

        )

        return result_list



    def update_params(self, params: dict):

        self.params.update(params)

        return True, "参数更新成功"



    def get_params(self):

        return self.params



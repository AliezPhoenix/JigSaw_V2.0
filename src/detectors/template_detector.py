from . import *

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
        self.template = template
        for x,y,w,h in search_roi:
            self.image = image[y:y+h,x:x+w]
        match_points = []
        result_list = []
        template_width = self.template.shape[1]
        template_height = self.template.shape[0]

        if len(self.template.shape) == 3:
            self.template = cv.cvtColor(self.template, cv.COLOR_BGR2GRAY)
        if len(self.image.shape) == 3:
            self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        self.template = cv.GaussianBlur(self.template, (5, 5), 0)
        self.image = cv.GaussianBlur(self.image, (5, 5), 0)


        result = cv.matchTemplate(self.image, self.template, cv.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            x,y = pt
            confidence = result[y,x]
            match_points.append((x,y,confidence))

        match_points = match_points.sort(key=lambda x: x[2], reverse=True)
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

        return result_list

    def update_params(self, params:dict):
        self.params.update(params)
        return True,"参数更新成功"

    def get_params(self):
        return self.params
import cv2 as cv
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
#from paddleocr import PaddleOCR
class bga_strip_log():
    def __init__(self,product_size,window_size,side) -> None:
        if side == "front":
            self.full_value = create_alternating_array(product_size[0],product_size[1],0,(0,99))
            self.window_value = create_alternating_array(window_size[0],window_size[1],0,(0,99))
        elif side == "back":
            self.full_value = create_alternating_array(product_size[0],product_size[1],1,(0,99))
            self.window_value = create_alternating_array(window_size[0],window_size[1],1,(0,99))
        else:
            raise ValueError("side must be 'front' or 'back'")
        
        self.postions = calculate_write_positions(self.full_value,self.window_value)
        self.image_dict = {}
        self.full_value_cols = product_size[1]
        self.full_value_rows = product_size[0]
        self.window_value_cols = window_size[1]
        self.window_value_rows = window_size[0]
        self.side = side
        for pos in self.postions:
            self.image_dict[pos[0:2]] = np.zeros((100,100))
        self.count = 0
        self.log_file = {"Lot_ID":"",
                        "Strip_ID":"",
                        "Date":"",
                        "Time":"",
                        "Activate_Defect_Type":"",
                        "Result":""}
        # 添加最大历史记录数量限制（用于内存管理）
        self.max_history_size = 500  # 默认保留最近200条记录
    def write(self, current_value, current_image):
        """
        将current_value中的值按照顺序填入full_value当前切片中非0的位置
        
        参数:
            current_value: 值列表，长度等于当前切片中所有非0元素的数量
            current_image: 当前图像
        """
        # 检查count是否超出范围
        if self.count >= len(self.postions):
            print(f"错误: count ({self.count}) 超出postions范围 ({len(self.postions)})")
            return
        
        if not current_value:
            print("警告: current_value为空，跳过写入")
            self.image_dict[self.postions[self.count][0:2]] = current_image
            self.count += 1
            return
        
        # 获取当前位置信息
        pos = self.postions[self.count]
        row_start, col_start, _, _, actual_row_end, actual_col_end, _, _ = pos
        
        # 调整切片范围到最大可用范围
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(actual_row_end, self.full_value_rows)
        col_end = min(actual_col_end, self.full_value_cols)
        
        # 验证调整后的范围有效性
        if row_end <= row_start or col_end <= col_start:
            print(f"错误: 调整后的切片范围无效: ({row_start}, {col_start}) 到 ({row_end}, {col_end})")
            self.image_dict[pos[0:2]] = current_image
            self.count += 1
            return
        
        # 获取切片并找到所有值为99的位置（99代表未检测到产品，需要写入检测结果）
        full_slice = self.full_value[row_start:row_end, col_start:col_end].copy()
        target_positions = [
            (row, col) 
            for row in range(full_slice.shape[0]) 
            for col in range(full_slice.shape[1]) 
            if full_slice[row, col] == 99
        ]
        
        if not target_positions:
            print("警告: 切片中没有值为99的位置，跳过写入")
            self.image_dict[pos[0:2]] = current_image
            self.count += 1
            return
        
        # 将current_value中的值填入值为99的位置
        if len(current_value) != len(target_positions):
            print(f"警告: current_value长度 ({len(current_value)}) 与切片99位置数量 ({len(target_positions)}) 不匹配")
        
        for i, (row, col) in enumerate(target_positions[:len(current_value)]):
            defect_type = current_value[i]
            # 根据defect_type列表中的缺陷类型设置对应的值
            if not isinstance(defect_type, list) or len(defect_type) == 0 or defect_type[0] == "OK":
                full_slice[row, col] = 2  # OK - 绿色
            else:
                # 根据不同的NG类型设置不同的值（按优先级）
                # 优先级：Mark > Size > Ball Count > Ball_Area > Shift > 其他
                if "Mark" in defect_type:
                    full_slice[row, col] = 1  # Mark - 红色
                elif "Size" in defect_type:
                    full_slice[row, col] = 3  # Size - 紫色
                elif "Ball Count" in defect_type:
                    full_slice[row, col] = 4  # BallCount - 橙色
                elif "Ball_Area" in defect_type:
                    full_slice[row, col] = 5  # Ball_Area - 黄色
                elif "Shift" in defect_type:
                    full_slice[row, col] = 6  # Shift - 棕色
                else:
                    full_slice[row, col] = 8  # 默认NG（如Scratch等）- 红色
        
        # 将修改后的切片写回full_value
        self.full_value[row_start:row_end, col_start:col_end] = full_slice
        
        # 保存图像并更新计数
        self.image_dict[pos[0:2]] = current_image
        self.count += 1
        
        # 如果图像数量超过限制，清理最旧的图像（保留最近的数据）
        if len(self.image_dict) > self.max_history_size:
            # 简化处理：如果超过限制，删除最旧的一半
            num_to_remove = len(self.image_dict) - self.max_history_size // 2
            keys_to_remove = list(self.image_dict.keys())[:num_to_remove]
            for key in keys_to_remove:
                if key in self.image_dict:
                    del self.image_dict[key]

    def get_pos_image(self,pos):
        return self.image_dict.get(pos[0:2], None)
    
    def cleanup_old_data(self, max_size=None):
        """
        清理旧数据，保留最近的数据
        
        参数:
            max_size: 最大保留数量，如果为None则使用self.max_history_size
        """
        if max_size is None:
            max_size = self.max_history_size
        
        if len(self.image_dict) <= max_size:
            return
        
        # 获取所有位置，按count排序，删除最旧的
        # 简化处理：如果超过限制，删除最旧的一半
        num_to_remove = len(self.image_dict) - max_size // 2
        keys_to_remove = list(self.image_dict.keys())[:num_to_remove]
        for key in keys_to_remove:
            if key in self.image_dict:
                del self.image_dict[key]
    def get_full_animation(self):
        array = np.array(self.full_value)
        h, w = array.shape

        # 根据整体图像比例计算方块尺寸
        # 整体图像比例为 y:480, x:150
        # 例如：30行5列 -> 方块尺寸 16*30 (480/30=16, 150/5=30)
        margin = 2       # 小方块间隔
        block_height = 480 // h  # 每个小方块的高度（像素）
        block_width = 150 // w    # 每个小方块的宽度（像素）
        img_h = h * block_height + (h + 1) * margin
        img_w = w * block_width + (w + 1) * margin

        # 初始化底图（灰色背景）
        img = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                y1 = margin + i * (block_height + margin)
                x1 = margin + j * (block_width + margin)
                y2 = y1 + block_height
                x2 = x1 + block_width
                if array[i, j] == 2:
                    color = (0, 255, 0)  # OK - 绿色
                elif array[i, j] == 1:
                    color = (0, 0, 255)  # Mark - 红色
                elif array[i, j] == 3:
                    color = (128, 0, 128)
                      # Size - 紫色
                elif array[i, j] == 4:
                    color = (0, 165, 255)  # BallCount - 橙色
                elif array[i, j] == 5:
                    color = (0, 255, 255)  # Ball_Area - 黄色
                elif array[i, j] == 6:
                    color = (42, 42, 165)  # Shift - 棕色
                elif array[i, j] == 8:
                    color = (0, 0, 255)  # NG - 红色
                elif array[i, j] == 99:
                    color = (255, 255, 255)  # 未检测到产品 - 白色
                elif array[i, j] == 0:
                    color = (0, 0, 0)  # 空白
                else:
                    color = (255, 255, 255)  # 其他 - 白色
                img[y1:y2, x1:x2] = color


        return img
    def display_animation(self):
        img = self.get_full_animation()
        
        # 获取图像参数（与 get_full_animation 保持一致）
        h, w = self.full_value.shape
        margin = 2
        block_height = 480 // h  # 每个小方块的高度（像素）
        block_width = 150 // w    # 每个小方块的宽度（像素）
        
        # 存储区域信息：每个区域对应一个pos的起始位置和像素范围
        regions = []
        for pos in self.postions:
            row_start, col_start, row_end, col_end, \
            actual_row_end, actual_col_end, \
            source_row_size, source_col_size = pos
            
            # 计算区域在图像中的像素坐标
            x_start = margin + col_start * (block_width + margin)
            y_start = margin + row_start * (block_height + margin)
            x_end = margin + actual_col_end * (block_width + margin)
            y_end = margin + actual_row_end * (block_height + margin)
            
            regions.append({
                'pos_start': (row_start, col_start),
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end
            })
        
        # 鼠标回调函数
        def mouse_callback(event, x, y, flags, param):
            # 查找鼠标当前位置属于哪个区域
            current_region = None
            for region in regions:
                if (region['x_start'] <= x < region['x_end'] and 
                    region['y_start'] <= y < region['y_end']):
                    current_region = region
                    break
            
            # 创建显示图像（从原始图像复制）
            display_img = img.copy()
            
            # 如果鼠标在某个区域内，绘制该区域的矩形框
            if current_region is not None:
                cv.rectangle(display_img, 
                            (current_region['x_start'], current_region['y_start']),
                            (current_region['x_end'], current_region['y_end']),
                            (255, 0, 255), 3)  # BGR格式，蓝色边框，线宽2
            
            # 更新显示
            cv.imshow("full_animation", display_img)
            
            # 处理鼠标左键点击事件
            if event == cv.EVENT_LBUTTONDOWN:
                if current_region is not None:
                    pos_start = current_region['pos_start']
                    print(f"点击区域 - pos起始位置: (row={pos_start[0]}, col={pos_start[1]})")
                    cv.namedWindow("current_image", cv.WINDOW_NORMAL)
                    cv.imshow("current_image",self.get_pos_image(pos_start))
                else:
                    print(f"点击位置 ({x}, {y}) 不在任何区域内")
        
        # 设置鼠标回调
        cv.namedWindow("full_animation", cv.WINDOW_NORMAL)
        cv.setMouseCallback("full_animation", mouse_callback)
        
        # 显示初始图像（不绘制任何矩形框）
        cv.imshow("full_animation", img)
        print("提示: 移动鼠标到区域上会显示框选，点击鼠标左键将输出对应区域的pos起始位置")
        cv.waitKey(0)
        cv.destroyAllWindows()

    def get_max_window_capacity(self):
        """计算window_value在棋盘模式下最多可以填充的元素数量"""
        total_cells = self.window_value_rows * self.window_value_cols
        return (total_cells + 1) // 2
    
    def write_window_value(self,index,result):
        """
        将结果写入window_value，使用棋盘模式填充，确保相邻位置不会同时有值
        
        参数:
            index: 一维数组的元素索引（从0开始）
            result: 检测结果，"NG" 或 "OK"
        
        填充规则:
            - front: 填充 (row+col) % 2 == 1 的位置，起始位置 [0][1]
            - back: 填充 (row+col) % 2 == 0 的位置，起始位置 [0][0]
        """
        pattern_mod = 1 if self.side == "front" else 0
        
        # 生成所有满足棋盘模式的位置列表
        valid_positions = [(row, col) for row in range(self.window_value_rows) 
                          for col in range(self.window_value_cols) 
                          if (row + col) % 2 == pattern_mod]
        
        # 根据index获取对应的行列位置并写入结果值
        current_row, current_col = valid_positions[index]
        print(pattern_mod,current_row,current_col)
        self.window_value[current_row][current_col] = 3 if result == "NG" else 2

    def clear_window_value(self):
        self.window_value = np.zeros((self.window_value_rows,self.window_value_cols))

    def write_log(self,Lot_ID,Strip_ID,current_pos,defect_result):
        for type in defect_result:
            self.log_file["Activate_Defect_Type"].join(type)

class counter():
    def __init__(self) -> None:
        self.lot_id = "None"
        self.total_count = 0
        self.each_count = {
            "mark" : 0,
            "size" : 0,
            "ball" : 0,
            "shift": 0,
            "scratch": 0,
        }
    def get_total(self) -> int:
        return self.total_count
    def add(self,prodcut_type):
        self.each_count[prodcut_type] += 1
        self.total_count +=1
    def get_yeild(self) -> float:
        total_ng = 0
        for key,value in self.each_count:
            total_ng += value
        yeild_rate = total_ng/self.total_count*100
        return yeild_rate


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def flip270(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def flip90(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr

def flip_horizontal_in_place(arr):
    for row in arr:
        row = row[::-1]
    return arr

def vaule_transmit(value,mode):
    value = flip_horizontal_in_place(value)
    value =np.array(value)
    if mode == 1 :
        return value.flatten()
    if mode == 2:
        return flip270(value).flatten()
    if mode == 3:
        return flip180(value).flatten()
    if mode == 4:
        return flip90(value).flatten()

def Sucker_Std():
        sucker_pos = []
        with open(r"result/sucker_coord.txt") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                sucker_pos.append(line.split(" "))
        for i in range(len(sucker_pos)):
            for j in range(len(sucker_pos[i])):
                sucker_pos[i][j] = int(sucker_pos[i][j])
        print(sucker_pos)
        return sucker_pos

def sucker_detect(image:np.ndarray,pixel_size:float):
    """
    Args:
        image: np.ndarray 输入图像灰色
        pixel_size: float 像素尺寸
    Returns:
        sucker_center: tuple 嘴中心坐标
        image_result: np.ndarray 结果图像
    """
    _,image_thresh = cv.threshold(image,240,255,cv.THRESH_BINARY)

    image_result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    image_open = cv.morphologyEx(image_thresh,cv.MORPH_OPEN,cv.getStructuringElement(cv.MORPH_RECT,(20,20)))
    cv.imshow("image_open", image_open)
    cv.waitKey(0)
    cv.destroyAllWindows()
    image_contours,_ = cv.findContours(image_open,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    image_center = (int(image.shape[1]/2),int(image.shape[0]/2))
    for i in range(0,len(image_contours)):
        sucker_rect = cv.minAreaRect(image_contours[i])
        sucker_area = cv.contourArea(image_contours[i])
        distence = abs(sucker_rect[0][1] - image_center[1])

        if distence <100 and sucker_area >30000:
            bounding_box = cv.boundingRect(image_contours[i])
            cv.rectangle(image_result, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (0, 255, 0), 2)
            sucker_center = (int(sucker_rect[0][0]),int(sucker_rect[0][1]))
            cv.line(image_result, (sucker_center[0]-200, sucker_center[1]), (sucker_center[0]+200, sucker_center[1]), (0, 0, 255), 2)
            cv.line(image_result, (sucker_center[0], sucker_center[1]-200), (sucker_center[0], sucker_center[1]+200), (0, 0, 255), 2)
            cv.line(image_result, (image_center[0]-200, image_center[1]), (image_center[0]+200, image_center[1]), (0, 255, 0), 2)
            cv.line(image_result, (image_center[0], image_center[1]-200), (image_center[0], image_center[1]+200), (0, 255, 0), 2)
            dis_x = int((sucker_center[0] - image_center[0])*pixel_size*1000)
            dis_y = int((sucker_center[1] - image_center[1])*pixel_size*1000)
            angle = sucker_rect[2]
            if angle > 45:
                angle = angle - 90
            angle = int(angle*1000)
            return (3,dis_y,dis_x,angle),image_result
    return (0,0,0,0),image_result

def Cross_detect(image:np.ndarray):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)[50:-50,100:-175]
    image_result = image.copy()
    mean_val = cv.mean(image_gray)[0]
    if 40 < mean_val < 43:
        _, binary = cv.threshold(image_gray, 35, 255, cv.THRESH_BINARY_INV)
        blurred   = cv.GaussianBlur(binary, (5, 5), 0)
        canny     = cv.Canny(blurred, 30, 120)
        lines     = cv.HoughLinesP(binary, 3, np.pi * 2 / 180, threshold=20, minLineLength=80, maxLineGap=10)
    else:
        blurred   = cv.GaussianBlur(image_gray, (5, 5), 0)
        canny     = cv.Canny(blurred, 30, 120)
        lines     = cv.HoughLinesP(canny, 3, np.pi * 2 / 180, threshold=20, minLineLength=56, maxLineGap=15)
    
    slope_threshold_min = -20
    slope_threshold_max = 20

    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算斜率
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            slope = int(slope * 10) if slope != float('inf') else slope
            if slope_threshold_min < abs(slope) and abs(slope) < slope_threshold_max and slope != 0:
                filtered_lines.append(line)
                
    if len(filtered_lines)>3:
        cv.putText(image_result, "NG", (200, 250), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image_result, (x1+100, y1+50), (x2+100, y2+50), (0, 0, 255), 2)
        return "NG",image_result
    else:
        return "OK",image_result

def template_match(image,golden_template,template_threshold,):
    """
    Function: 模板匹配函数，并去除重复匹配点
    Args:
        image: np.ndarray 灰度原图像
        golden_template: np.ndarray 金标准模板
        template_threshold: float 模板匹配阈值
    Returns:
        image_result: np.ndarray 结果图像
        temp_list: list 匹配点列表
    """
    image_result = image.copy()
    if len(image.shape) == 3:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()
    if len(golden_template.shape) == 3:
        golden_template_gray = cv.cvtColor(golden_template, cv.COLOR_BGR2GRAY)
    else:
        golden_template_gray = golden_template.copy()
    golden_template_gray = cv.GaussianBlur(golden_template_gray, (5, 5), 0)
    image_gray = cv.GaussianBlur(image_gray, (5, 5), 0)
    match_result = cv.matchTemplate(image_gray, golden_template_gray, cv.TM_CCOEFF_NORMED)
    loc = np.where(match_result >= template_threshold)
    
    # 收集所有匹配点及其相似度分数
    matches = []
    for pt in zip(*loc[::-1]):
        x, y = pt
        confidence = match_result[y, x]  # match_result的形状是(y, x)
        matches.append((x, y, confidence))
    
    # 按相似度降序排序，确保相似度高的优先处理（用于去重）
    matches.sort(key=lambda x: x[2], reverse=True)
    
    ###去重：保留相似度最高的匹配
    min_distance = min(golden_template.shape[1], golden_template.shape[0])
    temp_list= []
    for x, y, confidence in matches:
        too_close = False
        for existing_x, existing_y in temp_list:
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            temp_list.append((x, y))
    
    # 按位置排序：左上角为0，然后往右下递增
    # Y方向允许模板高度的1/2作为容差，即Y坐标差在容差范围内的点视为同一行
    template_height = golden_template.shape[0]
    y_tolerance = template_height / 2.0
    
    # 使用lambda表达式排序：将Y坐标按容差分组，然后按X坐标排序
    # 将Y坐标按容差向下取整，使得容差范围内的点被视为同一行
    temp_list.sort(key=lambda pt: (int(pt[1] / y_tolerance) if y_tolerance > 0 else int(pt[1]), pt[0]))  # 先按Y分组，再按X坐标排序
    
    return image_result,temp_list


def _mask_roi_regions(image_gray: np.ndarray, image_result: np.ndarray, roi_blocks: list) -> np.ndarray:
    """
    屏蔽ROI区域
    
    Args:
        image_gray: 灰度图像
        image_result: 结果图像（BGR格式）
        roi_blocks: 需要屏蔽的区域列表，每个元素为 (x, y, w, h) 格式的元组或列表
    
    Returns:
        屏蔽后的灰度图像
    """
    # 确保数组是可写的，如果是只读数组则创建副本
    if not image_gray.flags.writeable:
        image_gray = image_gray.copy()
    if not image_result.flags.writeable:
        image_result = image_result.copy()
    
    for roi in roi_blocks:
        try:
            # 验证ROI格式：必须是 (x, y, w, h) 格式的元组或列表
            if isinstance(roi, (tuple, list)) and len(roi) >= 4:
                x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            else:
                continue
            
            # 边界检查：确保ROI坐标在图像范围内
            h_img, w_img = image_gray.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            
            # 屏蔽该区域：将像素值设置为0，并在结果图像中用粉色矩形标记
            image_gray[y:y+h, x:x+w] = 0
            cv.rectangle(image_result, (x, y), (x + w, y + h), (255, 0, 255), -1)
        except (ValueError, IndexError, TypeError) as e:
            print(f"警告: 无法解析roi {roi}: {e}")
            continue
    
    return image_gray,image_result


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


def detect_size(image_bga: np.ndarray, size_tolerance: float, std_size: tuple[float, float], min_threshold: int, max_threshold: int, pixel_size: float) -> tuple[tuple[int, int], tuple[int, int, int, int], np.ndarray, list]:
    """
    检测产品外形尺寸（使用投影曲线方法进行边界检测）
    
    Args:
        image_bga: 输入的灰度图像
        size_tolerance: 产品尺寸容差（mm），可以是单个浮点数或 (tolerance_x, tolerance_y) 元组
        std_size: 标准产品尺寸 (width, height) 像素单位
        min_threshold: 二值化下阈值（灰度值区间下限）
        max_threshold: 二值化上阈值（灰度值区间上限）
        pixel_size: 像素尺寸（mm/pixel）
    Returns:
        tuple: (bool, product_size, size_box, box_points)
    """
    image_gray = image_bga.copy()
    h, w = image_gray.shape
    image_gray = cv.equalizeHist(image_gray)
    image_binary = cv.inRange(image_gray, min_threshold, max_threshold)
    image_binary = cv.bitwise_not(image_binary)
    # 默认参数：差分步长0.7，平滑窗口5
    gradient_dx = 0.7
    smooth_window = 5
    
    # 自适应ROI宽度：根据图像尺寸调整
    roi_width = max(30, min(80, int(min(h, w) * 0.1)))
    
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
        roi_image = image_binary[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        is_horizontal = roi_name in ['top', 'bottom']
        is_reverse = roi_name in ['top', 'left']
        
        # 计算投影曲线
        proj_curve = calculate_projection_curve(roi_image, 'horizontal' if is_horizontal else 'vertical')
        
        # 根据搜索方向处理投影曲线
        proj_curve_for_detection = proj_curve[::-1] if is_reverse else proj_curve
        
        # 检测边界
        boundary_offset = detect_boundary_subpixel(
            proj_curve_for_detection,
            need_reverse=is_reverse,
            dx=gradient_dx,
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
        return False, (0.0, 0.0), (0, 0, 0, 0), np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    
    width_pixel = right_boundary - left_boundary
    height_pixel = bottom_boundary - top_boundary
    
    x_min, y_min = float(left_boundary), float(top_boundary)
    x_max, y_max = float(right_boundary), float(bottom_boundary)
    
    size_box = (int(x_min), int(y_min), int(width_pixel), int(height_pixel))
    box_points = np.array([
        [x_min, y_min], [x_max, y_min],
        [x_max, y_max], [x_min, y_max]
    ], dtype=np.float32)
    
    product_size = (width_pixel * pixel_size, height_pixel * pixel_size)
    
    # 处理容差参数
    if isinstance(size_tolerance, (tuple, list)) and len(size_tolerance) >= 2:
        tolerance_x, tolerance_y = size_tolerance[0], size_tolerance[1]
    else:
        tolerance_x = tolerance_y = float(size_tolerance) if isinstance(size_tolerance, (int, float)) else 0.0
    
    # 判断尺寸是否合格
    result = False
    if width_pixel > 0 and height_pixel > 0:
        std_width, std_height = std_size
        if std_width > 0 and std_height > 0:
            std_width_mm = std_width * pixel_size
            std_height_mm = std_height * pixel_size
            result = (abs(std_width_mm - product_size[0]) <= tolerance_x and 
                     abs(std_height_mm - product_size[1]) <= tolerance_y)
        else:
            result = True
    
    return result, product_size, size_box, box_points




def detect_balls(image_bga: np.ndarray, min_threshold: int, max_threshold: int, 
                 min_area: int, max_area: int, radius_tolerance:float,
                 ball_search_roi: tuple[int, int, int, int],std_radius:float,pixel_size:float) -> tuple[int, float, list, np.ndarray]:
    """
    检测BGA锡球
    
    Args:
        image_bga: 输入的灰度图像
        min_threshold: 二值化处理的最小阈值
        max_threshold: 二值化处理的最大阈值
        min_area: 锡球轮廓的最小面积（像素）
        max_area: 锡球轮廓的最大面积（像素）
        area_tolerance: 面积容差系数，用于判断锡球面积是否在允许范围内
        ball_count_limit: 期望的锡球数量
        size_box: 产品边界框 (x_min, y_min, width, height)
        std_radius: 标准半径
        pixel_size: 像素尺寸
        Returns:
            tuple: (result_list)
            - result_list: dict 检测结果列表
                - "OK": list 合格锡球边界框列表
                - "NG": list 不合格锡球边界框列表
                - "Ball Count": int 检测到的锡球数量

    """
    
    image_gray = image_bga.copy()
    # 使用不同的阈值重新二值化图像，用于检测锡球
    image_gray_blur = cv.medianBlur(image_gray, 3)
    # 使用cv.inRange将指定区间内的像素值设为255，区间外的设为0
    image_binary_ball = cv.inRange(image_gray_blur, min_threshold, max_threshold)
    contours, hierarchy = cv.findContours(image_binary_ball, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    # 初始化变量
    filtered_ball_image = []
    contour_count = 0
    total_area = 0
    avg_area = 0
    
    # 预计算产品边界范围（留5像素边距），用于判断锡球是否在产品范围内
    # 确保 ball_search_roi 是整数类型（处理可能从配置文件读取的字符串）
    x_min_bound = ball_search_roi[0] + 5
    y_min_bound = ball_search_roi[1] + 5
    x_max_bound = ball_search_roi[0] + ball_search_roi[2] - 5
    y_max_bound = ball_search_roi[1] + ball_search_roi[3] - 5
    
    # 遍历所有轮廓，筛选出有效的锡球
    for contour in contours:
        area = cv.contourArea(contour)
        
        # 提前进行面积检查，避免不必要的moments计算
        if not (min_area <= area <= max_area):
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
    
    # 计算平均面积和标准半径（避免除零错误）
    avg_area = total_area / contour_count if contour_count > 0 else 0
    
    # 检测锡球面积和圆度
    result_list = {"OK": [], "NG": [],"Ball Count": contour_count, "OK_details": [], "NG_details": []}
    radius_list = []
    all_contours = []  # 存储所有球的轮廓，用于detect_shift
    for item in filtered_ball_image:
        ball, contour, box, area = item
        
        # 计算最小外接圆半径
        # cv.minEnclosingCircle() 返回 ((center_x, center_y), radius)
        # 所以 [1] 是半径，[0] 是圆心坐标
        (center_x, center_y), radius = cv.minEnclosingCircle(contour)
        radius_list.append(radius)
        # 计算面积偏差
        area_diff = abs(area - avg_area)
        
        # 计算半径偏差
        radius_diff = abs(radius - std_radius)
        
        # 计算半径和面积（转换为mm）
        radius_mm = radius * pixel_size
        area_mm2 = area * pixel_size * pixel_size
        
        # 创建详细信息字典
        ball_detail = {
            "box": box,  # (x, y, w, h)
            "center": (int(center_x), int(center_y)),
            "radius_pixel": radius,
            "radius_mm": radius_mm,
            "area_pixel": area,
            "area_mm2": area_mm2,
            "radius_diff": radius_diff * pixel_size,  # 转换为mm
            "area_diff": area_diff * pixel_size * pixel_size  # 转换为mm²
        }
        
        if radius_diff > radius_tolerance:
            result_list["NG"].append(box)
            result_list["NG_details"].append(ball_detail)
        else:
            result_list["OK"].append(box)
            result_list["OK_details"].append(ball_detail)
        
        # 保存轮廓（包括OK和NG的所有球）
        all_contours.append(contour)

    # 计算平均半径，如果列表为空则返回0
    avg_radius = np.mean(radius_list) if len(radius_list) > 0 else 0.0
    # 添加轮廓列表到返回结果中
    result_list["contours"] = all_contours
    return result_list, avg_radius


def detect_shift(ball_contours: list, size_box_or_contour, pixel_size: float) -> tuple[float, float, tuple, tuple]:
    """
    检测产品中所有球的中心与产品尺寸中心的偏移量（无绘制版本）
    通过直接计算所有锡球中心的平均值来确定产品中心，然后与尺寸中心计算整体偏移
    
    算法：
    1. 计算所有球轮廓的中心点（使用最小外接圆的圆心）
    2. 计算所有球中心的平均位置作为产品中心
    3. 计算产品尺寸的中心位置
    4. 偏移量 = 球中心平均位置 - 尺寸中心位置
    
    Args:
        ball_contours: 产品中所有球的轮廓列表，每个元素为np.ndarray格式的轮廓
        size_box_or_contour: 产品的尺寸box或尺寸轮廓
            - 如果为tuple/list: (x_min, y_min, width, height) 格式的尺寸box
            - 如果为np.ndarray: 产品的尺寸轮廓（4个顶点坐标）
        pixel_size: 像素尺寸（mm/pixel），用于将偏移量转换为实际单位
    
    Returns:
        tuple: (shift_x, shift_y, ball_center, size_center)
            - shift_x: float X方向的偏移量（mm），球中心X - 尺寸中心X
            - shift_y: float Y方向的偏移量（mm），球中心Y - 尺寸中心Y
            - ball_center: tuple[float, float] 球中心平均位置 (x, y) 像素坐标
            - size_center: tuple[float, float] 尺寸中心位置 (x, y) 像素坐标
    """
    # 计算所有球轮廓的中心（使用最小外接圆的圆心，对圆形球更准确）
    ball_centers = []
    for contour in ball_contours:
        try:
            # 使用最小外接圆计算圆心（对圆形球比质心更准确）
            (center_x, center_y), radius = cv.minEnclosingCircle(contour)
            ball_centers.append((float(center_x), float(center_y)))
        except (cv.error, AttributeError):
            continue
    
    # 如果没有有效的球中心，返回零偏移
    if len(ball_centers) == 0:
        return 0.0, 0.0, (0.0, 0.0), (0.0, 0.0)
    
    # 计算产品尺寸边界和中心
    if isinstance(size_box_or_contour, (tuple, list)) and len(size_box_or_contour) >= 4:
        # 处理尺寸box格式: (x_min, y_min, width, height)
        size_center_x = float(size_box_or_contour[0]) + float(size_box_or_contour[2]) / 2.0
        size_center_y = float(size_box_or_contour[1]) + float(size_box_or_contour[3]) / 2.0
        size_center = (size_center_x, size_center_y)
    elif isinstance(size_box_or_contour, np.ndarray) and size_box_or_contour.shape[0] >= 4:
        # 处理尺寸轮廓格式: np.ndarray，包含4个顶点坐标
        contour_points = size_box_or_contour.reshape(-1, 2)
        x_min_bound = float(np.min(contour_points[:, 0]))
        y_min_bound = float(np.min(contour_points[:, 1]))
        x_max_bound = float(np.max(contour_points[:, 0]))
        y_max_bound = float(np.max(contour_points[:, 1]))
        size_center_x = (x_min_bound + x_max_bound) / 2.0
        size_center_y = (y_min_bound + y_max_bound) / 2.0
        size_center = (size_center_x, size_center_y)
    else:
        # 无效的输入格式
        return 0.0, 0.0, (0.0, 0.0), (0.0, 0.0)
    
    # 将球中心转换为numpy数组便于处理
    ball_centers_array = np.array(ball_centers)
    
    # 计算所有球中心的平均位置（产品中心）
    ball_center_x = float(np.mean(ball_centers_array[:, 0]))
    ball_center_y = float(np.mean(ball_centers_array[:, 1]))
    ball_center = (ball_center_x, ball_center_y)
    
    # 计算偏移量（像素单位）
    shift_x_pixel = ball_center_x - size_center_x
    shift_y_pixel = ball_center_y - size_center_y
    
    # 转换为实际单位（mm）
    shift_x = shift_x_pixel * pixel_size
    shift_y = shift_y_pixel * pixel_size
    
    return shift_x*0.7, shift_y*0.7, ball_center, size_center

def detect_mark(image_bga: np.ndarray, mark_roi: list, 
                min_threshold: int, max_threshold: int, min_mark_area: int,auto_or_manual:str) -> tuple[bool,np.ndarray,float]:
    """
    检测Mark标记
    Args:
        image_bga: 输入的灰度图像
        mark_roi: Mark区域列表，每个元素为 (x, y, w, h) 格式的元组或列表
        min_threshold: 二值化处理的最小阈值
        max_threshold: 二值化处理的最大阈值
        min_mark_area: Mark区域的最小面积
        auto_or_manual: "auto" 自动检测，"manual" 手动检测
    Returns:
        - defect: bool 是否存在缺陷
        - mark_contour: np.ndarray 标记区域轮廓
        - mark_area: float 标记区域面积（像素）
    """
    image_gray = image_bga.copy()
    mark_contour = None
    mark_area = 0.0
    # 获取Mark区域
    image_gray_mark = image_gray[mark_roi[0][1]:mark_roi[0][1]+mark_roi[0][3],mark_roi[0][0]:mark_roi[0][0]+mark_roi[0][2]]
    if auto_or_manual == "auto":
        mean_val = int(cv.mean(image_gray_mark)[0]*1.05)
        # 使用cv.inRange将指定区间内的像素值设为255，区间外的设为0
        image_binary_mark = cv.inRange(image_gray_mark, mean_val, max_threshold)
    if auto_or_manual == "manual":
        # 使用cv.inRange将指定区间内的像素值设为255，区间外的设为0
        image_binary_mark = cv.inRange(image_gray_mark, min_threshold, max_threshold)
    ###去除杂点
    # 使用形态学开运算去除细小的杂点（先腐蚀后膨胀）
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    image_binary_mark = cv.morphologyEx(image_binary_mark, cv.MORPH_OPEN, kernel)
    contours, hierarchy = cv.findContours(image_binary_mark, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > min_mark_area:
            mark_contour = contour
            mark_area = area
            return True, mark_contour, mark_area
            
    return False, mark_contour, mark_area

def detect_scratch(image_bga: np.ndarray, scratch_roi: list, min_threshold: int, max_threshold: int, 
                   scratch_length_threshold: float, pixel_size: float, roi_blocks: list = None) -> tuple[list, list]:
    """
    检测划痕：在指定区域检测细小划痕，当划痕长度大于指定长度时标记，并判定为不良
    
    Args:
        image_bga: 输入的灰度图像
        scratch_roi: 划痕检测区域列表，每个元素为 (x, y, w, h) 格式的元组或列表
        min_threshold: 二值化处理的最小阈值
        max_threshold: 二值化处理的最大阈值
        scratch_length_threshold: 划痕长度阈值（mm），超过此长度的划痕将被判定为不良
        pixel_size: 像素尺寸（mm/pixel），用于将像素长度转换为实际长度
        roi_blocks: 需要屏蔽的区域列表（可选），每个元素为 (x, y, w, h) 格式
    
    Returns:
        tuple: (defect_type, ng_scratch_contours)
            - defect_type: list 缺陷类型列表，["OK"] 或 ["NG", "Scratch"]
            - ng_scratch_contours: list NG的划痕轮廓列表，每个元素为numpy数组
    """
    defect_type = ["OK"]
    image_gray = image_bga.copy()
    
    # 屏蔽ROI区域（如果提供了roi_blocks）
    if roi_blocks is not None and isinstance(roi_blocks, list) and len(roi_blocks) > 0:
        # 创建一个临时图像用于屏蔽
        temp_image = cv.cvtColor(image_bga, cv.COLOR_GRAY2BGR)
        image_gray, _ = _mask_roi_regions(image_gray, temp_image, roi_blocks)
    
    # 检查是否有划痕检测区域
    if not isinstance(scratch_roi, list) or len(scratch_roi) == 0:
        return defect_type, []
    
    # 将划痕长度阈值从mm转换为像素
    scratch_length_threshold_pixels = scratch_length_threshold / pixel_size if pixel_size > 0 else scratch_length_threshold
    
    # 存储检测到的NG划痕轮廓
    ng_scratch_contours = []
    
    # 在每个划痕检测区域内进行检测
    for roi_idx, roi in enumerate(scratch_roi):
        if not isinstance(roi, (tuple, list)) or len(roi) < 4:
            continue
        
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        
        # 边界检查
        h_img, w_img = image_gray.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        
        # 提取ROI区域
        roi_image = image_gray[y:y+h, x:x+w]
        if roi_image.size == 0:
            continue
        
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
            if area < 10:  # 过滤太小的轮廓
                continue
            
            # 计算轮廓的最小外接矩形
            rect = cv.minAreaRect(contour)
            width, height = rect[1]
            if width < height:
                width, height = height, width  # 确保width是长边
            
            # 计算长宽比，划痕通常是细长的
            aspect_ratio = width / height if height > 0 else 0
            
            # 计算轮廓长度（使用轮廓周长）
            perimeter = cv.arcLength(contour, closed=False)
            
            # 或者使用最小外接矩形的长度作为划痕长度
            scratch_length_pixels = max(width, height)
            scratch_length_mm = scratch_length_pixels * pixel_size
            
            # 判断是否为划痕：长宽比大于阈值 或 长度超过阈值
            # 划痕特征：细长（长宽比大）或长度超过阈值
            if aspect_ratio > 3.0 or scratch_length_mm > scratch_length_threshold:
                # 如果长度超过阈值，判定为不良并保存轮廓
                if scratch_length_mm > scratch_length_threshold:
                    # 将轮廓坐标转换回原图坐标系
                    contour_global = contour + np.array([x, y], dtype=np.int32)
                    ng_scratch_contours.append(contour_global)
                    defect_type[0] = "NG"
                    if "Scratch" not in defect_type:
                        defect_type.append("Scratch")
    
    return defect_type, ng_scratch_contours

def Defect_3(image:np.ndarray,pixel_size:float):
    _,image_thresh = cv.threshold(image,240,255,cv.THRESH_BINARY)
    image_open = cv.morphologyEx(image_thresh,cv.MORPH_OPEN,cv.getStructuringElement(cv.MORPH_RECT,(20,20)))
    image_contours,_ = cv.findContours(image_open,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    image_product = 0
    product_center = [0,0]
    max_product_area = 0
    max_index = 0
    image_center = (int(image.shape[1]/2),int(image.shape[0]/2))
    
    for i in range(0,len(image_contours)):
        sucker_rect = cv.minAreaRect(image_contours[i])
        sucker_area = cv.contourArea(image_contours[i])
        distence = abs(sucker_rect[0][1] - image_center[1])
        if distence <100 and sucker_area >30000:
            x, y, w, h = cv.boundingRect(image_contours[i])
            image_product = image.copy()[y:y+h,x:x+w]
            image_product_color = cv.cvtColor(image_product,cv.COLOR_GRAY2BGR)

    _,image_product_thresh = cv.threshold(image_product,150,255,cv.THRESH_BINARY_INV)
    if image_product_thresh.dtype != np.uint8:
        image_product_thresh_uint8 = image_product_thresh.astype(np.uint8)
    else:
        image_product_thresh_uint8 = image_product_thresh
    image_product_contours, _ = cv.findContours(image_product_thresh_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for i in range(0,len(image_product_contours)):
        product_area = cv.contourArea(image_product_contours[i])
        if product_area > max_product_area:
            max_product_area = product_area
            max_index = i
    product_rect = cv.minAreaRect(image_product_contours[max_index])
    bounding_box = cv.boundingRect(image_product_contours[max_index])
    try:
        cv.rectangle(image_product_color,(bounding_box[0],bounding_box[1]),(bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]),(0,255,0),2)
    except:
        image_product_color = image.copy()
        return (0,0,0,0),image_product_color
    product_center[0]= int(product_rect[0][0])
    product_center[1] = int(product_rect[0][1])
    image_product_center = (int(image_product.shape[1]/2),int(image_product.shape[0]/2))
    cv.line(image_product_color,(product_center[0]-50,product_center[1]),(product_center[0]+50,product_center[1]),(0,0,255),2)
    cv.line(image_product_color,(product_center[0],product_center[1]-100),(product_center[0],product_center[1]+100),(0,0,255),2)
    cv.line(image_product_color,(image_product_center[0]-50,image_product_center[1]),(image_product_center[0]+50,image_product_center[1]),(0,255,0),2)
    cv.line(image_product_color,(image_product_center[0],image_product_center[1]-100),(image_product_center[0],image_product_center[1]+100),(0,255,0),2)
    dis_x = int((image_product_center[0] - product_center[0])*pixel_size*1000) 
    dis_y = int((image_product_center[1] - product_center[1])*pixel_size*1000)
    angle = product_rect[2]
    if angle > 45:
        angle = angle - 90
    angle = int(angle*1000)
    image_result = cv.cvtColor(image,cv.COLOR_GRAY2BGR)
    image_result[y:y+h,x:x+w] = image_product_color
    return(3,dis_y,dis_x,angle),image_result

def Nrotate(angle,valuex,valuey,pointx,pointy):
 nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
 nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
 return round(nRotatex,5), round(nRotatey,5)#####Return the rotated point counter clockwise

def Srotate(angle,valuex,valuey,pointx,pointy):
 sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
 sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
 return round(sRotatex,5),round(sRotatey,5)#####Return the rotated point clockwise

def offset(image:np.ndarray):

    _,image_thresh = cv.threshold(image,150,255,cv.THRESH_BINARY)
    image_open = cv.morphologyEx(image_thresh,cv.MORPH_OPEN,cv.getStructuringElement(cv.MORPH_RECT,(20,20)))
    image_contours,_ = cv.findContours(image_open,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    for i in range(0,len(image_contours)):
        sucker_rect = cv.minAreaRect(image_contours[i])
        sucker_area = cv.contourArea(image_contours[i])
        distence = abs(sucker_rect[0][1] - 512)
        if distence <100 and sucker_area >30000:
            x, y, w, h = cv.boundingRect(image_contours[i])
    center = (x+w/2,y+h/2)
    cv.cvtcvtColor(image,cv.COLOR_GRAY2BGR)
    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv.line(image,(center[0]-50,center[1]),(center[0]+50,center[1]),(0,0,255),2)
    cv.line(image,(center[0],center[1]-50),(center[0],center[1]+50),(0,0,255),2)

    return center,image

def create_alternating_array(rows, cols, start_element, values=(0, 3)):
    """
    创建交替填充的数组
    
    Args:
        rows: 行数
        cols: 列数
        start_element: 起始元素（0或1），决定交替模式从哪个值开始
        values: 用于交替的两个值，默认为(0, 1)。例如：(0, 1), (1, 0), (2, 3)等
    
    Returns:
        一个rows x cols的数组，按照交替模式填充values中的两个值
    """
    start = int(start_element) % 2
    row_indices = np.arange(rows)[:, np.newaxis]
    col_indices = np.arange(cols)
    pattern = (row_indices + col_indices + start) % 2
    # 将0和1的模式映射到用户指定的两个值
    result = np.where(pattern == 0, values[0], values[1])
    return result



def calculate_write_positions(target_array, window_array):
    """
    计算所有可写入位置（使用不重叠滑动，步长等于窗口大小）
    按照蛇形轨迹：从左上到左下，向右移动一格然后往上，到达最上方之后，往右移动再往下
    
    参数:
        target_array: 目标数组（numpy数组）
        window_array: 窗口数组（numpy数组），要赋值的数据块
    
    返回:
        positions: 位置列表，每个元素为 (row_start, col_start, row_end, col_end, 
                 actual_row_end, actual_col_end, source_row_size, source_col_size)
    """
    target_rows, target_cols = target_array.shape
    window_rows, window_cols = window_array.shape
    
    # 强制使用不重叠滑动，步长等于窗口大小
    stride_row = window_rows
    stride_col = window_cols
    
    # 计算列位置
    col_positions = list(range(0, target_cols, stride_col))
    
    positions = []
    
    # 按列遍历，每列内按蛇形移动
    for col_idx, col_start in enumerate(col_positions):
        # 计算当前列的所有行位置
        row_positions_list = list(range(0, target_rows, stride_row))
        
        # 偶数索引列（0, 2, 4...）：从上到下
        # 奇数索引列（1, 3, 5...）：从下到上
        if col_idx % 2 == 0:
            # 从上到下
            row_positions_iter = row_positions_list
        else:
            # 从下到上（反转列表）
            row_positions_iter = reversed(row_positions_list)
        
        for row_start in row_positions_iter:
            # 计算窗口的结束位置
            row_end = row_start + window_rows
            col_end = col_start + window_cols
            
            # 计算实际可写入的区域（考虑边界）
            actual_row_end = min(row_end, target_rows)
            actual_col_end = min(col_end, target_cols)
            
            # 计算需要从 window_array 中取出的部分
            source_row_size = actual_row_end - row_start
            source_col_size = actual_col_end - col_start
            
            # 只记录有效的写入位置
            if source_row_size > 0 and source_col_size > 0:
                positions.append((
                    row_start, col_start, 
                    row_end, col_end,
                    actual_row_end, actual_col_end,
                    source_row_size, source_col_size
                ))
    
    return positions


def sort_results_by_position(results, template_h, template_w):
    """
    按位置对检测结果进行排序（先按行，再按列）
    
    Args:
        results: 检测结果列表，每个结果包含 "x" 和 "y" 键
        template_h: 模板高度，用于计算Y方向容差
        template_w: 模板宽度（未使用，保留兼容性）
    
    Returns:
        排序后的结果列表
    """
    if not results:
        return results
    y_tolerance = template_h / 2.0
    rows = []
    for result in results:
        y = result["y"]
        found_row = False
        for row in rows:
            if abs(y - row["y_center"]) <= y_tolerance:
                row["items"].append(result)
                found_row = True
                break
        if not found_row:
            rows.append({"y_center": y, "items": [result]})
    for row in rows:
        row["items"].sort(key=lambda r: r["x"])
    rows.sort(key=lambda r: r["y_center"])
    sorted_results = []
    for row in rows:
        sorted_results.extend(row["items"])
    return sorted_results


def draw_detection_results(image_result, ok_balls, ng_balls, box_points, 
                           mark_contour, mark_has_defect, defect_type, 
                           mark_roi, is_dry, is_transfer, size_check_enable, ng_scratch_contours=None):
    """
    在图像上绘制检测结果
    
    Args:
        image_result: 要绘制的图像（BGR格式）
        ok_balls: OK球列表，每个元素为 [x, y, w, h]
        ng_balls: NG球列表，每个元素为 [x, y, w, h]
        box_points: 尺寸检测的边界框点（numpy数组 (4, 2) 或列表 [x, y, w, h]）或None
        mark_contour: Mark检测的轮廓（numpy数组）或None
        mark_has_defect: Mark是否有缺陷
        defect_type: 缺陷类型列表，如 ["OK"] 或 ["NG", "Size"]
        mark_roi: Mark ROI区域，用于计算轮廓的全局坐标
        is_dry: 是否为干燥台模式
        is_transfer: 是否为移栽台模式
        size_check_enable: 是否启用尺寸检测
        ng_scratch_contours: NG的划痕轮廓列表，每个元素为numpy数组
    
    Returns:
        绘制后的图像（原地修改，也返回）
    """
    color_red, color_green = (0, 0, 255), (0, 255, 0)
    
    for box in ok_balls:
        cv.rectangle(image_result, (box[0], box[1]), 
                    (box[0] + box[2], box[1] + box[3]), color_green, 2)
    for box in ng_balls:
        cv.rectangle(image_result, (box[0], box[1]), 
                    (box[0] + box[2], box[1] + box[3]), color_red, 2)
    
    if mark_contour is not None and mark_roi and len(mark_roi) > 0:
        mark_contour_global = mark_contour + np.array(
            [mark_roi[0][0], mark_roi[0][1]], dtype=np.int32
        )
        if is_dry:
            mark_color = color_red if mark_has_defect else color_green
        else:
            mark_color = color_green if mark_has_defect else color_red
        cv.drawContours(image_result, [mark_contour_global], -1, mark_color, 2)
    
    # 绘制NG的划痕轮廓
    if ng_scratch_contours is not None and len(ng_scratch_contours) > 0:
        for scratch_contour in ng_scratch_contours:
            if scratch_contour is not None and len(scratch_contour) > 0:
                scratch_contour_int = scratch_contour.astype(np.int32)
                cv.drawContours(image_result, [scratch_contour_int], -1, color_red, 2)
    
    if box_points is not None:
        is_ng = defect_type[0] == "NG"
        h, w = image_result.shape[:2]
        center = (w // 2, h // 2)
        # 处理两种格式：numpy数组 (4, 2) 或列表 [x, y, w, h]
        if isinstance(box_points, (list, tuple)) and len(box_points) == 4:
            # [x, y, w, h] 格式，转换为 (4, 2) 格式
            x, y, w_box, h_box = float(box_points[0]), float(box_points[1]), float(box_points[2]), float(box_points[3])
            box_points_array = np.array([
                [x, y], [x + w_box, y],
                [x + w_box, y + h_box], [x, y + h_box]
            ], dtype=np.float32)
        else:
            # numpy数组 (4, 2) 格式
            box_points_array = np.array(box_points, dtype=np.float32)
        box_points_int = box_points_array.astype(np.int32)
        cv.drawContours(image_result, [box_points_int], 0, 
                      color_red if is_ng else color_green, 3)
        text = ("Size" if "Size" in defect_type else
               "Mark" if "Mark" in defect_type else 
               "Shift" if "Shift" in defect_type else 
               "Scratch" if "Scratch" in defect_type else "Ball_Area") if is_ng else "OK"
        color = color_red if is_ng else color_green
        cv.putText(image_result, text, center, cv.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    
    return image_result


def convert_numpy_obj(obj):
    """
    将numpy对象转换为Python原生类型，用于序列化
    
    Args:
        obj: 要转换的对象，可能是numpy类型、列表、字典等
    
    Returns:
        转换后的Python原生类型对象
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [convert_numpy_obj(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_obj(value) for key, value in obj.items()}
    return obj


def calculate_cpk(data_list, usl, lsl):
    """
    计算CPK值（过程能力指数）
    CPK = min((USL - μ) / (3σ), (μ - LSL) / (3σ))
    
    Args:
        data_list: 数据列表
        usl: 上规格限（Upper Specification Limit）
        lsl: 下规格限（Lower Specification Limit）
    
    Returns:
        CPK值（float）或None（如果无法计算）
    """
    if not data_list or len(data_list) < 2:
        return None
    try:
        mean = np.mean(data_list)
        std = np.std(data_list, ddof=1)  # 使用样本标准差
        if std == 0:
            return None
        cpu = (usl - mean) / (3 * std) if usl is not None else None
        cpl = (mean - lsl) / (3 * std) if lsl is not None else None
        if cpu is not None and cpl is not None:
            return max(cpu, cpl)
        elif cpu is not None:
            return cpu
        elif cpl is not None:
            return cpl
        return None
    except Exception:
        return None


def Defect_fulltray(image: np.ndarray, rows: int, cols: int, 
                    empty_tray_template: np.ndarray = None, 
                    grid_roi: tuple = None,
                    diff_threshold: float = 0.15,
                    brightness_threshold: float = 0.1,
                    feature_threshold: float = 0.05) -> tuple:
    """
    检测满盘装载区域是否有产品
    
    检测策略（多方法融合，提高准确率）：
    1. 差异检测：与空板模板对比，计算像素差异
    2. 特征检测：检测反射亮点（有产品）或椭圆形暗点（空板）
    3. 亮度统计：计算格子内的亮度分布特征
    
    Args:
        image: 输入的BGR或灰度图像
        rows: 网格行数（默认8）
        cols: 网格列数（默认16）
        empty_tray_template: 空板模板图像（BGR或灰度），如果提供则使用差异检测
        grid_roi: 网格ROI区域 (x, y, w, h)，如果提供则只检测该区域
        diff_threshold: 差异检测阈值（0-1），差异超过此值判定为有产品
        brightness_threshold: 亮度差异阈值（0-1），用于判断亮度差异
        feature_threshold: 特征检测阈值（0-1），用于判断特征存在性
    
    Returns:
        tuple: (ret, ret_content, result_matrix, result_image)
            - ret: int 返回码，0表示成功，非0表示失败
            - ret_content: str 返回信息
            - result_matrix: np.ndarray bool类型的二维数组 (rows, cols)，True表示有产品，False表示无产品
            - result_image: np.ndarray 标注了检测结果的彩色图像
    """
    try:
        # 参数验证
        if image is None or image.size == 0:
            return 1, "图像为空", None, None
        
        if rows <= 0 or cols <= 0:
            return 1, f"行列数无效: rows={rows}, cols={cols}", None, None
        
        # 图像预处理
        if len(image.shape) == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image_result = image.copy()
        else:
            image_gray = image.copy()
            image_result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        # 如果提供了grid_roi，裁剪到ROI区域
        if grid_roi is not None and len(grid_roi) >= 4:
            x, y, w, h = int(grid_roi[0]), int(grid_roi[1]), int(grid_roi[2]), int(grid_roi[3])
            h_img, w_img = image_gray.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            image_gray = image_gray[y:y+h, x:x+w]
            image_result = image_result[y:y+h, x:x+w]
        
        # 预处理空板模板（如果提供）
        empty_gray = None
        if empty_tray_template is not None:
            if len(empty_tray_template.shape) == 3:
                empty_gray = cv.cvtColor(empty_tray_template, cv.COLOR_BGR2GRAY)
            else:
                empty_gray = empty_tray_template.copy()
            
            # 如果提供了grid_roi，同样裁剪空板模板
            if grid_roi is not None and len(grid_roi) >= 4:
                x, y, w, h = int(grid_roi[0]), int(grid_roi[1]), int(grid_roi[2]), int(grid_roi[3])
                h_empty, w_empty = empty_gray.shape[:2]
                x = max(0, min(x, w_empty - 1))
                y = max(0, min(y, h_empty - 1))
                w = max(1, min(w, w_empty - x))
                h = max(1, min(h, h_empty - y))
                empty_gray = empty_gray[y:y+h, x:x+w]
            
            # 确保空板模板和当前图像尺寸一致
            if empty_gray.shape != image_gray.shape:
                empty_gray = cv.resize(empty_gray, (image_gray.shape[1], image_gray.shape[0]))
        
        # 计算每个格子的尺寸
        h, w = image_gray.shape[:2]
        cell_height = h / rows
        cell_width = w / cols
        
        # 初始化结果矩阵
        result_matrix = np.zeros((rows, cols), dtype=bool)
        
        # 对每个格子进行检测
        for row in range(rows):
            for col in range(cols):
                # 计算当前格子的坐标
                y_start = int(row * cell_height)
                y_end = int((row + 1) * cell_height)
                x_start = int(col * cell_width)
                x_end = int((col + 1) * cell_width)
                
                # 确保坐标在图像范围内
                y_start = max(0, min(y_start, h - 1))
                y_end = max(y_start + 1, min(y_end, h))
                x_start = max(0, min(x_start, w - 1))
                x_end = max(x_start + 1, min(x_end, w))
                
                # 提取当前格子的ROI
                cell_roi = image_gray[y_start:y_end, x_start:x_end]
                
                if cell_roi.size == 0:
                    result_matrix[row, col] = False
                    continue
                
                # 方法1：差异检测（如果有空板模板）
                diff_score = 0.0
                if empty_gray is not None:
                    empty_cell = empty_gray[y_start:y_end, x_start:x_end]
                    if empty_cell.shape == cell_roi.shape:
                        # 计算归一化差异
                        print("empty:  ",cv.mean(empty_cell)[0],"  ","cell:  ",cv.mean(cell_roi)[0])
                        empty_mean = cv.mean(empty_cell)[0]
                        cell_mean = cv.mean(cell_roi)[0]
                        #diff = cv.absdiff(cell_roi.astype(np.float32), empty_cell.astype(np.float32))
                        diff_score = abs(cell_mean-empty_mean)/empty_mean # 归一化到0-1
                        cv.imshow("diff",cv.absdiff(cell_roi,empty_cell))
                        cv.waitKey()
                        print(diff_score)
                
                # 方法2：特征检测
                # 检测亮点（有产品时通常有反射点）
                # 检测暗点（空板时通常有椭圆形开口）
                cell_blur = cv.GaussianBlur(cell_roi, (5, 5), 0)
                mean_brightness = np.mean(cell_blur) / 255.0
                std_brightness = np.std(cell_blur) / 255.0
                
                # 检测高亮区域（反射点）
                _, bright_mask = cv.threshold(cell_blur, int(mean_brightness * 255 * 1.3), 255, cv.THRESH_BINARY)
                bright_area_ratio = np.sum(bright_mask > 0) / cell_roi.size
                
                # 检测暗区域（椭圆形开口）
                _, dark_mask = cv.threshold(cell_blur, int(mean_brightness * 255 * 0.7), 255, cv.THRESH_BINARY_INV)
                dark_area_ratio = np.sum(dark_mask > 0) / cell_roi.size
                
                # 特征分数：亮点比例 - 暗点比例
                feature_score = bright_area_ratio - dark_area_ratio * 0.5
                
                # 方法3：亮度统计
                # 有产品时，亮度分布通常更不均匀（有反射点）
                brightness_score = std_brightness
                
                # 综合判断（多方法融合）
                has_product = False
                
                if empty_gray is not None:
                    # 如果有空板模板，主要依赖差异检测
                    if diff_score > diff_threshold:
                        has_product = True
                else:
                    # 如果没有空板模板，使用特征检测和亮度统计
                    # 有产品：亮点多、亮度分布不均匀
                    # 空板：暗点多（椭圆形开口）、亮度分布相对均匀
                    if feature_score > feature_threshold:
                        has_product = True
                    elif brightness_score > brightness_threshold and mean_brightness > 0.3:
                        has_product = True
                    elif bright_area_ratio > 0.1:  # 有足够的亮点区域
                        has_product = True
                
                result_matrix[row, col] = has_product
                
                # 在结果图像上标注
                color = (0, 255, 0) if has_product else (0, 0, 255)  # 绿色=有产品，红色=无产品
                cv.rectangle(image_result, (x_start, y_start), (x_end, y_end), color, 2)
                
                # 添加文本标注
                text = "P" if has_product else "E"  # P=Product, E=Empty
                text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x_start + (x_end - x_start - text_size[0]) // 2
                text_y = y_start + (y_end - y_start + text_size[1]) // 2
                cv.putText(image_result, text, (text_x, text_y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return 0, "检测成功", result_matrix, image_result
        
    except Exception as e:
        import traceback
        error_msg = f"检测失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return 1, error_msg, None, None


# ==================== 深度学习模型推理相关函数 ====================

class MobileNetV2Classifier(nn.Module):
    """MobileNetV2分类器"""
    
    def __init__(self, num_classes=2, pretrained=False):
        super(MobileNetV2Classifier, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms(is_train=True, input_size=224):
    """
    获取数据增强变换
    
    Args:
        is_train: 是否为训练集
        input_size: 输入图像尺寸
    """
    if is_train:
        # 训练集：数据增强
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=5),  # 随机旋转±5度
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 随机平移
            transforms.ColorJitter(brightness=0.2, contrast=0.1),  # 亮度对比度调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
    else:
        # 验证集/测试集：只做resize和标准化
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_model(model_path, device='cpu'):
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
        device: 设备 ('cpu' 或 'cuda')
    
    Returns:
        加载后的模型
    """
    model = MobileNetV2Classifier(num_classes=2, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model


def predict_single_image(model, image_path, device='cpu', input_size=150):
    """
    预测单张图像
    
    Args:
        model: 加载的模型
        image_path: 图像路径（字符串）或图像数组（np.ndarray）
        device: 设备 ('cpu' 或 'cuda')
        input_size: 输入图像尺寸
    
    Returns:
        tuple: (prediction, confidence)
            - prediction: 0=无产品, 1=有产品
            - confidence: 置信度 (0-1)
    """
    # 读取图像
    if isinstance(image_path, str):
        # 从文件路径读取
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
    elif isinstance(image_path, np.ndarray):
        # 直接使用传入的图像数组
        if len(image_path.shape) == 3:
            # 如果是BGR图像，转换为灰度
            img = cv.cvtColor(image_path, cv.COLOR_BGR2GRAY)
        else:
            img = image_path.copy()
    else:
        raise ValueError(f"不支持的图像输入类型: {type(image_path)}")
    
    # 转换为RGB
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    
    # 预处理
    transform = get_transforms(is_train=False, input_size=input_size)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    prediction = predicted.item()
    confidence = confidence.item()
    
    return prediction, confidence

def hex_to_string(hex_list):
    """
    将16进制列表转换为字符串
    Args:
        hex_list: 16进制列表
    Returns:
        s: 字符串
    """
    s = ''
    for reg in hex_list:
        if reg == 0:
            break  # 遇到0时停止处理
        # 提取高字节和低字节
        high_byte = (reg >> 8) & 0xFF
        low_byte = reg & 0xFF
        # 转换为ASCII字符（先低字节后高字节，忽略空字符）
        if low_byte != 0:
            s += chr(low_byte)
        if high_byte != 0:
            s += chr(high_byte)
    return s

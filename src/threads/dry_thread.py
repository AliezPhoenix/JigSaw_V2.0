# 从共享导入文件导入所有需要的模块
from src.threads.thread_imports import *
    
class DryThread(QThread):
    _update_image_signal = pyqtSignal(np.ndarray)
    _update_statistics_signal = pyqtSignal(dict)  # 统计更新信号
    _update_message_signal = pyqtSignal(str)

    #———————————————————————————————初始化————————————————————————————————————————————————————————————————
    def __init__(self,params:dict,HM:Hardware_Manager,MM:ModBus_Manager,ui:'MainWindow'):
        super().__init__()
        self.HM = HM    ##硬件管理器
        self.MM = MM    ##Modbus管理器
        self.ui = ui    
        self.ball_detector = BallDetector()  ##球检测器
        self.size_detector = SizeDetector() ##尺寸检测器
        self.shift_detector = ShiftDetector()
        self.mark_detector = MarkDetector() ##标记检测器
        self.scratch_detector = ScratchDetector() ##划痕检测器
        self.template_detector = TemplateDetector() ##模板检测器
        self.bga_strip = Bga_Strip(strip_side="",strip_lot="",strip_sn="",strip_create_time="",params=params)
        self.update_params(params)
        
        # 图像异步保存
        self.image_save_queue = Queue()
        self._init_image_save_thread()
        
        # 垃圾回收计数器
        self.gc_counter = 0
        self.gc_interval = 10  # 每10次循环执行一次垃圾回收
        
        # 暂停/恢复标志
        self.is_paused = False
    
    #——————————————————————————————参数更新函数————————————————————————————————————————————————————————————————————
    def update_params(self,params:dict):
        self.params = params
        self.size_detect_params = {
            "min_threshold": self.params["min_threshold_size"],
            "max_threshold": self.params["max_threshold_size"],
            "allow_tolerance_x": self.params["product_size_tolerance_x"],
            "allow_tolerance_y": self.params["product_size_tolerance_y"],
            "roi_width": 80,
            "std_size": self.params["product_size"],
            "pixel_size": self.params["pixel_size"],
        }
        self.ball_detect_params = {
            "min_threshold": self.params["min_threshold_ball"],
            "max_threshold": self.params["max_threshold_ball"],
            "ball_area_max_threshold": self.params["ball_area_max_threshold"],
            "ball_area_min_threshold": self.params["ball_area_min_threshold"],
            "ball_radius_tolerance": self.params["ball_radius_tolerance"],
            "std_radius": self.params["std_radius"],
            "ball_search_roi": self.params["ball_search_roi"]   
        }

        self.shift_detect_params = {
            "pixel_size": self.params["pixel_size"],
            "error_correction_factor": 0.7,
            "allow_tolerance_x": self.params["shift_x_tolerance"],
            "allow_tolerance_y": self.params["shift_y_tolerance"],
        }

        self.mark_detect_params = {
            "min_threshold": self.params["min_threshold_mark"],
            "max_threshold": self.params["max_threshold_mark"],
            "min_mark_area": self.params["min_mark_area"],
            "auto_threshold_factor": 1.05,
            "pixel_size": self.params["pixel_size"],
            "mark_detect_mode": "manual",
            "mark_roi": self.params["mark_roi"]
        }

        self.template_detect_params = {
            "template_threshold": self.params["template_threshold"],
            "search_roi": self.params["search_roi"]
        }

        self.scratch_detect_params = {
            "min_threshold": self.params.get("min_threshold_scratch", 0),
            "max_threshold": self.params.get("max_threshold_scratch", 255),
            "scratch_length_threshold": self.params.get("scratch_length", 5.0),
            "pixel_size": self.params["pixel_size"],
            "scratch_roi": self.params.get("scratch_roi", []),
            "roi_blocks": self.params.get("roi_block", [])
        }
        
        self.ball_detector.update_params(self.ball_detect_params)
        self.size_detector.update_params(self.size_detect_params)
        self.shift_detector.update_params(self.shift_detect_params)
        self.mark_detector.update_params(self.mark_detect_params)
        self.scratch_detector.update_params(self.scratch_detect_params)
        self.template_detector.update_params(self.template_detect_params)

    #——————————————————————————————图像异步保存函数————————————————————————————————————————————————————
    def _init_image_save_thread(self):
        """初始化图像保存线程"""
        def _save_images_worker():
            while True:
                try:
                    item = self.image_save_queue.get(timeout=1.0)
                    if item is None:
                        break
                    image, filepath = item
                    save_dir = os.path.dirname(filepath)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    cv.imwrite(filepath, image)
                    self.image_save_queue.task_done()
                except Exception as e:
                    from queue import Empty
                    if isinstance(e, Empty):
                        continue
                    print(f"异步保存图像错误: {str(e)}")
                    try:
                        self.image_save_queue.task_done()
                    except:
                        pass
        
        save_thread = threading.Thread(target=_save_images_worker, daemon=True)
        save_thread.start()    
    def _async_save_image(self, image: np.ndarray, filepath: str):
        """异步保存图像"""
        try:
            self.image_save_queue.put((image.copy(), filepath), block=False)
        except Exception as e:
            print(f"添加图像到保存队列失败: {str(e)}") 
    def _generate_image_filename(self,defect_type: str = "UNKNOWN") -> str:
        """生成图像文件名
        
        Args:
            defect_type: 缺陷类型(NG,ORI)
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        lot_id = self.bga_strip.strip_lot if hasattr(self.bga_strip, 'strip_lot') else "LOT"
        sn_id = self.bga_strip.strip_sn if hasattr(self.bga_strip, 'strip_sn') else "SN"
        side = self.bga_strip.strip_side if hasattr(self.bga_strip, 'strip_side') else "front"
        
        save_dir = "Image/Data_Save_Dry"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        

        filename = f"{lot_id}_{sn_id}_Dry_{side}_{defect_type}_{timestamp}.bmp"
        
        return os.path.join(save_dir, filename)
    
    #——————————————————————————————暂停/恢复函数————————————————————————————————————————————————————————————————————
    def pause(self):
        """暂停线程执行"""
        self.is_paused = True
    def resume(self):
        """恢复线程执行"""
        self.is_paused = False
    
    #——————————————————————————————边界检查辅助函数————————————————————————————————————————————————————————————————————
    def _check_image_bounds(self, image: np.ndarray, x: int, y: int, 
                            width: int, height: int) -> bool:
        """检查坐标和尺寸是否在图像范围内
        
        Args:
            image: 图像数组
            x: 起始X坐标
            y: 起始Y坐标
            width: 宽度
            height: 高度
        
        Returns:
            bool: True表示在范围内，False表示越界
        """
        if image is None or len(image.shape) < 2:
            return False
        
        img_h, img_w = image.shape[:2]
        
        # 检查坐标和尺寸有效性
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return False
        
        # 检查是否超出边界
        if x + width > img_w or y + height > img_h:
            return False
        
        return True
    def _check_modbus_data(self, discrete_inputs: list, input_registers: list) -> bool:
        """检查ModBus数据是否完整
        
        Args:
            discrete_inputs: 离散输入列表
            input_registers: 输入寄存器列表
        
        Returns:
            bool: True表示数据完整，False表示数据不完整
        """
        if discrete_inputs is None or len(discrete_inputs) < 4:
            return False
        
        if input_registers is None or len(input_registers) < 3:
            return False
        
        return True
    def _check_template_valid(self, template: np.ndarray) -> bool:
        """检查模板图像是否有效
        
        Args:
            template: 模板图像
        
        Returns:
            bool: True表示有效，False表示无效
        """
        if template is None:
            return False
        
        if len(template.shape) < 2:
            return False
        
        if template.shape[0] <= 0 or template.shape[1] <= 0:
            return False
        
        return True
    
    #——————————————————————————————主循环函数————————————————————————————————————————————————————————————————————
    def run(self):
        trigger_camera_last = 0
        trigger_finished_last = 0
        while True:
            # 检查暂停状态
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            #——————————————————读取ModBus数据————————————————————————————————————
            discrete_input_list = self.MM.read("dry_modbus",address=0,count=4,function_code=cst.READ_DISCRETE_INPUTS)
            input_register_list = self.MM.read("dry_modbus",address=2,count=3,function_code=cst.READ_INPUT_REGISTERS)
            
            # 边界检查：ModBus数据完整性
            if not self._check_modbus_data(discrete_input_list, input_register_list):
                self._update_message_signal("ModBus数据完整性验证失败")
                time.sleep(0.01)
                continue
            
            lot = hex_to_string(self.MM.read("dry_modbus",address=16,count=10,function_code=cst.READ_INPUT_REGISTERS))
            sn =hex_to_string(self.MM.read("dry_modbus",address=26,count=10,function_code=cst.READ_INPUT_REGISTERS))
            trigger_camera ,trigger_front ,trigger_back,trigger_finished = discrete_input_list
            mode,trigger_count,trigger_finished = input_register_list

                        

            #——————————————————————根具正反初始化bga_strip对象————————————————————————————————————
            if trigger_count  == 1:
                self.bga_strip = Bga_Strip(
                    strip_side= "front" if trigger_front == 1 else "back",
                    strip_lot=lot,
                    strip_sn=sn,
                    strip_create_time=datetime.now().strftime("%Y%m%d%H%M%S"),
                    params=self.params
                )   


            #——————————————————————检测触发————————————————————————————————
            if (trigger_camera and not trigger_camera_last) or (trigger_finished and not trigger_finished_last):
                #——————————————————图像采集——————————————————————————————
                ret,msg,image = self.HM.capture_image("dry_cam")
                image = cv.rotate(image,cv.ROTATE_90_CLOCKWISE) ####干燥台相机旋转90度
                if not ret:
                    self._update_message_signal.emit(f"采集图像失败: {msg}")
                    time.sleep(0.01)
                    continue
                image_result = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR)

                #——————————————————模板图像加载————————————————————————————————————
                template =cv.imread(self.params["golden_template_path"])
                
                # 边界检查：模板有效性
                if not self._check_template_valid(template):
                    self._update_message_signal.emit("警告: 模板图像无效，跳过检测")
                    time.sleep(0.01)
                    continue
                
                template_height = template.shape[0]
                template_width = template.shape[1]

                #————————————————————检测流程————————————————————————————————————
                template_pos_list = self.template_detector.detect(template,image,self.template_detect_params)
                current_product_list = []
                for x,y in template_pos_list:
                    # 边界检查：产品图像提取位置
                    if not self._check_image_bounds(image, x, y, template_width, template_height):
                        self._update_message_signal.emit(f"警告: 模板位置 ({x}, {y}) 超出图像边界 ({image.shape[1]}, {image.shape[0]})，跳过")
                        time.sleep(0.01)
                        continue
                    
                    product_image = image[y:y+template_height,x:x+template_width]
                    success, msg, product_info = self._detect_product(x,y,product_image)
                    if not success:
                        filepath_result = self._generate_image_filename("ERROR")
                        filepath_ori = self._generate_image_filename("ORI")
                        self._async_save_image(product_info["product_image_result"], filepath_result)
                        self._async_save_image(product_image, filepath_ori)
                    #——————————————————如果是NG产品，异步保存图像（同时保存原图和检测结果图）————————————————————————————————————
                    if product_info["defect_type"][0] == "NG":
                        defect_type = product_info["defect_type"][1] if len(product_info["defect_type"]) > 1 else "UNKNOWN"
                        filepath_result = self._generate_image_filename(defect_type)
                        filepath_ori = self._generate_image_filename("ORI")
                        self._async_save_image(product_info["product_image_result"], filepath_result)
                        self._async_save_image(product_image, filepath_ori)
                    
                    
                    current_product_list.append(product_info)

                #——————————————————更新显示和统计信息——————————————————————————————————————————————————————
                self.update_display_image(image_result,False)
                self.bga_strip.write(current_product_list,image)
                stats_info = self.bga_strip.get_statistics_info()
                if stats_info:
                    self._update_statistics_signal.emit(stats_info)
                
                #——————————————————完成信号触发，结果分析写入————————————————————————————————————
                if  trigger_finished == 1 and trigger_finished_last == 0:
                    send_data = self.bga_strip.full_value.copy()
                    log_info = self.bga_strip.get_log_info()
                    self._write_modbus_registers(send_data, mode)
                    self.write_log_to_file(log_info)
                self.MM.write(alias="dry_modbus",address = 0,value_list=[1],function_code=cst.WRITE_SINGLE_COIL)
            elif trigger_camera ==0 and trigger_finished == 0:
                self.MM.write(alias="dry_modbus",address = 0,value_list=[0],function_code=cst.WRITE_SINGLE_COIL)
            else:
                self.MM.write(alias="dry_modbus",address = 1,value_list=[0],function_code=cst.WRITE_SINGLE_COIL)
                self.MM.write(alias="dry_modbus",address = 2,value_list=[0],function_code=cst.WRITE_SINGLE_COIL)
            trigger_camera_last = trigger_camera
            trigger_finished_last = trigger_finished
            
            #————————————————————————实时显示画面————————————————————
            if self.ui.radioButton_live_dry.isChecked():
                #——————————————————只有在没有处理触发信号时才更新实时显示，避免重复采集图像————————————————————————————————————
                if not ((trigger_camera and not trigger_camera_last) or (trigger_finished and not trigger_finished_last)):
                    #——————————————————采集图像————————————————————
                    ret,msg, image_live = self.HM.capture_image("dry_cam")
                    if not ret:
                        self._update_message_signal.emit(f"采集图像失败: {msg}")
                        continue
                    image_live = cv.rotate(image_live, cv.ROTATE_90_CLOCKWISE)
                    image_live_bgr = cv.cvtColor(image_live.copy(), cv.COLOR_GRAY2BGR)
                    self.update_display_image(image_live_bgr,True)
            else:
                time.sleep(0.01)
            
            #———————————————————垃圾回收————————————————————————————
            self.gc_counter += 1
            if self.gc_counter >= self.gc_interval:
                gc.collect()
                self.gc_counter = 0
    
    #——————————————————————————————检测产品函数————————————————————————————————————————————————————————————————————
    def _detect_product(self,x,y,product_image:np.ndarray):
        product_info = {
            "x": x,
            "y": y,
            "product_image_result": None,
            "size_result": None,
            "ball_result": None,
            "mark_result": None,
            "shift_result": None,
            "scratch_result": None,
            "defect_type": ["None"],
        }

        if self.params.get("mark_check_enable", False):
            mark_detect_result = self.mark_detector.detect(product_image)
            if not mark_detect_result[0]:
                # 检测失败，记录错误但继续处理
                return False,f"Mark检测失败: {mark_detect_result[1]}",product_info
            else:
                if not mark_detect_result[2]["is_valid"]:
                    # 未检测到Mark → OK
                    product_info["mark_result"] = mark_detect_result
                else:
                    # 检测到Mark → NG
                    product_info["defect_type"].remove("None")
                    product_info["defect_type"].append("Mark")
                    product_info["product_image_result"] = self.draw_detection_results(product_image,product_info)
                    return True,"成功",product_info
                
        if self.params.get("size_check_enable", False):
            size_detect_result = self.size_detector.detect(product_image)
            if not size_detect_result[0]:
                # 检测失败，记录错误但继续处理
                return False,f"Size检测失败: {size_detect_result[1]}",product_info
            else:
                if size_detect_result[2]["is_valid"]:
                    # 尺寸合格 → OK，继续检测
                    product_info["size_result"] = size_detect_result
                else:
                    # 尺寸不合格 → NG
                    product_info["defect_type"].remove("None")
                    product_info["defect_type"].append("Size")
                    product_info["product_image_result"] = self.draw_detection_results(product_image,product_info)
                    return True,"成功",product_info

        if self.params.get("ball_check_enable", False):
            ball_detect_result = self.ball_detector.detect(product_image)
            if not ball_detect_result[0]:
                # 检测失败，记录错误但继续处理
                return False,f"Ball检测失败: {ball_detect_result[1]}",product_info
            else:
                if ball_detect_result[2]["is_valid"]:
                    # 球检测合格 → OK，继续检测
                    product_info["ball_result"] = ball_detect_result
                else:
                    # 球检测不合格 → NG
                    product_info["defect_type"].remove("None")
                    if ball_detect_result[2]["ball_count"] != self.params.get("ball_count", 0):
                        product_info["defect_type"].append("Ball Count")
                    else:
                        product_info["defect_type"].append("Ball")
                    product_info["product_image_result"] = self.draw_detection_results(product_image,product_info)
                    return True,"成功",product_info

        if self.params.get("shift_check_enable", False):
            # Shift检测需要ball_result和size_result
            if product_info["ball_result"] is not None and product_info["size_result"] is not None:
                # shift_detector.detect()期望接收dict参数，从tuple中提取result_dict
                ball_result_dict = product_info["ball_result"][2]
                size_result_dict = product_info["size_result"][2]
                shift_detect_result = self.shift_detector.detect(ball_result_dict, size_result_dict)
                if not shift_detect_result[0]:
                    # 检测失败，记录错误但继续处理
                    return False,f"Shift检测失败: {shift_detect_result[1]}",product_info
                else:
                    if shift_detect_result[2]["is_valid"]:
                        # 偏移合格 → OK，继续检测
                        product_info["shift_result"] = shift_detect_result   
                    else:
                        # 偏移不合格 → NG
                        product_info["defect_type"].remove("None")
                        product_info["defect_type"].append("Shift")
                        product_info["product_image_result"] = self.draw_detection_results(product_image,product_info)
                        return True,"成功",product_info

        if self.params.get("scratch_check_enable", False):
            scratch_detect_result = self.scratch_detector.detect(product_image)
            if not scratch_detect_result[0]:
                # 检测失败，记录错误但继续处理
                return False,f"Scratch检测失败: {scratch_detect_result[1]}",product_info
            else:
                if scratch_detect_result[2]["is_valid"]:
                    # 划痕检测合格 → OK，继续检测
                    product_info["scratch_result"] = scratch_detect_result
                else:
                    # 划痕检测不合格 → NG
                    product_info["defect_type"].remove("None")
                    product_info["defect_type"].append("Scratch")
                    product_info["product_image_result"] = self.draw_detection_results(product_image,product_info)
                    return True,"成功",product_info

        product_info["product_image_result"] = self.draw_detection_results(product_image,product_info)
        product_info["defect_type"] = "OK" if product_info["defect_type"] == "None" else product_info["defect_type"]
        return True,"成功",product_info

    #——————————————————————————————绘制检测结果函数————————————————————————————————————————————————————————————————————
    def draw_detection_results(self,image_result:np.ndarray,product_info:dict):
        """调用通用绘制方法"""
        success, msg, result_image = draw_detection_results(image_result, product_info, mark_color="red")
        if msg:
            self._update_message_signal.emit(msg)
        return result_image

    #——————————————————————————————更新显示图像动画函数————————————————————————————————————————————————————————————————————
    def update_display_image(self,image:np.ndarray,live_on:bool):
        if not live_on:
            animation = self.bga_strip.get_full_animation()
        else:
            animation = None
        self._update_image_signal.emit(image,animation)

    #——————————————————————————————写入日志文件函数————————————————————————————————————————————————————————————————————
    def write_log_to_file(self, log_info: dict):
        """
        将日志信息写入Excel文件
        
        Args:
            log_info: get_log_info()返回的日志信息字典
        """
        if not log_info:
            self._update_message_signal.emit("警告: 日志信息为空，跳过写入")
            return
        
        try:
            # 创建日志目录
            log_dir = "Log"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 生成文件名
            end_time = datetime.now()
            timestamp = end_time.strftime("%Y%m%d_%H%M%S")
            side = self.bga_strip.strip_side if hasattr(self.bga_strip, 'strip_side') else "front"
            lot_id = self.bga_strip.strip_lot if hasattr(self.bga_strip, 'strip_lot') else "none"
            sn_id = self.bga_strip.strip_sn if hasattr(self.bga_strip, 'strip_sn') else "none"


            log_filename = f"{lot_id}_{sn_id}_dry_{side}_{timestamp}.xlsx"
            
            log_filepath = os.path.join(log_dir, log_filename)
            
            # 创建Excel工作簿
            wb = Workbook()
            ws = wb.active
            ws.title = "检测日志"
            
            # 定义样式
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=11)
            ng_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            ng_font = Font(bold=True)
            normal_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
            center_alignment = Alignment(horizontal="center", vertical="center")
            
            row_num = 1
            detection_type_map = {
                "Size": "尺寸检测不良数",
                "Area": "锡球面积检测不良数",
                "Ball Count": "锡球数量检测不良数",
                "Mark": "Mark检测不良数",
                "Scratch": "划痕检测不良数",
                "Shift": "偏移检测不良数"
            }
            detection_enable_map = {
                "Size": self.params.get('size_check_enable', True) if hasattr(self, 'params') else True,
                "Area": self.params.get('ball_check_enable', True) if hasattr(self, 'params') else True,
                "Ball Count": self.params.get('ball_check_enable', True) if hasattr(self, 'params') else True,
                "Mark": self.params.get('mark_check_enable', True) if hasattr(self, 'params') else True,
                "Scratch": self.params.get('scratch_check_enable', True) if hasattr(self, 'params') else True,
                "Shift": self.params.get('shift_check_enable', True) if hasattr(self, 'params') else True
            }
            
            # 写入检测流程信息
            ws.cell(row=row_num, column=1, value="检测流程信息").font = Font(bold=True, size=12)
            row_num += 1
            
            process_info = log_info.get("process_info", {})
            if process_info.get("start_time"):
                ws.cell(row=row_num, column=1, value="开始时间")
                ws.cell(row=row_num, column=2, value=process_info["start_time"])
                row_num += 1
            
            ws.cell(row=row_num, column=1, value="结束时间")
            ws.cell(row=row_num, column=2, value=process_info.get("end_time", ""))
            row_num += 1
            
            if process_info.get("duration_seconds") is not None:
                ws.cell(row=row_num, column=1, value="持续时间(秒)")
                cell = ws.cell(row=row_num, column=2, value=f"{process_info['duration_seconds']:.2f}")
                row_num += 1
            
            ws.cell(row=row_num, column=1, value="总产品数")
            ws.cell(row=row_num, column=2, value=process_info.get("total_products", 0))
            row_num += 1
            
            ws.cell(row=row_num, column=1, value="NG总数")
            cell = ws.cell(row=row_num, column=2, value=process_info.get("ng_total", 0))
            cell.fill = ng_fill
            cell.font = ng_font
            row_num += 1
            
            # 写入已启用检测项目
            enabled_detections = process_info.get("enabled_detections", [])
            ws.cell(row=row_num, column=1, value="已启用检测项目")
            ws.cell(row=row_num, column=2, value="、".join(enabled_detections) if enabled_detections else "无")
            row_num += 2
            
            # 写入统计信息
            ws.cell(row=row_num, column=1, value="统计信息").font = Font(bold=True, size=12)
            row_num += 1
            
            statistics = log_info.get("statistics", {})
            stats_info = []
            
            # 从product_list中提取宽度和高度值用于显示极值
            product_list = log_info.get("product_list", [])
            width_values = []
            height_values = []
            for product in product_list:
                width_str = product.get("width", "")
                height_str = product.get("height", "")
                if width_str:
                    try:
                        width_values.append(float(width_str))
                    except:
                        pass
                if height_str:
                    try:
                        height_values.append(float(height_str))
                    except:
                        pass
            
            if statistics.get("width_range") is not None and width_values:
                stats_info.append(("宽度极值(mm)", f"{statistics['width_range']:.4f} (最大:{max(width_values):.4f} - 最小:{min(width_values):.4f})"))
            
            if statistics.get("height_range") is not None and height_values:
                stats_info.append(("高度极值(mm)", f"{statistics['height_range']:.4f} (最大:{max(height_values):.4f} - 最小:{min(height_values):.4f})"))
            
            if statistics.get("avg_width") is not None and statistics.get("avg_height") is not None:
                stats_info.append(("平均尺寸(mm)", f"宽度:{statistics['avg_width']:.4f}, 高度:{statistics['avg_height']:.4f}"))
            
            if statistics.get("avg_ball_radius") is not None:
                stats_info.append(("平均球半径(mm)", f"{statistics['avg_ball_radius']:.4f}"))
            
            if statistics.get("shift_x_max") is not None and statistics.get("shift_x_min") is not None:
                stats_info.append(("偏移X极值(mm)", f"最大:{statistics['shift_x_max']:.4f}, 最小:{statistics['shift_x_min']:.4f}"))
            
            if statistics.get("shift_y_max") is not None and statistics.get("shift_y_min") is not None:
                stats_info.append(("偏移Y极值(mm)", f"最大:{statistics['shift_y_max']:.4f}, 最小:{statistics['shift_y_min']:.4f}"))
            
            if statistics.get("shift_x_cpk") is not None:
                stats_info.append(("偏移X CPK", f"{statistics['shift_x_cpk']:.4f}"))
            
            if statistics.get("shift_y_cpk") is not None:
                stats_info.append(("偏移Y CPK", f"{statistics['shift_y_cpk']:.4f}"))
            
            stats_info.extend([("", ""), ("各检测项目不良统计", "")])
            
            # 写入统计信息
            for label, value in stats_info:
                ws.cell(row=row_num, column=1, value=label)
                cell = ws.cell(row=row_num, column=2, value=value)
                if label == "各检测项目不良统计":
                    cell.font = Font(bold=True)
                elif label in detection_type_map.values() and isinstance(value, (int, float)) and value > 0:
                    cell.fill = ng_fill
                    cell.font = ng_font
                row_num += 1
            
            # 写入各检测项目不良统计
            defect_statistics = log_info.get("defect_statistics", {})
            for ng_type, count in defect_statistics.items():
                if detection_enable_map.get(ng_type, False):
                    label = detection_type_map.get(ng_type, ng_type)
                    ws.cell(row=row_num, column=1, value=label)
                    cell = ws.cell(row=row_num, column=2, value=count)
                    if count > 0:
                        cell.fill = ng_fill
                        cell.font = ng_font
                    row_num += 1
            
            row_num += 1
            
            # 写入产品详细数据表头
            headers = ["序号", "宽度(mm)", "高度(mm)", "有Mark", "NG球数", "NG球信息", "偏移量X(mm)", "偏移量Y(mm)", "是否NG", "NG类型"]
            for col_idx, header in enumerate(headers, start=1):
                cell = ws.cell(row=row_num, column=col_idx, value=header)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_alignment
            row_num += 1
            
            # 写入产品详细数据
            for product in product_list:
                is_ng_product = product.get("is_ng", "否") == "是"
                
                row_data = [
                    str(product.get("product_index", "")),
                    product.get("width", ""),
                    product.get("height", ""),
                    product.get("has_mark", "否"),
                    product.get("ng_ball_count", "0"),
                    product.get("ng_ball_info", ""),
                    product.get("shift_x", "0.0000"),
                    product.get("shift_y", "0.0000"),
                    product.get("is_ng", "否"),
                    product.get("ng_types", "")
                ]
                
                for col_idx, value in enumerate(row_data, start=1):
                    cell = ws.cell(row=row_num, column=col_idx, value=value)
                    cell.fill = ng_fill if is_ng_product else normal_fill
                    cell.font = ng_font if is_ng_product else None
                    cell.alignment = center_alignment
                row_num += 1
            
            # 自动调整列宽
            for col_idx, header in enumerate(headers, start=1):
                col_letter = get_column_letter(col_idx)
                max_length = max(len(str(header)), 
                               max(len(str(ws.cell(row=r, column=col_idx).value or "")) 
                                   for r in range(1, row_num)))
                ws.column_dimensions[col_letter].width = min(max_length + 6, 60)
            
            # 保存文件
            wb.save(log_filepath)
            print(f"日志文件已保存: {log_filepath}")
            print(f"共记录 {process_info.get('total_products', 0)} 个产品的检测信息，其中NG产品 {process_info.get('ng_total', 0)} 个")
            
        except Exception as e:
            print(f"写入日志文件错误: {str(e)}")
            traceback.print_exc()

    #——————————————————————————————写入Modbus寄存器函数————————————————————————————————————————————————————————————————
    def _write_modbus_registers(self,value_array, mode):
        """
        写入Modbus寄存器
        Args:
            value_array: 要写入的值数组
            mode: 写入模式
        Returns:
            True: 写入成功
            False: 写入失败
        """
        value_array = np.array(value_array)
        mask = ~np.isin(value_array, [0, 1, 2, 3])
        value_array[mask] = 3
        side = self.bga_strip.strip_side
        coil_address = 1 if side == "front" else 2
        register_start = 2 if side == "front" else 2002
        
        result = value_transmit(value_array, mode)
        result_list = [max(0, min(65535, int(val))) for val in result.tolist()]
        
        if not result_list:
            print(f"警告: {side} 数据为空，跳过写入")
            return
        
        MAX_REGISTERS = 123
        total_registers = len(result_list)
        
        if total_registers > MAX_REGISTERS:
            num_batches = (total_registers + MAX_REGISTERS - 1) // MAX_REGISTERS
            for batch_idx in range(num_batches):
                start_idx = batch_idx * MAX_REGISTERS
                end_idx = min(start_idx + MAX_REGISTERS, total_registers)
                batch_data = result_list[start_idx:end_idx]
                current_register_start = register_start + start_idx
                try:
                    self.MM.write(alias="dry_modbus",address = current_register_start,value_list=batch_data,function_code=cst.WRITE_MULTIPLE_REGISTERS)
                except Exception as e:
                    print(f"错误: {side} 批次 {batch_idx + 1} 写入失败: {str(e)}")
                    return False
                time.sleep(0.01)
            self.MM.write(alias="dry_modbus",address = coil_address,value_list=[1],function_code=cst.WRITE_SINGLE_COIL)
            return True
        else:
            self.MM.write(alias="dry_modbus",address = register_start,value_list=result_list,function_code=cst.WRITE_MULTIPLE_REGISTERS)
            self.MM.write(alias="dry_modbus",address = coil_address,value_list=[1],function_code=cst.WRITE_SINGLE_COIL)
            return True
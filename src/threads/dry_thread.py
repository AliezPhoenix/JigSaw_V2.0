# 从共享导入文件导入所有需要的模块
import logging
from ast import Continue
from winsound import SND_ALIAS
from src.threads.thread_imports import *
from src.support.operation_log import log_operation

class DryThread(QThread):
    _update_image_signal = pyqtSignal(np.ndarray, object)  # (图像, Bga_Strip|None)
    _update_statistics_signal = pyqtSignal(dict)  # 统计更新信号
    _update_message_signal = pyqtSignal(str)
    _strip_choice_prompt_signal = pyqtSignal(str)  # 工位标识，提示在分选机选择处理方案

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
        # 与 JIGSAW_Rebuild WorkThread：循环前 side='front'，Modbus 选址与 bga_obj 均依 current_side
        self.bga_strip = Bga_Strip(
            station="dry",
            strip_side="front",
            strip_lot="",
            strip_sn="",
            strip_create_time="",
            params=params,
        )
        self.current_side = "front"
        self.workflow_start_time = None
        # 模板缓存（必须在 update_params 之前初始化，因为 update_params 会访问 _template_path）
        self._template = None
        self._template_path = None
        self.update_params(params)
        
        # 图像异步保存
        self.image_save_queue = Queue()
        self._init_image_save_thread()
        
        # 垃圾回收计数器
        self.gc_counter = 0
        self.gc_interval = 10  # 每10次循环执行一次垃圾回收
        
        # 暂停/恢复标志
        self.is_paused = False
        
        # NG 监控：上次告警码，用于节流
        self._last_alarm_code = 0
        # Lot 级统计（主界面按 lot_id 汇总）
        self._lot_stats = {
            "lot_id": "",
            "total_count": 0,
            "ng_count": 0,
            "defect_counts": {"Mark": 0, "Size": 0, "Ball_Area": 0, "Ball Count": 0, "Scratch": 0, "Shift": 0}
        }
    
    #——————————————————————————————参数更新函数————————————————————————————————————————————————————————————————————
    def update_params(self,params:dict):
        self.params = params
        self.size_detect_params = {
            "min_threshold": self.params.get("min_threshold_size", 0),
            "max_threshold": self.params.get("max_threshold_size", 255),
            "allow_tolerance_x": self.params.get("product_size_tolerance_x", 0.1),
            "allow_tolerance_y": self.params.get("product_size_tolerance_y", 0.1),
            "roi_width": 80,
            "std_size": self.params.get("product_size", [10.0, 15.0]),
            "pixel_size": self.params.get("pixel_size", 0.008823),
            "pixel_size_x": self.params.get("pixel_size_x"),
        }
        self.ball_detect_params = {
            "min_threshold": self.params.get("min_threshold_ball", 0),
            "max_threshold": self.params.get("max_threshold_ball", 255),
            "ball_area_max_threshold": self.params.get("ball_area_max_threshold", 1500),
            "ball_area_min_threshold": self.params.get("ball_area_min_threshold", 500),
            "ball_radius_tolerance": self.params.get("ball_radius_tolerance", 0.05),
            "std_radius": self.params.get("std_radius", 0.17),
            "expected_ball_count": self.params.get("ball_count", 0),
            "ball_search_roi": self.params.get("ball_search_roi", []),
            "pixel_size": self.params.get("pixel_size", 0.008823)   
        }

        self.shift_detect_params = {
            "pixel_size": self.params.get("pixel_size", 0.008823),
            "error_correction_factor": 0.7,
            "allow_tolerance_x": self.params.get("shift_x_tolerance", 0.05),
            "allow_tolerance_y": self.params.get("shift_y_tolerance", 0.05),
        }

        self.mark_detect_params = {
            "min_threshold": self.params.get("min_threshold_mark", 0),
            "max_threshold": self.params.get("max_threshold_mark", 255),
            "auto_threshold_factor": 1.05,
            "pixel_size": self.params.get("pixel_size", 0.008823),
            "mark_detect_mode": self.params.get("mark_detect_mode", "manual"),
            "mark_rois": self.params.get("mark_rois", []),
            "mark_roi_min_areas": self.params.get("mark_roi_min_areas", []),
            "allow_mark": self.params.get("allow_mark", False),
        }

        self.template_detect_params = {
            "template_threshold": self.params.get("template_threshold", 0.7),
            "search_roi": self.params.get("search_roi", [])
        }

        self.scratch_detect_params = {
            "min_threshold": self.params.get("min_threshold_scratch", 0),
            "max_threshold": self.params.get("max_threshold_scratch", 255),
            "scratch_length_threshold": self.params.get("scratch_length", 5.0),
            "pixel_size": self.params.get("pixel_size", 0.008823),
            "scratch_roi": self.params.get("scratch_roi", []),
            "roi_blocks": self.params.get("roi_block", [])
        }
        
        self.ball_detector.update_params(self.ball_detect_params)
        self.size_detector.update_params(self.size_detect_params)
        self.shift_detector.update_params(self.shift_detect_params)
        self.mark_detector.update_params(self.mark_detect_params)
        self.scratch_detector.update_params(self.scratch_detect_params)
        self.template_detector.update_params(self.template_detect_params)
        # 模板路径变化时重新加载
        self._template_path = self.params.get("golden_template_path")
        self._template = cv.imread(self._template_path) if self._template_path else None

    def _get_template(self):
        """获取模板图像（使用缓存，路径变化时在 update_params 中已更新）"""
        return self._template

    #——————————————————————————————图像异步保存函数————————————————————————————————————————————————————————————————————
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
        lot_id = sanitize_filename_part(self.bga_strip.strip_lot if hasattr(self.bga_strip, 'strip_lot') else "LOT")
        sn_id = sanitize_filename_part(self.bga_strip.strip_sn if hasattr(self.bga_strip, 'strip_sn') else "SN")
        side = self.current_side
        
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

    def _dry_plc_allow_handshake(self) -> bool:
        """本拍是否允许写完成线圈（及 strip 完成时的日志）。需 PLC 二选一时阻塞至 continue，否则为 False。"""
        print(f"DRY:alarm_sent_current: {self.alarm_sent_current}")

        if not self.alarm_sent_current:
            return True
        self._strip_choice_prompt_signal.emit("dry")
        choice = wait_for_strip_plc_choice(
            self.MM,
            "dry_modbus",
            lambda: self.isInterruptionRequested(),
        )
        if choice == "timeout":
            self._strip_choice_prompt_signal.emit("timeout")
            return False
        
        return choice == "continue"

    def _dry_finish_strip_log_and_coil(self, mode,ng_sectors):
        """strip 完成：写结果寄存器与日志然后发送完成。"""
        
        log_info = self.bga_strip.get_log_info()
        send_data = self.bga_strip.full_value.copy()
        self._write_modbus_registers(send_data, mode, side=self.current_side,ng_sectors=ng_sectors)
        self.write_log_to_file(log_info)
        time.sleep(0.05)
        self.MM.write(alias="dry_modbus", address=0, value_list=[1], function_code=cst.WRITE_SINGLE_COIL)
        log_operation(
            "DryThread",
            "strip 完成 Modbus 握手线圈",
            level=logging.INFO,
            modbus_alias="dry_modbus",
            function_code="WRITE_SINGLE_COIL",
            address=0,
            values="[1]",
            strip_side=self.current_side,
        )
    
    def _params_summary_for_log(self) -> str:
        p = self.params or {}
        parts = []
        for k in ("product_count", "total_rows", "total_cols", "product_type", "template_threshold", "pixel_size"):
            if k in p:
                parts.append(f"{k}={p[k]}")
        return ", ".join(parts) if parts else "—"

    #——————————————————————————————主循环函数————————————————————————————————————————————————————————————————————
    def run(self):
        trigger_camera_last = 1
        trigger_finished_last = 0
        trigger_count_last = 0
        self.current_side = "front"
        debug = False
        _main_loop_logged = False
        while not self.isInterruptionRequested():
            if not _main_loop_logged:
                log_operation(
                    "DryThread",
                    "主循环开始",
                    level=logging.INFO,
                    params_summary=self._params_summary_for_log(),
                )
                _main_loop_logged = True
            # 检查暂停状态
            if self.is_paused:
                time.sleep(0.1)
                continue

            #——————————————————读取ModBus数据————————————————————————————————————
            _,discrete_input_list = self.MM.read("dry_modbus",address=0,count=4,function_code=cst.READ_DISCRETE_INPUTS)
            _,input_register_list = self.MM.read("dry_modbus",address=2,count=3,function_code=cst.READ_INPUT_REGISTERS)
            
            # 边界检查：ModBus数据完整性
            if not self._check_modbus_data(discrete_input_list, input_register_list):
                self._update_message_signal.emit("ModBus数据完整性验证失败")
                time.sleep(0.01)
                continue
            
            trigger_camera, trigger_front, trigger_back, trigger_finished = discrete_input_list
            mode, trigger_count, sector_change_flag = input_register_list
            NG_sector_1,NG_sector_2,NG_sector_3 = self.MM.read("dry_modbus",address=13,count=3,function_code=cst.READ_INPUT_REGISTERS)[1]
            ng_sectors = {
                "sector_1": int(NG_sector_1),
                "sector_2": int(NG_sector_2),
                "sector_3": int(NG_sector_3)
            }
            #——————————————————————根据正反初始化 bga_strip（对齐 JIGSAW_Rebuild Threads.py WorkThread）————————————————————————————————————
            if trigger_count == 1 and trigger_count_last != trigger_count:
                self.workflow_start_time = datetime.now()
                lot = hex_to_string(self.MM.read("dry_modbus",address=20,count=30,function_code=cst.READ_INPUT_REGISTERS)[1])
                sn = hex_to_string(self.MM.read("dry_modbus",address=55,count=30,function_code=cst.READ_INPUT_REGISTERS)[1])
                print(f"DRY:lot:{lot},sn:{sn}")
                # lot_id 变化时重置 lot 级统计与告警状态
                if lot != self._lot_stats.get("lot_id", ""):
                    self._lot_stats = {
                        "lot_id": lot,
                        "total_count": 0,
                        "ng_count": 0,
                        "defect_counts": {"Mark": 0, "Size": 0, "Ball_Area": 0, "Ball Count": 0, "Scratch": 0, "Shift": 0}
                    }
                    self._last_alarm_code = 0
                # 与 Rebuild：仅 trigger_front XOR trigger_back 时新建对应 BGA 并更新 side；否则保持 bga_strip 与 current_side
                if trigger_front and not trigger_back:
                    self.bga_strip = Bga_Strip(
                        station="dry",
                        strip_side="front",
                        strip_lot=lot,
                        strip_sn=sn,
                        strip_create_time=datetime.now().strftime("%Y%m%d%H%M%S"),
                        params=self.params,
                    )
                    self.current_side = "front"
                elif trigger_back and not trigger_front:
                    self.bga_strip = Bga_Strip(
                        station="dry",
                        strip_side="back",
                        strip_lot=lot,
                        strip_sn=sn,
                        strip_create_time=datetime.now().strftime("%Y%m%d%H%M%S"),
                        params=self.params,
                    )
                    self.current_side = "back"
                elif not trigger_front and not trigger_back:
                    trigger_count_last = 0  ##强制触发下次再读一遍
                #报警复位
                self.MM.write(alias="dry_modbus", address=1, value_list=[0], function_code=cst.WRITE_SINGLE_REGISTER)
                alarm_code = 0
                self._last_alarm_code = alarm_code
                log_operation(
                    "DryThread",
                    "干燥台 strip 单面拍照开始",
                    level=logging.INFO,
                    lot_id=lot or "未设置",
                    sn=sn or "未设置",
                    strip_side=self.current_side,
                    params_summary=self._params_summary_for_log(),
                )
            #——————————————————————检测触发————————————————————————————————
            if (trigger_camera and not trigger_camera_last) or (trigger_finished and not trigger_finished_last):
                self.alarm_sent_current = False
                _shot_src = "camera" if (trigger_camera and not trigger_camera_last) else "finished_edge"
                shot_centers = []
                log_operation(
                    "DryThread",
                    "单次拍照触发",
                    level=logging.INFO,
                    trigger_source=_shot_src,
                    strip_side=self.current_side,
                    lot_id=getattr(self.bga_strip, "strip_lot", "") or "未设置",
                    sn=getattr(self.bga_strip, "strip_sn", "") or "未设置",
                )
                #——————————————————图像采集——————————————————————————————
                ret,msg,image = self.HM.capture_image("dry_cam")
                if not ret or debug:
                    self._update_message_signal.emit(f"采集图像失败: {msg}")
                    time.sleep(0.01)
                    continue
                image = cv.rotate(image,cv.ROTATE_90_CLOCKWISE) ####干燥台相机旋转90度
                image_result = ensure_bgr_u8(image, copy=True)

                grid_r = int(self.bga_strip.window_rows or 0)
                grid_c = int(self.bga_strip.window_cols or 0)
                if grid_r <= 0 or grid_c <= 0:
                    grid_r, grid_c = 1, 1
                img_h, img_w = image.shape[:2]
                search_roi = self.params.get("search_roi") or []
                grid_roi = normalize_grid_search_roi(search_roi, img_w, img_h, grid_r, grid_c)
                if grid_roi is None:
                    self._update_message_signal.emit("警告: dry search_roi 非网格结构，请重新创建 Search ROI")
                    time.sleep(0.01)
                    continue
                slot = [[None] * grid_c for _ in range(grid_r)]
                for gr in range(grid_r):
                    for gcol in range(grid_c):
                        x, y, cell_w, cell_h = grid_roi[gr][gcol]
                        if not self._check_image_bounds(
                            image, x, y, cell_w, cell_h
                        ):
                            self._update_message_signal.emit(
                                f"警告: cell ROI ({x}, {y}, {cell_w}, {cell_h}) 超出图像边界 ({img_w}, {img_h})"
                            )
                            time.sleep(0.01)
                            continue
                        product_image = image[
                            y : y + cell_h, x : x + cell_w
                        ]
                        success, msg, product_info = self._detect_product(
                            x, y, product_image
                        )
                        if not success:
                            filepath_result = self._generate_image_filename("ERROR")
                            filepath_ori = self._generate_image_filename("ORI")
                            self._async_save_image(
                                product_info["product_image_result"], filepath_result
                            )
                            self._async_save_image(product_image, filepath_ori)
                        if "OK" not in product_info["defect_type"]:
                            defect_type = (
                                product_info["defect_type"][0]
                                if len(product_info["defect_type"]) > 1
                                else "UNKNOWN"
                            )
                            filepath_result = self._generate_image_filename(defect_type)
                            filepath_ori = self._generate_image_filename("ORI")
                            self._async_save_image(product_info["product_image_result"], filepath_result)
                            self._async_save_image(product_image, filepath_ori)
                        slot[gr][gcol] = product_info
                        shot_centers.append((x + cell_w // 2, y + cell_h // 2))
                        image_result[y : y + cell_h, x : x + cell_w] = product_info["product_image_result"]
                        image_result = cv.rectangle(image_result, (x, y), (x + cell_w, y + cell_h), (0, 255, 255), 2)
                #——————————————————更新显示和统计信息——————————————————————————————————————————————————————

                self._update_image_signal.emit(image_result, self.bga_strip)
                self.bga_strip.write(slot, image)
                stats_info = self.bga_strip.get_statistics_info()
                dc = (stats_info or {}).get("defect_counts") or {}
                ng_s = ",".join(f"{k}:{v}" for k, v in sorted(dc.items()) if v) or "无"
                ctr_s = ";".join(f"({cx},{cy})" for cx, cy in shot_centers) or "—"
                log_operation(
                    "DryThread",
                    "单次拍照结束",
                    level=logging.INFO,
                    template_match_centers=ctr_s,
                    product_count=str((stats_info or {}).get("total_count", 0)),
                    ng_types=ng_s,
                    strip_side=self.current_side,
                )


                #———————————————————警告相关流程————————————————————————————————————————————————————————————————————
                ng_monitor = self.params.get("ng_monitor", {})
                alarm_code = check_ng_alarm(stats_info, ng_monitor) if stats_info else 0

                if alarm_code != self._last_alarm_code and self.alarm_sent_current == False:
                    try:
                        ok, err = self.MM.write(alias="dry_modbus", address=1, value_list=[alarm_code], function_code=cst.WRITE_SINGLE_REGISTER)
                        if ok:
                            self._last_alarm_code = alarm_code
                            # 仅本周期写入非零告警码时为 True；写 0 复位或 final 与累计告警无关
                            self.alarm_sent_current = alarm_code != 0
                        else:
                            print(f"NG监控 Modbus 写入失败: {err}")
                    except Exception as e:
                        print(f"NG监控 Modbus 写入异常: {e}")
                
                
                #———————————————————strip完成相关流程————————————————————————————————————————————————————————————————————
                strip_done_modbus = (
                    trigger_finished == 1
                    and trigger_finished_last == 0
                    and (trigger_front == 1 or trigger_back == 1)
                )
                allow_handshake = self._dry_plc_allow_handshake()
                if strip_done_modbus:
                    if stats_info:
                        self._lot_stats["total_count"] += stats_info.get("total_count", 0)
                        self._lot_stats["ng_count"] += stats_info.get("ng_count", 0)
                        for k, v in (stats_info.get("defect_counts") or {}).items():
                            self._lot_stats["defect_counts"][k] = self._lot_stats["defect_counts"].get(k, 0) + v

                    if allow_handshake:
                        self._dry_finish_strip_log_and_coil(mode,ng_sectors)

                    #——————————————————————警告复位————————————————————————————
                    final_alarm = check_ng_alarm(stats_info, ng_monitor) if stats_info else 0
                    if final_alarm == 0 and self._last_alarm_code != 0:
                        try:
                            self.MM.write(alias="dry_modbus", address=1, value_list=[0], function_code=cst.WRITE_SINGLE_REGISTER)
                            self._last_alarm_code = 0
                        except Exception as e:
                            print(f"NG监控 复位 Modbus 写入异常: {e}")

                    # strip 已完成：只发一次 lot 级完工统计（避免与下方「进行中」合并重复双计）
                    lot_emit = self._build_lot_stats_for_emit(None, is_strip_finished=True)
                    self._update_statistics_signal.emit(lot_emit)

                elif allow_handshake:
                    self.MM.write(alias="dry_modbus", address=0, value_list=[1], function_code=cst.WRITE_SINGLE_COIL)

                #———————————————————统计信息写入——————————————————————————
                if stats_info and not strip_done_modbus:
                    merged = self._build_lot_stats_for_emit(stats_info, is_strip_finished=False)
                    self._update_statistics_signal.emit(merged)
                
            elif trigger_camera ==0 and trigger_finished == 0:
                self.MM.write(alias="dry_modbus",address = 0,value_list=[0],function_code=cst.WRITE_SINGLE_COIL)
            else:
                self.MM.write(alias="dry_modbus",address = 1,value_list=[0],function_code=cst.WRITE_SINGLE_COIL)
                self.MM.write(alias="dry_modbus",address = 2,value_list=[0],function_code=cst.WRITE_SINGLE_COIL)
            trigger_camera_last = trigger_camera
            trigger_finished_last = trigger_finished
            trigger_count_last = trigger_count
            #————————————————————————实时显示画面————————————————————
            if self.ui.radioButton_live_dry.isChecked():
                #——————————————————只有在没有处理触发信号时才更新实时显示，避免重复采集图像————————————————————————————————————
                if not ((trigger_camera and not trigger_camera_last) or (trigger_finished and not trigger_finished_last)):
                    #——————————————————采集图像————————————————————
                    ret,msg, image_live = self.HM.capture_image("dry_cam")
                    if not ret:
                        self._update_message_signal.emit(f"采集图像失败: {msg}")
                        continue
                    h, w = image_live.shape[:2]
                    
                    image_live_bgr = ensure_bgr_u8(image_live, copy=True)
                    image_live_bgr = cv.rotate(image_live_bgr, cv.ROTATE_90_CLOCKWISE)
                    cv.line(image_live_bgr, (0, h // 2), (w, h // 2), (0, 255, 0), 2)
                    cv.line(image_live_bgr, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)
                    self._update_image_signal.emit(image_live_bgr,None)
            else:
                time.sleep(0.01)
            
            #———————————————————垃圾回收————————————————————————————
            self.gc_counter += 1
            if self.gc_counter >= self.gc_interval:
                gc.collect()
                self.gc_counter = 0

        log_operation("DryThread", "主循环退出", level=logging.INFO)
    
    def _build_lot_stats_for_emit(self, strip_stats: dict, is_strip_finished: bool) -> dict:
        """构建主界面所需的 lot 级统计（与 get_statistics_info 格式一致）"""
        total = self._lot_stats["total_count"]
        ng = self._lot_stats["ng_count"]
        dc = dict(self._lot_stats["defect_counts"])
        if not is_strip_finished and strip_stats:
            total += strip_stats.get("total_count", 0)
            ng += strip_stats.get("ng_count", 0)
            for k, v in (strip_stats.get("defect_counts") or {}).items():
                dc[k] = dc.get(k, 0) + v
        yield_rate = ((total - ng) / total * 100) if total > 0 else 0.0
        return {
            "station": "干燥台",
            "lot_id": self._lot_stats.get("lot_id", "") or (strip_stats.get("lot_id", "") if strip_stats else ""),
            "total_count": total,
            "ng_count": ng,
            "yield_rate": yield_rate,
            "defect_counts": dc
        }
    
    #——————————————————————————————检测产品函数————————————————————————————————————————————————————————————————————
    def _detect_product(self,x,y,product_image:np.ndarray):
        # 准备检测器字典
        # 调用通用检测函数（提前返回模式）
        success, msg, product_info = execute_product_detection(
            image=product_image,
            detectors={
            "ball_detector": self.ball_detector,
            "size_detector": self.size_detector,
            "mark_detector": self.mark_detector,
            "shift_detector": self.shift_detector,
            "scratch_detector": self.scratch_detector,
                },
            params={
            "mark_check_enable": self.params.get("mark_check_enable", True),
            "size_check_enable": self.params.get("size_check_enable", True),
            "ball_check_enable": self.params.get("ball_check_enable", True),
            "shift_check_enable": self.params.get("shift_check_enable", True),
            "scratch_check_enable": self.params.get("scratch_check_enable", True),
            "allow_mark": False,
            "roi_block": self.params.get("roi_block", []),
            },
            detect_type=None,  # 执行所有启用的检测
            early_return_on_ng=True,  # 生产模式：检测到NG立即返回
            error_callback=None  # 不显示对话框，返回错误信息
        )
        
        # 添加位置信息
        product_info["x"] = x
        product_info["y"] = y
        
        # 如果检测失败，直接返回
        if not success:
            return False, msg, product_info
        
        # 绘制检测结果
        product_info["product_image_result"] = self.draw_detection_results(product_image, product_info)
        
        return True, "成功", product_info

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
            side = self.current_side
            lot_id = sanitize_filename_part(self.bga_strip.strip_lot if hasattr(self.bga_strip, 'strip_lot') else "none")
            sn_id = sanitize_filename_part(self.bga_strip.strip_sn if hasattr(self.bga_strip, 'strip_sn') else "none")


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
                "Ball_Area": "锡球面积检测不良数",
                "Ball Count": "锡球数量检测不良数",
                "Mark": "Mark检测不良数",
                "Scratch": "划痕检测不良数",
                "Shift": "偏移检测不良数"
            }
            detection_enable_map = {
                "Size": self.params.get('size_check_enable', True) if hasattr(self, 'params') else True,
                "Ball_Area": self.params.get('ball_check_enable', True) if hasattr(self, 'params') else True,
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
    def _write_modbus_registers(self, value_array, mode, side=None,ng_sectors=None):
        """
        写入Modbus寄存器（与 Rebuild _write_modbus_registers(side, value_array, mode) 一致，side 决定线圈/寄存器基址）
        Args:
            value_array: 要写入的值数组
            mode: 写入模式
            side: "front" | "back"，默认 self.current_side
        Returns:
            True: 写入成功
            False: 写入失败
        """
        value_array = np.array(value_array)
        value_array = map_product_type_to_sector(value_array, ng_sectors)
        # 将value_array中的所有值为99的元素，改为3
        value_array = np.where(value_array == 99, 3, value_array)
        side = self.current_side if side is None else side
        coil_address = 1 if side == "front" else 2
        register_start = 2 if side == "front" else 2002
        
        result = value_transmit(value_array, mode)
        result_list = [max(0, min(65535, int(val))) for val in result.tolist()]
        print(f"DRY:result_list: {result_list}")
        if not result_list:
            print(f"警告: {side} 数据为空，跳过写入")
            return False

        # 分批由 ModBus_Manager.write 统一处理（MAX_REGISTERS=123），避免与 hardware 重复判断
        ok, err = self.MM.write(
            alias="dry_modbus",
            address=register_start,
            value_list=result_list,
            function_code=cst.WRITE_MULTIPLE_REGISTERS,
        )
        if not ok:
            print(f"错误: {side} 写入保持寄存器失败: {err}")
            return False
        ok_c, err_c = self.MM.write(
            alias="dry_modbus",
            address=coil_address,
            value_list=[1],
            function_code=cst.WRITE_SINGLE_COIL,
        )
        if not ok_c:
            print(f"错误: {side} 写入完成线圈失败: {err_c}")
            return False
        log_operation(
            "DryThread",
            "strip 完成 Modbus 结果寄存器与完成线圈",
            level=logging.INFO,
            modbus_alias="dry_modbus",
            strip_side=side,
            register_address=register_start,
            register_values=str(result_list),
            done_coil_address=coil_address,
            done_coil_values="[1]",
        )
        return True
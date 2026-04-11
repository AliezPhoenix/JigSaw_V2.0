# 从共享导入文件导入所有需要的模块
import logging
import os
from src.threads.thread_imports import *
import torch
from src.support.operation_log import log_operation
from src.support.support_funs import (
    fulltray_load_model,
    fulltray_predict_single_image,
    hex_to_string,
    ensure_gray_u8,
    ensure_bgr_u8,
)


class FulltrayThread(QThread):
    _update_image_signal = pyqtSignal(object)  # (图像, Bga_Strip|None)，由 _update_display 统一处理显示
    _update_result_signal = pyqtSignal(bool, int, int, int, float)  # (is_ok, product_count, total_cells, empty_count, avg_confidence)
    _update_config_changed_signal = pyqtSignal(str)
    # ———————————————————————————————初始化————————————————————————————————————————————————————————————————
    def __init__(self, params: dict, HM: Hardware_Manager, MM: ModBus_Manager, ui: 'MainWindow'):
        super().__init__()
        self.HM = HM    # 硬件管理器
        self.MM = MM    # Modbus管理器
        self.ui = ui
        self.params = params

        # 深度学习模型相关
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = None

        # 图像异步保存
        self.image_save_queue = Queue()
        self._init_image_save_thread()

        # 垃圾回收计数器
        self.gc_counter = 0
        self.gc_interval = 10

        # 暂停/恢复标志
        self.is_paused = False

        # 初始化模型
        self._load_model_if_needed()

    # ——————————————————————————————参数更新函数————————————————————————————————————————————————————————————————————
    def update_params(self, params: dict):
        self.params = params
        if self.params.get("model_path") != self.model_path:
            self._load_model_if_needed()

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
                    print(f"满盘异步保存图像错误: {str(e)}")
                    try:
                        self.image_save_queue.task_done()
                    except Exception:
                        pass

        save_thread = threading.Thread(target=_save_images_worker, daemon=True)
        save_thread.start()

    def _async_save_image(self, image: np.ndarray, filepath: str):
        """异步保存图像"""
        try:
            self.image_save_queue.put((image.copy(), filepath), block=False)
        except Exception as e:
            print(f"添加满盘图像到保存队列失败: {str(e)}")

    def _generate_image_filename(self, image_type: str, defect_type: str = "NONE", save_dir: str = None) -> str:
        """生成满盘图像文件名"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"FullTray_{image_type}_{defect_type}_{timestamp}.bmp"
        if save_dir is None:
            save_dir = "Image/Full_Tray/NG" if image_type == "NG" else "Image/Full_Tray/OK"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, filename)

    def _load_model_if_needed(self):
        """如果需要，加载模型"""
        model_path = self.params.get("model_path", None)
        if model_path and os.path.exists(model_path):
            try:
                self.model = fulltray_load_model(model_path, device=self.device)
                self.model_path = model_path
                print(f"满盘检测模型加载成功: {model_path}")
            except Exception as e:
                print(f"满盘检测模型加载失败: {str(e)}")
                self.model = None
                self.model_path = None
        else:
            self.model = None
            self.model_path = None

    def reload_model(self):
        """重新加载模型"""
        model_path = self.params.get("model_path", None)
        if model_path and os.path.exists(model_path):
            try:
                self.model = fulltray_load_model(model_path, device=self.device)
                self.model_path = model_path
                print(f"满盘检测模型重新加载成功: {model_path}")
                return True
            except Exception as e:
                print(f"满盘检测模型重新加载失败: {str(e)}")
                self.model = None
                self.model_path = None
                return False
        else:
            print(f"满盘模型路径无效或不存在: {model_path}")
            self.model = None
            self.model_path = None
            return False

    def _detect_fulltray_dl(self, image: np.ndarray, rows: int, cols: int, grid_roi=None):
        """
        使用深度学习模型进行满盘检测

        Args:
            image: 输入图像
            rows: 行数
            cols: 列数
            grid_roi: ROI区域 [x, y, w, h]，如果为None则使用整张图像

        Returns:
            tuple: (result_matrix, result_image, confidence_matrix)
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")

        if grid_roi is not None and len(grid_roi) >= 4:
            x, y, w, h = int(grid_roi[0]), int(grid_roi[1]), int(grid_roi[2]), int(grid_roi[3])
            roi_image = image[y:y+h, x:x+w]
        else:
            roi_image = image.copy()

        roi_gray = ensure_gray_u8(roi_image, copy=True)
        result_image = ensure_bgr_u8(roi_gray, copy=True)
        roi_h, roi_w = roi_gray.shape
        cell_h = roi_h // rows
        cell_w = roi_w // cols

        result_matrix = np.zeros((rows, cols), dtype=bool)
        confidence_matrix = np.zeros((rows, cols), dtype=float)
        input_size = self.params.get("input_size", 150)

        for i in range(rows):
            for j in range(cols):
                y1 = i * cell_h
                x1 = j * cell_w
                y2 = (i + 1) * cell_h if i < rows - 1 else roi_h
                x2 = (j + 1) * cell_w if j < cols - 1 else roi_w
                cell = roi_gray[y1:y2, x1:x2]

                try:
                    prediction, confidence = fulltray_predict_single_image(
                        self.model, cell, device=self.device, input_size=input_size
                    )
                    result_matrix[i, j] = (prediction == 1)
                    confidence_matrix[i, j] = confidence

                    cell_center_x = (x1 + x2) // 2
                    cell_center_y = (y1 + y2) // 2
                    cell_radius = min(cell_w, cell_h) // 4

                    if prediction == 1:
                        cv.circle(result_image, (cell_center_x, cell_center_y), cell_radius, (0, 255, 0), 2)
                    else:
                        offset = cell_radius
                        cv.line(result_image,
                               (cell_center_x - offset, cell_center_y - offset),
                               (cell_center_x + offset, cell_center_y + offset),
                               (0, 0, 255), 2)
                        cv.line(result_image,
                               (cell_center_x - offset, cell_center_y + offset),
                               (cell_center_x + offset, cell_center_y - offset),
                               (0, 0, 255), 2)
                except Exception as e:
                    print(f"满盘Cell[{i},{j}] 预测失败: {e}")
                    cell_center_x = (x1 + x2) // 2
                    cell_center_y = (y1 + y2) // 2
                    cell_radius = min(cell_w, cell_h) // 4
                    offset = cell_radius
                    cv.line(result_image,
                           (cell_center_x - offset, cell_center_y - offset),
                           (cell_center_x + offset, cell_center_y + offset),
                           (0, 255, 255), 2)
                    cv.line(result_image,
                           (cell_center_x - offset, cell_center_y + offset),
                           (cell_center_x + offset, cell_center_y - offset),
                           (0, 255, 255), 2)
                    result_matrix[i, j] = False

        if grid_roi is not None and len(grid_roi) >= 4:
            x, y, w, h = int(grid_roi[0]), int(grid_roi[1]), int(grid_roi[2]), int(grid_roi[3])
            image_result_full = ensure_bgr_u8(image, copy=True)
            image_result_full[y:y+h, x:x+w] = result_image
            result_image = image_result_full

        return result_matrix, result_image, confidence_matrix

    def _update_ui_display(self, image_result: np.ndarray):
        """发射 ndarray 供 _update_display 统一处理显示"""
        if image_result is None:
            return
        try:
            image_result = ensure_bgr_u8(image_result, copy=True)
            self._update_image_signal.emit(image_result.copy())
        except Exception as e:
            print(f"满盘更新UI显示错误: {str(e)}")

    def pause(self):
        """暂停线程执行"""
        self.is_paused = True

    def resume(self):
        """恢复线程执行"""
        self.is_paused = False

    def run(self):
        print("FulltrayThread start")
        trigger_camera = 0
        trigger_camera_last = 0
        config_changed_flag_last = 0
        _main_loop_logged = False
        while not self.isInterruptionRequested():
            if not _main_loop_logged:
                log_operation(
                    "FulltrayThread",
                    "主循环开始",
                    level=logging.INFO,
                    rows=str(self.params.get("rows", "")),
                    cols=str(self.params.get("cols", "")),
                    model=os.path.basename(str(self.params.get("model_path", "") or "") or "未设置"),
                )
                _main_loop_logged = True
            if self.is_paused:
                time.sleep(0.1)
                continue

            try:
                success, discrete_list = self.MM.read("fulltray_modbus", address=0, count=3, function_code=cst.READ_DISCRETE_INPUTS)
                if success and discrete_list is not None and len(discrete_list) >= 3:
                    trigger_camera = discrete_list[0]
                    empty_flag = discrete_list[1]
                    config_changed_flag = discrete_list[2]
                else:
                    time.sleep(0.01)
                    continue
            except Exception as e:
                print(f"满盘读取Modbus寄存器错误: {str(e)}")
                time.sleep(0.01)
                continue

            if trigger_camera == 1 and trigger_camera_last == 0:
                try:
                    ret, msg, image = self.HM.capture_image("fulltray_cam")
                    if not ret or image is None:
                        print("满盘检测：获取图像失败")
                        trigger_camera_last = trigger_camera
                        continue

                    if self.params.get("model_path") != self.model_path:
                        self.reload_model()

                    if self.model is None:
                        print("满盘检测：模型未加载，跳过检测")
                        trigger_camera_last = trigger_camera
                        continue

                    log_operation(
                        "FulltrayThread",
                        "满盘检测触发",
                        level=logging.INFO,
                        rows=str(self.params.get("rows", 8)),
                        cols=str(self.params.get("cols", 16)),
                        model=os.path.basename(str(self.params.get("model_path", "") or "") or "未设置"),
                    )

                    search_roi = self.params.get("search_roi", [])
                    search_roi = search_roi if isinstance(search_roi, list) and len(search_roi) >= 4 else None

                    result_matrix, result_image, confidence_matrix = self._detect_fulltray_dl(
                        image,
                        self.params.get("rows", 8),
                        self.params.get("cols", 16),
                        search_roi
                    )

                    is_ok = np.all(result_matrix)
                    product_count = int(np.sum(result_matrix))
                    total_cells = self.params.get("rows", 8) * self.params.get("cols", 16)
                    empty_count = total_cells - product_count
                    avg_confidence = float(np.mean(confidence_matrix))

                    print(f"满盘检测: {'OK' if is_ok else 'NG'}, 有产品={product_count}/{total_cells}, 置信度={avg_confidence:.2%}")

                    if not is_ok:
                        filepath_ori = self._generate_image_filename("ORI", "NONE", save_dir="Image/Full_Tray/NG")
                        filepath_result = self._generate_image_filename("NG", "RESULT", save_dir="Image/Full_Tray/NG")
                        self._async_save_image(image, filepath_ori)
                        self._async_save_image(result_image, filepath_result)

                    self._update_ui_display(result_image)
                    self._update_result_signal.emit(is_ok, product_count, total_cells, empty_count, avg_confidence)
                    if hasattr(self.ui, 'work_fulltray_current_image'):
                        self.ui.work_fulltray_current_image = result_image.copy()

                    if is_ok:
                        self.MM.write(alias="fulltray_modbus", address=2, value_list=[2], function_code=cst.WRITE_SINGLE_REGISTER)
                    else:
                        self.MM.write(alias="fulltray_modbus", address=2, value_list=[1], function_code=cst.WRITE_SINGLE_REGISTER)

                    self.MM.write(alias="fulltray_modbus", address=0, value_list=[1], function_code=cst.WRITE_SINGLE_COIL)

                except Exception as e:
                    print(f"满盘检测错误: {str(e)}")
                    traceback.print_exc()

                trigger_camera_last = trigger_camera
            else:
                if trigger_camera == 0 and trigger_camera_last == 0:
                    self.MM.write(alias="fulltray_modbus", address=0, value_list=[0], function_code=cst.WRITE_SINGLE_COIL)

                if hasattr(self.ui, 'radioButton_fulltray_live') and self.ui.radioButton_fulltray_live.isChecked():
                    try:
                        ret, msg, live_image = self.HM.capture_image("fulltray_cam")
                        if ret and live_image is not None:
                            self._update_ui_display(live_image)
                    except Exception as e:
                        print(f"满盘实时画面显示错误: {e}")
                else:
                    time.sleep(0.01)

            if config_changed_flag == 1 and config_changed_flag_last == 0:
                config_name = self.MM.read(alias="fulltray_modbus", address=7, count=10, function_code=cst.READ_HOLDING_REGISTERS)
                config_name = hex_to_string(config_name)
                self._update_config_changed_signal.emit(config_name)

            
            trigger_camera_last = trigger_camera
            config_changed_flag_last = config_changed_flag
            self.gc_counter += 1
            if self.gc_counter >= self.gc_interval:
                gc.collect()
                self.gc_counter = 0

            time.sleep(0.01)

        log_operation("FulltrayThread", "主循环退出", level=logging.INFO)

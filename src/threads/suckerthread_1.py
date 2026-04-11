# 从共享导入文件导入所有需要的模块
import logging
from src.threads.thread_imports import *
from src.detectors.sucker_detector import SuckerDetector
import modbus_tk.defines as cst
from src.support.operation_log import log_operation


def _draw_result_on_image(image, result_dict):
    """在图像上绘制检测结果（矩形和十字线）"""
    if image is None:
        return None
    img = ensure_bgr_u8(image, copy=True)
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 绘制中心十字线
    cv.line(img, (center_x - 50, center_y), (center_x + 50, center_y), (0, 255, 0), 2)
    cv.line(img, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 0), 2)

    # 绘制吸嘴边界框
    sucker_box = result_dict.get("sucker_box_point")
    if sucker_box is not None:
        x, y, bw, bh = sucker_box
        cv.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
        cx, cy = x + bw // 2, y + bh // 2
        cv.line(img, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 1)
        cv.line(img, (cx, cy - 30), (cx, cy + 30), (0, 255, 255), 1)

    # 绘制产品边界框
    product_box = result_dict.get("product_box_point")
    if product_box is not None:
        x, y, bw, bh = product_box
        cv.rectangle(img, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
        cx, cy = x + bw // 2, y + bh // 2
        cv.line(img, (cx - 30, cy), (cx + 30, cy), (255, 0, 255), 1)
        cv.line(img, (cx, cy - 30), (cx, cy + 30), (255, 0, 255), 1)

    return img


class SuckerThread1(QThread):
    _update_image_signal = pyqtSignal(np.ndarray, object)  # (图像, Bga_Strip|None)
    _update_statistics_signal = pyqtSignal(dict)
    _update_message_signal = pyqtSignal(str)

    def __init__(self, params: dict, HM: Hardware_Manager, MM: ModBus_Manager, ui: 'MainWindow'):
        super().__init__()
        self.HM = HM
        self.MM = MM
        self.ui = ui
        self.params = params or {}
        self.cam_alias = "sucker1_cam"
        self.modbus_alias = "sucker1_modbus"
        self.pixel_size = float(self.params.get("pixel_size", 0.001))
        self.sucker_detector = SuckerDetector(self._build_detector_params())
        self.is_paused = False
        self.live_display_enabled = False  # 线程安全标志，由主线程通过 set_live_display_enabled 更新

    def _build_detector_params(self):
        """从 params 构建 SuckerDetector 所需参数"""
        keys = [
            "pixel_size", "min_threshold_sucker", "max_threshold_sucker",
            "min_threshold_product", "max_threshold_product",
            "min_area_sucker", "min_area_product"
        ]
        return {k: self.params[k] for k in keys if k in self.params}

    def update_params(self, params: dict):
        self.params = params or {}
        self.pixel_size = float(self.params.get("pixel_size", 0.001))
        self.sucker_detector.update_params(self._build_detector_params())

    def run(self):
        print("SuckerThread1 start")
        trigger_last = 0
        # 初始化线圈为 0
        try:
            self.MM.write(self.modbus_alias, 0, [0], cst.WRITE_SINGLE_COIL)
        except Exception:
            pass

        _main_loop_logged = False
        while not self.isInterruptionRequested():
            if not _main_loop_logged:
                log_operation(
                    "SuckerThread1",
                    "主循环开始",
                    level=logging.INFO,
                    cam=self.cam_alias,
                    pixel_size=str(self.pixel_size),
                )
                _main_loop_logged = True
            if self.is_paused:
                time.sleep(0.1)
                continue

            # 读取 Modbus 离散输入
            ok_trigger, trigger_data = self.MM.read(
                self.modbus_alias, 0, 1, cst.READ_DISCRETE_INPUTS
            )
            ok_mode, mode_data = self.MM.read(
                self.modbus_alias, 3, 1, cst.READ_DISCRETE_INPUTS
            )
            trigger = trigger_data[0] if ok_trigger and trigger_data else 0
            mode = mode_data[0] if ok_mode and mode_data else 0

            # 采集图像
            ret, msg, image = self.HM.capture_image(self.cam_alias)

            if not ret or image is None:
                trigger_last = trigger
                time.sleep(0.01)
                continue

            # 上升沿触发
            if trigger == 1 and trigger_last == 0:
                start_time = time.time()
                if hasattr(self.ui, "radio_picturing_mode") and self.ui.radio_picturing_mode.isChecked():
                    image_result = ensure_bgr_u8(image, copy=True)
                    result = (0, 0, 0, 0)
                elif hasattr(self.ui, "radio_detecting_mode") and self.ui.radio_detecting_mode.isChecked():
                    detect_mode = "product" if mode == 0 else "sucker"
                    success, msg_det, result_dict = self.sucker_detector.detect(image, detect_mode)
                    if success and result_dict.get("is_valid"):
                        result = (
                            3,
                            result_dict.get("dis_y", 0),
                            result_dict.get("dis_x", 0),
                            result_dict.get("angle", 0),
                        )
                        image_result = _draw_result_on_image(image, result_dict)
                    else:
                        result = (0, 0, 0, 0)
                        image_result = _draw_result_on_image(image, result_dict or {})
                    if image_result is None:
                        image_result = ensure_bgr_u8(image, copy=True)
                else:
                    image_result = ensure_bgr_u8(image, copy=True)
                    result = (0, 0, 0, 0)

                # 写 Modbus 寄存器
                try:
                    self.MM.write(
                        self.modbus_alias, 2, list(result), cst.WRITE_MULTIPLE_REGISTERS
                    )
                    self.MM.write(self.modbus_alias, 0, [1], cst.WRITE_SINGLE_COIL)
                except Exception as e:
                    print(f"SuckerThread1 Modbus write error: {e}")

                # 发射图像信号（Bga 为 None，图像需为 BGR 格式）
                if image_result is not None:
                    self._update_image_signal.emit(ensure_bgr_u8(image_result, copy=True), None)

                print(f"SuckerThread1 cycle time: {(time.time() - start_time) * 1000:.1f} ms")
            elif trigger == 0 and trigger_last == 0:
                try:
                    self.MM.write(self.modbus_alias, 0, [0], cst.WRITE_SINGLE_COIL)
                except Exception:
                    pass

            # 实时显示（带辅助线）- 使用线程安全标志，禁止在工作线程中访问 UI
            if image is not None and self.live_display_enabled:
                h, w = image.shape[:2]
                img_live = ensure_bgr_u8(image, copy=True)
                cv.line(img_live, (0, h // 2), (w, h // 2), (0, 255, 0), 2)
                cv.line(img_live, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)

                self._update_image_signal.emit(img_live, None)

            trigger_last = trigger
            time.sleep(0.01)

        log_operation("SuckerThread1", "主循环退出", level=logging.INFO)

    def set_live_display_enabled(self, enabled: bool):
        """由主线程调用，更新实时显示开关（避免工作线程直接访问 UI 导致死锁）"""
        self.live_display_enabled = enabled

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

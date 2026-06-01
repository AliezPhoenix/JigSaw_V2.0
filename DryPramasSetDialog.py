from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from ui.DryPramasSetDialog_ui import Ui_DryPramasSetDialog
import cv2 as cv
import numpy as np
from src.detectors.ball_detector import BallDetector
from src.detectors.mark_detector import MarkDetector
from src.detectors.size_detector import SizeDetector
from src.detectors.shift_detector import ShiftDetector
from src.detectors.scratch_detector import ScratchDetector
from src.config.config_manager import ConfigManager
from src.support.support_funs import (
    selectROI,
    draw_detection_results,
    execute_product_detection,
    ensure_gray_u8,
    ensure_bgr_u8,
)
from ui.mark_roi_manage_dialog import MarkRoiManageDialog

class DryPramasSetDialog(Ui_DryPramasSetDialog, QDialog):
    def __init__(self,config_manager:'ConfigManager',parent=None):
        super().__init__(parent)
        # 设置界面（从 .ui 文件生成的代码）
        self.setupUi(self)
        self.is_init = False
        self.showMaximized()

        self.ball_detector = BallDetector()
        self.size_detector = SizeDetector()
        self.shift_detector = ShiftDetector()
        self.mark_detector = MarkDetector()
        self.scratch_detector = ScratchDetector()
        self.config_manager = config_manager
        self.local_params = self.config_manager.get_section("work_dry_params")
        self._migrate_mark_roi_min_areas()

        # 存储原始图像
        self.template_image = None  # 模板图像（用于显示，可能包含屏蔽效果，用于检测）

        # 设置 SpinBox 范围
        self.spin_thresh_lower_ball.setMinimum(0)
        self.spin_thresh_upper_ball.setMinimum(0)
        self.spin_thresh_lower_size.setMinimum(0)
        self.spin_thresh_upper_size.setMinimum(0)
        self.spin_area_min.setMinimum(0)
        self.spin_area_max.setMinimum(0)
        # Mark检测的 SpinBox
        self.spinBox_threshold_min_mark_dry.setMinimum(0)
        self.spinBox_threshold_max_mark_dry.setMinimum(0)
        # 划痕检测的 SpinBox
        self.spin_thresh_lower_scratch.setMinimum(0)
        self.spin_thresh_upper_scratch.setMinimum(0)
        
        # 设置 Slider 范围（与 SpinBox 同步）
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(255)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(255)
        self.horizontalSlider_3.setMinimum(0)
        self.horizontalSlider_3.setMaximum(255)
        self.horizontalSlider_4.setMinimum(0)
        self.horizontalSlider_4.setMaximum(255)
        # Mark检测的 Slider
        self.horizontalSlider_threshold_min_mark_dry.setMinimum(0)
        self.horizontalSlider_threshold_min_mark_dry.setMaximum(255)
        self.horizontalSlider_threshold_max_mark_dry.setMinimum(0)
        self.horizontalSlider_threshold_max_mark_dry.setMaximum(255)
        # 划痕检测的 Slider
        self.horizontalSlider_thresh_lower_scratch.setMinimum(0)
        self.horizontalSlider_thresh_lower_scratch.setMaximum(255)
        self.horizontalSlider_thresh_upper_scratch.setMinimum(0)
        self.horizontalSlider_thresh_upper_scratch.setMaximum(255)
        
        # 绑定 Slider 和 SpinBox 的双向同步
        # 尺寸检测的 threshold
        self.horizontalSlider_3.valueChanged.connect(self.spin_thresh_lower_size.setValue)
        self.spin_thresh_lower_size.valueChanged.connect(self.horizontalSlider_3.setValue)
        self.horizontalSlider_4.valueChanged.connect(self.spin_thresh_upper_size.setValue)
        self.spin_thresh_upper_size.valueChanged.connect(self.horizontalSlider_4.setValue)
        
        # 锡球检测的 threshold
        self.horizontalSlider.valueChanged.connect(self.spin_thresh_lower_ball.setValue)
        self.spin_thresh_lower_ball.valueChanged.connect(self.horizontalSlider.setValue)
        self.horizontalSlider_2.valueChanged.connect(self.spin_thresh_upper_ball.setValue)
        self.spin_thresh_upper_ball.valueChanged.connect(self.horizontalSlider_2.setValue)
        
        # Mark检测的 threshold
        self.horizontalSlider_threshold_min_mark_dry.valueChanged.connect(self.spinBox_threshold_min_mark_dry.setValue)
        self.spinBox_threshold_min_mark_dry.valueChanged.connect(self.horizontalSlider_threshold_min_mark_dry.setValue)
        self.horizontalSlider_threshold_max_mark_dry.valueChanged.connect(self.spinBox_threshold_max_mark_dry.setValue)
        self.spinBox_threshold_max_mark_dry.valueChanged.connect(self.horizontalSlider_threshold_max_mark_dry.setValue)
        
        # 划痕检测的 threshold
        self.horizontalSlider_thresh_lower_scratch.valueChanged.connect(self.spin_thresh_lower_scratch.setValue)
        self.spin_thresh_lower_scratch.valueChanged.connect(self.horizontalSlider_thresh_lower_scratch.setValue)
        self.horizontalSlider_thresh_upper_scratch.valueChanged.connect(self.spin_thresh_upper_scratch.setValue)
        self.spin_thresh_upper_scratch.valueChanged.connect(self.horizontalSlider_thresh_upper_scratch.setValue)
        

        # 连接信号和槽
        self.btn_load_template.clicked.connect(self.load_template_image)
        self.btn_run_test.clicked.connect(lambda: self.run_template_test("all"))
        self.btn_close.clicked.connect(self.accept)
        self.btn_load_params.clicked.connect(self.update_params)
        self.btn_ignore_roi.clicked.connect(lambda: self.create_roi("block"))
        self.btn_confirm_load.clicked.connect(self.confirm_load)
        self.btn_clear_roi.clicked.connect(lambda: self.clear_check_roi("block"))
        # Mark检测区域
        self.pushButton_manage_mark_roi_dry.clicked.connect(self._open_mark_roi_manage)
        self.pushButton_create_ball_search_roi_dry.clicked.connect(lambda: self.create_roi("ball"))
        self.pushButton_clear_ball_search_roi_dry.clicked.connect(lambda:self.clear_check_roi("ball_search"))
        # 划痕检测区域相关按钮
        self.pushButton_create_scratch_roi_dry.clicked.connect(lambda: self.create_roi("scratch"))
        self.pushButton_clear_scratch_roi_dry.clicked.connect(lambda:self.clear_check_roi("scratch"))
        

        #——————————————————————detector实例化————————————————————

        
        # 加载本地参数
 
        
    
        self.update_params()
        # 锡球检测的 slider 触发 ball 检测
        self.horizontalSlider.valueChanged.connect(lambda: self.auto_run_test("ball"))
        self.horizontalSlider_2.valueChanged.connect(lambda: self.auto_run_test("ball"))
        # 尺寸检测的 slider 触发 size 检测
        self.horizontalSlider_3.valueChanged.connect(lambda: self.auto_run_test("size"))
        self.horizontalSlider_4.valueChanged.connect(lambda: self.auto_run_test("size"))
        # Mark检测的 slider 触发 mark 检测
        self.horizontalSlider_threshold_min_mark_dry.valueChanged.connect(lambda: self.auto_run_test("mark"))
        self.horizontalSlider_threshold_max_mark_dry.valueChanged.connect(lambda: self.auto_run_test("mark"))
        # 划痕检测的 slider 触发 scratch 检测
        self.horizontalSlider_thresh_lower_scratch.valueChanged.connect(lambda: self.auto_run_test("scratch"))
        self.horizontalSlider_thresh_upper_scratch.valueChanged.connect(lambda: self.auto_run_test("scratch"))
        
        self.is_init = True
    
    def auto_run_test(self,detect_type=None):
        """当 Slider 数值变化时自动触发测试（仅在初始化完成后）"""
        if self.is_init and self.template_image is not None:
            self.run_template_test(detect_type)
    
    def load_template_image(self):
        """加载模板图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模板图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            image = cv.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "错误", "无法加载图像文件")
                return
            
            # 保存原始模板图像（用于模板匹配）
            self.template_image = image.copy()
            # 用于显示的模板图像（包含标记）
            self._update_template_display_with_markers()
            self.info_label.setText(f"已加载模板图像: {file_path}")
    
    def display_template_image(self, image):
        if image is None:
            return
        image = ensure_bgr_u8(image, copy=True)
        height, width, channel = image.shape
        label = self.template_image_label
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        # 获取 label 的大小，保持宽高比缩放
        label_size = label.size()
        # 如果label尺寸无效，先处理事件等待更新
        if label_size.width() <= 0 or label_size.height() <= 0:
            QApplication.processEvents()
            label_size = label.size()
            # 如果仍然无效，使用sizeHint或默认尺寸
            if label_size.width() <= 0 or label_size.height() <= 0:
                size_hint = label.sizeHint()
                if size_hint.width() > 0 and size_hint.height() > 0:
                    label_size = size_hint
                else:
                    # 使用默认尺寸
                    label_size = label.geometry().size()
                    if label_size.width() <= 0 or label_size.height() <= 0:
                        label_size = label.parent().size() if label.parent() else label.size()
        
        scaled_pixmap = pixmap.scaled(label_size.width(), label_size.height(), 
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setScaledContents(False)
        label.setPixmap(scaled_pixmap)
        label.update()
        QApplication.processEvents()

    def _pixmap_to_bgr_image(self, pixmap: QPixmap):
        """
        将 QLabel/QPixmap 图像转换为 OpenCV 可处理的 BGR uint8 ndarray。
        返回 None 表示转换失败或 pixmap 无效。
        """
        if pixmap is None or pixmap.isNull():
            return None
        try:
            qimg = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            width = qimg.width()
            height = qimg.height()
            if width <= 0 or height <= 0:
                return None

            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            # bytesPerLine 可能包含对齐填充，按 stride reshape 后再裁到真实 width
            stride = qimg.bytesPerLine() // 4
            rgba = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stride, 4))[:, :width, :]
            bgr = cv.cvtColor(rgba, cv.COLOR_RGBA2BGR)
            return bgr.copy()
        except Exception:
            return None
        
    def _update_template_display_with_markers(self):
        """更新模板显示，包括绘制ROI屏蔽区域和Mark检测区域"""
        if self.template_image is None:
            return
        
        display_image = ensure_bgr_u8(self.template_image, copy=True)
        
        # 绘制ROI屏蔽区域（黑色填充）
        roi_block = self.local_params.get("roi_block", [])
        if isinstance(roi_block, list):
            for roi in roi_block:
                if isinstance(roi, (tuple, list)) and len(roi) >= 4:
                    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                    cv.rectangle(display_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        # 绘制Mark检测区域（多 ROI，颜色区分）
        mark_rois = self.local_params.get("mark_rois", [])
        colors = [(255, 0, 0), (255, 128, 0), (200, 0, 200), (0, 128, 255)]
        if isinstance(mark_rois, list):
            for idx, roi in enumerate(mark_rois):
                if isinstance(roi, (tuple, list)) and len(roi) >= 4:
                    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                    c = colors[idx % len(colors)]
                    cv.rectangle(display_image, (x, y), (x + w, y + h), c, 2)
        
        # 绘制锡球搜索区域（黄色边框）
        ball_search_roi = self.local_params.get("ball_search_roi", [])
        if isinstance(ball_search_roi, list) and len(ball_search_roi) == 4:
            x, y, w, h = int(ball_search_roi[0]), int(ball_search_roi[1]), int(ball_search_roi[2]), int(ball_search_roi[3])
            cv.rectangle(display_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # 绘制划痕检测区域（绿色边框）
        scratch_roi = self.local_params.get("scratch_roi", [])
        if isinstance(scratch_roi, list) and len(scratch_roi) == 4:
            x, y, w, h = int(scratch_roi[0]), int(scratch_roi[1]), int(scratch_roi[2]), int(scratch_roi[3])
            cv.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.display_template_image(display_image)

    def _migrate_mark_roi_min_areas(self):
        """旧键 min_mark_area → mark_roi_min_areas；保存前不再依赖 min_mark_area。"""
        lp = self.local_params
        mrs = lp.get("mark_rois") or []
        if "mark_roi_min_areas" not in lp or lp["mark_roi_min_areas"] is None:
            if mrs and "min_mark_area" in lp:
                try:
                    v = int(lp["min_mark_area"])
                except (TypeError, ValueError):
                    v = 0
                lp["mark_roi_min_areas"] = [v] * len(mrs)
            else:
                lp["mark_roi_min_areas"] = []
        self._sync_mark_roi_min_areas_len()
        lp.pop("min_mark_area", None)

    def _sync_mark_roi_min_areas_len(self):
        mrs = self.local_params.get("mark_rois") or []
        areas = list(self.local_params.get("mark_roi_min_areas") or [])
        while len(areas) < len(mrs):
            areas.append(-1)
        self.local_params["mark_roi_min_areas"] = areas[: len(mrs)]

    def _open_mark_roi_manage(self):
        dlg = MarkRoiManageDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self._sync_mark_roi_min_areas_len()
            self._update_template_display_with_markers()
            if self.is_init and self.template_image is not None:
                self.auto_run_test("mark")

    def display_processed_image(self, image):
        """在处理后图像 label 中显示图像"""
        if image is None:
            return
        
        bgr = ensure_bgr_u8(image, copy=True)
        rgb_image = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        channel = rgb_image.shape[2]
        bytes_per_line = channel * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        
        # 缩放图像以适应 label（保持宽高比）
        scaled_pixmap = pixmap.scaled(
            self.processed_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.processed_image_label.setPixmap(scaled_pixmap)
    
    def _draw_detection_results(self, image_result:np.ndarray, product_info:dict):
        """在图像上绘制检测结果（调用通用方法）
        
        Args:
            image_result: 结果图像（BGR格式或灰度图）
            product_info: 产品信息字典，包含检测结果元组
        
        Returns:
            np.ndarray: 绘制后的图像
        """
        success, msg, result_image = draw_detection_results(image_result, product_info, mark_color="red")
        if msg:
            print(msg)
        return result_image
    
    def _generate_result_text(self, product_info):
        """生成检测结果文本
        
        Args:
            product_info: 产品信息字典，包含检测结果
        
        Returns:
            str: 格式化的结果文本
        """
        defect_type_list = product_info.get("defect_type", ["OK"])
        roi_block = self.local_params.get("roi_block", [])
        pixel_size = self.local_params.get("pixel_size", 0.001)
        
        should_detect_size = self.local_params.get("size_check_enable", True)
        should_detect_ball = self.local_params.get("ball_check_enable", True)
        should_detect_mark = self.local_params.get("mark_check_enable", True)
        should_detect_scratch = self.local_params.get("scratch_check_enable", True)
        should_detect_shift = self.local_params.get("shift_check_enable", True)
        
        result_text = "=" * 50 + "\n"
        result_text += "处理结果详情\n"
        result_text += "=" * 50 + "\n\n"
        
        # 总体结果
        result_text += f"【总体结果】\n"
        result_text += f"判定: {defect_type_list[0]}\n"
        result_text += f"缺陷类型: {', '.join(defect_type_list[1:]) if len(defect_type_list) > 1 else '无'}\n"
        result_text += f"当前屏蔽区域数量: {len(roi_block)}\n\n"
        
        # Mark检测结果
        if should_detect_mark and product_info.get("mark_result") is not None:
            try:
                mark_result = product_info["mark_result"]
                if isinstance(mark_result, tuple) and len(mark_result) >= 3:
                    result_dict = mark_result[2]
                    is_valid = result_dict.get("is_valid", False)
                    allow_mark = self.local_params.get("allow_mark", False)
                    result_text += f"【Mark检测】\n"
                    per_roi = result_dict.get("per_roi") or []
                    if per_roi:
                        result_text += f"聚合({'AND' if allow_mark else 'OR'}) is_valid={is_valid}\n"
                        for pr in per_roi:
                            iv = pr.get("is_valid", False)
                            idx = pr.get("index", 0) + 1
                            a = float(pr.get("mark_area", 0.0))
                            amm = float(
                                pr.get("mark_area_mm", a * pixel_size * pixel_size)
                            )
                            st = "有Mark" if iv else "无Mark"
                            result_text += f"  ROI{idx}: {st}, 面积 {a:.1f}像素² ({amm:.2f}mm²)\n"
                    else:
                        mark_area = result_dict.get("mark_area", 0.0)
                        mark_area_mm = result_dict.get(
                            "mark_area_mm", mark_area * pixel_size * pixel_size
                        )
                        if mark_area > 0:
                            result_text += f"Mark面积: {mark_area:.1f}像素² ({mark_area_mm:.2f}mm²)\n"
                        else:
                            result_text += f"Mark面积: 未检测到\n"
                    if allow_mark:
                        result_text += f"判定: {'OK (Mark齐全)' if is_valid else 'NG (Mark不全)'}\n"
                    else:
                        result_text += f"判定: {'NG (检测到Mark)' if is_valid else 'OK (未检测到Mark)'}\n"
                    result_text += "\n"
            except Exception as e:
                result_text += f"【Mark检测】\n"
                result_text += f"错误: {str(e)}\n\n"
        
        # 尺寸检测结果
        if should_detect_size and product_info.get("size_result") is not None:
            try:
                size_result = product_info["size_result"]
                if isinstance(size_result, tuple) and len(size_result) >= 3:
                    result_dict = size_result[2]
                    width = result_dict.get("width", None)
                    height = result_dict.get("height", None)
                    is_valid = result_dict.get("is_valid", False)
                    
                    result_text += f"【尺寸检测】\n"
                    result_text += f"判定: {'OK' if is_valid else 'NG'}\n"
                    if width is not None and height is not None:
                        result_text += f"产品尺寸: 宽 {width:.4f}mm × 高 {height:.4f}mm\n"
                    else:
                        result_text += f"产品尺寸: 检测失败\n"
                    result_text += "\n"
            except Exception as e:
                result_text += f"【尺寸检测】\n"
                result_text += f"错误: {str(e)}\n\n"
        
        # 锡球检测结果
        if should_detect_ball and product_info.get("ball_result") is not None:
            try:
                ball_result = product_info["ball_result"]
                if isinstance(ball_result, tuple) and len(ball_result) >= 3:
                    result_dict = ball_result[2]
                    ball_count = result_dict.get("ball_count", 0)
                    avg_radius = result_dict.get("avg_radius", 0.0)
                    avg_radius_mm = result_dict.get("avg_radius_mm", avg_radius * pixel_size)
                    ok_details = result_dict.get("ok_details", [])
                    ng_details = result_dict.get("ng_details", [])
                    is_valid = result_dict.get("is_valid", False)
                    
                    result_text += f"【锡球检测】\n"
                    result_text += f"判定: {'OK' if is_valid else 'NG'}\n"
                    result_text += f"锡球总数: {ball_count}\n"
                    expected_count = self.local_params.get("ball_count", 0)
                    if expected_count > 0:
                        result_text += f"期望数量: {expected_count}\n"
                    if avg_radius_mm > 0:
                        result_text += f"平均半径: {avg_radius_mm:.2f}mm\n"
                    result_text += "\n"
                    
                    if ok_details:
                        result_text += f"合格锡球 ({len(ok_details)}颗):\n"
                        for i, detail in enumerate(ok_details, 1):
                            radius_mm = detail.get("radius_mm", detail.get("radius", 0.0) * pixel_size)
                            area_mm2 = detail.get("area_mm2", detail.get("area", 0.0) * pixel_size * pixel_size)
                            center = detail.get("center", (0, 0))
                            result_text += f"  锡球{i}: 半径={radius_mm:.2f}mm, "
                            result_text += f"面积={area_mm2:.2f}mm², "
                            result_text += f"中心=({center[0]}, {center[1]})\n"
                        result_text += "\n"
                    
                    if ng_details:
                        result_text += f"不合格锡球 ({len(ng_details)}颗):\n"
                        for i, detail in enumerate(ng_details, 1):
                            radius_mm = detail.get("radius_mm", detail.get("radius", 0.0) * pixel_size)
                            area_mm2 = detail.get("area_mm2", detail.get("area", 0.0) * pixel_size * pixel_size)
                            radius_diff_mm = detail.get("radius_diff_mm", 0.0)
                            center = detail.get("center", (0, 0))
                            result_text += f"  锡球{i}: 半径={radius_mm:.2f}mm, "
                            result_text += f"面积={area_mm2:.2f}mm², "
                            if radius_diff_mm > 0:
                                result_text += f"半径偏差={radius_diff_mm:.2f}mm, "
                            result_text += f"中心=({center[0]}, {center[1]})\n"
                        result_text += "\n"
                    
                    # 偏移检测结果（与锡球检测一起显示）
                    if should_detect_shift and product_info.get("shift_result") is not None:
                        try:
                            shift_result = product_info["shift_result"]
                            shift_dict = shift_result[2]
                            shift_x = shift_dict.get("shift_x", 0.0)
                            shift_y = shift_dict.get("shift_y", 0.0)
                            ball_center = shift_dict.get("ball_center", None)
                            size_center = shift_dict.get("size_center", None)
                            is_valid_shift = shift_dict.get("is_valid", False)
                            
                            shift_x_tolerance = self.local_params.get("shift_x_tolerance", 0.1)
                            shift_y_tolerance = self.local_params.get("shift_y_tolerance", 0.1)
                            
                            result_text += f"【偏移检测】\n"
                            result_text += f"判定: {'OK' if is_valid_shift else 'NG'}\n"
                            result_text += f"偏移量: X={shift_x:.4f}mm, Y={shift_y:.4f}mm\n"
                            result_text += f"容差: X=±{shift_x_tolerance:.4f}mm, Y=±{shift_y_tolerance:.4f}mm\n"
                            if ball_center is not None and len(ball_center) >= 2:
                                result_text += f"球中心位置: ({ball_center[0]:.2f}, {ball_center[1]:.2f}) 像素\n"
                            if size_center is not None and len(size_center) >= 2:
                                result_text += f"尺寸中心位置: ({size_center[0]:.2f}, {size_center[1]:.2f}) 像素\n"
                            result_text += "\n"
                        except Exception as e:
                            result_text += f"【偏移检测】\n"
                            result_text += f"错误: {str(e)}\n\n"
                    elif should_detect_shift:
                        result_text += f"【偏移检测】\n"
                        result_text += f"需要同时启用尺寸检测和锡球检测\n\n"
            except Exception as e:
                result_text += f"【锡球检测】\n"
                result_text += f"错误: {str(e)}\n\n"
        
        # 划痕检测结果
        if should_detect_scratch and product_info.get("scratch_result") is not None:
            try:
                scratch_result = product_info["scratch_result"]
                if isinstance(scratch_result, tuple) and len(scratch_result) >= 3:
                    result_dict = scratch_result[2]
                    is_valid = result_dict.get("is_valid", False)
                    ng_scratch_contours = result_dict.get("ng_scratch_contours", [])
                    
                    result_text += f"【划痕检测】\n"
                    result_text += f"判定: {'OK' if is_valid else 'NG'}\n"
                    scratch_length_threshold = self.local_params.get("scratch_length", 5.0)
                    result_text += f"划痕长度阈值: {scratch_length_threshold}mm\n"
                    if ng_scratch_contours:
                        result_text += f"检测到 {len(ng_scratch_contours)} 条NG划痕\n"
                    result_text += "\n"
            except Exception as e:
                result_text += f"【划痕检测】\n"
                result_text += f"错误: {str(e)}\n\n"
        
        result_text += "=" * 50 + "\n"
        
        return result_text
 
    def run_template_test(self, detect_type=None):
        """运行模板测试：根据检测开关执行所有启用的检测
        
        Args:
            detect_type: 已废弃，保留以兼容旧代码。检测由检测开关控制。
        """
        # 检查是否已加载测试图像和模板图像

        try:
            # 从界面更新参数到local_params
            self._update_params_from_ui()
            self.update_params()

            
            if self.local_params.get("ball_area_min_threshold") >= self.local_params.get("ball_area_max_threshold"):
                QMessageBox.warning(self, "错误", "锡球面积下限必须小于上限")
                return
            
            template_gray = ensure_gray_u8(self.template_image, copy=True)
            
            # 构建检测器字典（与 main_window.template_validity_test 一致）
            detectors = {
                "ball_detector": self.ball_detector,
                "size_detector": self.size_detector,
                "mark_detector": self.mark_detector,
                "shift_detector": self.shift_detector,
                "scratch_detector": self.scratch_detector,
            }

            # 检测开关统一从 local_params 获取
            detect_params = {
                "mark_check_enable": self.local_params.get("mark_check_enable", True),
                "size_check_enable": self.local_params.get("size_check_enable", True),
                "ball_check_enable": self.local_params.get("ball_check_enable", True),
                "shift_check_enable": self.local_params.get("shift_check_enable", True),
                "scratch_check_enable": self.local_params.get("scratch_check_enable", True),
                "allow_mark": self.local_params.get("allow_mark", False),
                "roi_block": self.local_params.get("roi_block", []),
            }

            # detect_type 为 None 或 "all" 时使用 local_params 的开关；为 ball/size/mark/scratch/shift 时仅执行该检测
            pass_detect_type = None if (detect_type is None or detect_type == "all") else detect_type

            success, msg, product_info = execute_product_detection(
                image=template_gray,
                detectors=detectors,
                params=detect_params,
                detect_type=pass_detect_type,
                early_return_on_ng=False,
                error_callback=None
            )

            if not success:
                QMessageBox.warning(self, "警告", msg)
                return
            
            # 绘制检测结果
            image_result = self._draw_detection_results(self.template_image, product_info)
            
            # 拼接预处理图像和结果图像
            if detect_type == "all":
                image_binary = cv.inRange(template_gray,self.local_params["min_threshold_mark"],self.local_params["max_threshold_mark"])
            elif detect_type == "size":
                #image_gray = cv.equalizeHist(template_gray)
                image_binary = cv.inRange(template_gray, self.local_params["min_threshold_size"], self.local_params["max_threshold_size"])
                #image_binary = cv.bitwise_not(image_binary)
            elif detect_type == "ball":
                image_binary = cv.inRange(template_gray,self.local_params["min_threshold_ball"],self.local_params["max_threshold_ball"])
            elif detect_type == "mark":
                image_binary = cv.inRange(template_gray,self.local_params["min_threshold_mark"],self.local_params["max_threshold_mark"])
            elif detect_type == "scratch":
                image_binary = cv.inRange(template_gray,self.local_params["min_threshold_scratch"],self.local_params["max_threshold_scratch"])

            image_binary = ensure_bgr_u8(image_binary, copy=True)
            self.processed_image = np.vstack((image_binary, image_result))
            self.display_processed_image(self.processed_image)
            
            # 生成并显示文本结果
            result_text = self._generate_result_text(product_info)
            self.textEdit_test_result.setPlainText(result_text)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"测试运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_params(self):
        """加载参数"""
        if self.config_manager is None:
            QMessageBox.warning(self, "错误", "请先加载配方")
            return     
        try:
            pixel_size = self.local_params["pixel_size"]
            product_size = self.local_params["product_size"]
            roi_block = self.local_params["roi_block"]
            #————————————————————————ball参数————————————————————————————————
            min_threshold_ball = self.local_params["min_threshold_ball"]
            max_threshold_ball = self.local_params["max_threshold_ball"]
            ball_area_min_threshold = self.local_params["ball_area_min_threshold"]
            ball_area_max_threshold = self.local_params["ball_area_max_threshold"]
            ball_count = self.local_params["ball_count"]
            ball_radius_tolerance = self.local_params["ball_radius_tolerance"]
            std_radius = self.local_params["std_radius"]
            ball_search_roi = self.local_params["ball_search_roi"]
            

            self.ball_detector.update_params(
                {
                    "min_threshold": min_threshold_ball,
                    "max_threshold": max_threshold_ball,
                    "ball_area_min_threshold": ball_area_min_threshold,
                    "ball_area_max_threshold": ball_area_max_threshold,
                    "ball_radius_tolerance": ball_radius_tolerance,
                    "std_radius": std_radius,
                    "expected_ball_count": ball_count,
                    "pixel_size": pixel_size,
                    "ball_search_roi": ball_search_roi,
                })
            
            #————————————————————size参数————————————————————————
            product_size_tolerance_x = self.local_params["product_size_tolerance_x"]
            product_size_tolerance_y = self.local_params["product_size_tolerance_y"]
            min_threshold_size = self.local_params["min_threshold_size"]
            max_threshold_size = self.local_params["max_threshold_size"]

            self.size_detector.update_params(
                {
                    "min_threshold": min_threshold_size,
                    "max_threshold": max_threshold_size,
                    "allow_tolerance_x": product_size_tolerance_x,
                    "allow_tolerance_y": product_size_tolerance_y,
                    "roi_width": 80,
                    "std_size": (product_size[0], product_size[1]),
                    "pixel_size": pixel_size,
                    "pixel_size_x": self.local_params.get("pixel_size_x"),
                })
            
            #————————————————————————mark参数——————————————————————
            self._migrate_mark_roi_min_areas()
            min_threshold_mark = self.local_params["min_threshold_mark"]
            max_threshold_mark = self.local_params["max_threshold_mark"]
            mark_rois = self.local_params.get("mark_rois", [])
            mark_roi_min_areas = list(self.local_params.get("mark_roi_min_areas") or [])
            self.mark_detector.update_params(
                {
                    "min_threshold": min_threshold_mark,
                    "max_threshold": max_threshold_mark,
                    "pixel_size": pixel_size,
                    "mark_detect_mode": "manual",
                    "mark_rois": mark_rois,
                    "mark_roi_min_areas": mark_roi_min_areas,
                    "allow_mark": self.local_params.get("allow_mark", False),
                })
            #————————————————————————scratch参数——————————————————————
            min_threshold_scratch = self.local_params["min_threshold_scratch"]
            max_threshold_scratch = self.local_params["max_threshold_scratch"]
            scratch_length = self.local_params["scratch_length"]
            scratch_roi = self.local_params["scratch_roi"]
 
            self.lineEdit_scratch_length.setText(str(scratch_length))
            
            self.scratch_detector.update_params(
                {
                    "min_threshold": min_threshold_scratch,
                    "max_threshold": max_threshold_scratch,
                    "scratch_length_threshold": scratch_length,  # 注意：参数名是scratch_length_threshold
                    "pixel_size": pixel_size,
                    "scratch_roi": scratch_roi,
                    "roi_blocks": roi_block,
                })
            #——————————————————————偏移检测参数——————————————————————
            shift_x_tolerance = self.local_params["shift_x_tolerance"]
            shift_y_tolerance = self.local_params["shift_y_tolerance"]
            self.shift_detector.update_params(
                {
                    "allow_tolerance_x": shift_x_tolerance,
                    "allow_tolerance_y": shift_y_tolerance,
                    "pixel_size": pixel_size,
                    "error_correction_factor": 0.7,
                })
   
            if not self.is_init:
                size_check_enable = self.local_params["size_check_enable"]
                ball_check_enable = self.local_params["ball_check_enable"]
                mark_check_enable = self.local_params["mark_check_enable"]
                scratch_check_enable = self.local_params["scratch_check_enable"]
                shift_check_enable = self.local_params["shift_check_enable"]
                self.radioButton_size_check_enable.setChecked(bool(size_check_enable))
                self.radioButton_ball_check_enable.setChecked(bool(ball_check_enable))
                self.radioButton_mark_check_enable.setChecked(bool(mark_check_enable))
                self.radioButton_scratch_check_enable.setChecked(bool(scratch_check_enable))
                self.radioButton_shift_check_enable.setChecked(bool(shift_check_enable))
                self.spin_thresh_lower_ball.setValue(min_threshold_ball)
                self.spin_thresh_upper_ball.setValue(max_threshold_ball)
                self.horizontalSlider.setValue(min_threshold_ball)
                self.horizontalSlider_2.setValue(max_threshold_ball)
                self.spin_area_min.setValue(ball_area_min_threshold)
                self.spin_area_max.setValue(ball_area_max_threshold)
                self.line_ballnum_allow.setText(str(ball_count))
                self.line_ball_radius_tolerance_allow.setText(str(ball_radius_tolerance))
                self.lineEdit_ball_radius_std.setText(str(std_radius))
                self.line_X_tolerance_allow.setText(str(product_size_tolerance_x))
                self.line_Y_tolerance_allow.setText(str(product_size_tolerance_y))
                self.spin_thresh_lower_size.setValue(int(min_threshold_size))
                self.spin_thresh_upper_size.setValue(int(max_threshold_size))
                self.horizontalSlider_3.setValue(int(min_threshold_size))
                self.horizontalSlider_4.setValue(int(max_threshold_size))
                self.spinBox_threshold_min_mark_dry.setValue(int(min_threshold_mark))
                self.spinBox_threshold_max_mark_dry.setValue(int(max_threshold_mark))
                self.horizontalSlider_threshold_min_mark_dry.setValue(int(min_threshold_mark))
                self.horizontalSlider_threshold_max_mark_dry.setValue(int(max_threshold_mark))
                self.spin_thresh_lower_scratch.setValue(int(min_threshold_scratch))
                self.spin_thresh_upper_scratch.setValue(int(max_threshold_scratch))
                self.horizontalSlider_thresh_lower_scratch.setValue(int(min_threshold_scratch))
                self.horizontalSlider_thresh_upper_scratch.setValue(int(max_threshold_scratch))
                self.lineEdit_shift_x_tolerance.setText(str(shift_x_tolerance))
                self.lineEdit_shift_y_tolerance.setText(str(shift_y_tolerance))
                # 确保UI更新完成，特别是label的尺寸
                QApplication.processEvents()
            
                parent_label = getattr(self.parent(), "label_template_display_dry") 
                template_pixmap = parent_label.pixmap() 
                template_image = self._pixmap_to_bgr_image(template_pixmap)
                if template_image is not None:
                    self.template_image = template_image
                    print(f"template_image: {self.template_image.shape}")
                    QApplication.processEvents()
                    self._update_template_display_with_markers()
                else:
                    QMessageBox.warning(self, "警告", "无法从模板显示区域读取有效图像，请先加载模板图像")
                QApplication.processEvents()
                QMessageBox.information(self, "成功", "参数已成功加载")
            
            # ng_monitor 报警参数（每次加载都同步到 UI）
            ng_monitor = self.local_params.get("ng_monitor", {})
            self.checkBox_ng_monitor_enabled.setChecked(ng_monitor.get("monitor_enabled", True))
            limits = ng_monitor.get("defect_limits", {})
            defect_edit_map = [
                ("Mark", "lineEdit_defect_limit_Mark", 5),
                ("Size", "lineEdit_defect_limit_Size", 3),
                ("Ball_Area", "lineEdit_defect_limit_Area", 3),
                ("Ball Count", "lineEdit_defect_limit_BallCount", 3),
                ("Scratch", "lineEdit_defect_limit_Scratch", 3),
                ("Shift", "lineEdit_defect_limit_Shift", 3),
            ]
            for key, attr, default in defect_edit_map:
                val = limits.get("Ball_Area", limits.get("Area", default)) if key == "Ball_Area" else limits.get(key, default)
                getattr(self, attr).setText(str(val))
            self.lineEdit_min_yield_rate.setText(str(ng_monitor.get("min_yield_rate", 95.0)))
            self.lineEdit_min_yield_sample_size.setText(str(ng_monitor.get("min_yield_sample_size", 100)))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载参数失败: {str(e)}")
            import traceback
            traceback.print_exc()
      
    def _update_params_from_ui(self):
        """从当前界面获取用户设置的参数并写入local_params中"""
        try:
            # 锡球检测参数
            self.local_params["min_threshold_ball"] = int(self.spin_thresh_lower_ball.value())
            self.local_params["max_threshold_ball"] = int(self.spin_thresh_upper_ball.value())
            self.local_params["ball_area_min_threshold"] = int(self.spin_area_min.value())
            self.local_params["ball_area_max_threshold"] = int(self.spin_area_max.value())
            self.local_params["ball_radius_tolerance"] = float(self.line_ball_radius_tolerance_allow.text())
            
            # 从LineEdit获取参数（需要处理空值）
            if hasattr(self, 'line_ballnum_allow') and self.line_ballnum_allow.text().strip():
                self.local_params["ball_count"] = int(self.line_ballnum_allow.text())
            if hasattr(self, 'line_ball_radius_tolerance_allow') and self.line_ball_radius_tolerance_allow.text().strip():
                self.local_params["ball_radius_tolerance"] = float(self.line_ball_radius_tolerance_allow.text())
            if hasattr(self, 'lineEdit_ball_radius_std') and self.lineEdit_ball_radius_std.text().strip():
                self.local_params["std_radius"] = float(self.lineEdit_ball_radius_std.text())
            
            # 尺寸检测参数
            if hasattr(self, 'spin_thresh_lower_size'):
                self.local_params["min_threshold_size"] = int(self.spin_thresh_lower_size.value())
            if hasattr(self, 'spin_thresh_upper_size'):
                self.local_params["max_threshold_size"] = int(self.spin_thresh_upper_size.value())
            
            # 尺寸容差参数
            if hasattr(self, 'line_X_tolerance_allow') and hasattr(self, 'line_Y_tolerance_allow'):
                x_text = self.line_X_tolerance_allow.text().strip()
                y_text = self.line_Y_tolerance_allow.text().strip()
                if x_text and y_text:
                    self.local_params["product_size_tolerance_x"] = float(x_text)
                    self.local_params["product_size_tolerance_y"] = float(y_text)
            
            # Mark检测参数
            if hasattr(self, 'spinBox_threshold_min_mark_dry'):
                self.local_params["min_threshold_mark"] = int(self.spinBox_threshold_min_mark_dry.value())
            if hasattr(self, 'spinBox_threshold_max_mark_dry'):
                self.local_params["max_threshold_mark"] = int(self.spinBox_threshold_max_mark_dry.value())
            
            # 划痕检测参数
            if hasattr(self, 'spin_thresh_lower_scratch'):
                self.local_params["min_threshold_scratch"] = int(self.spin_thresh_lower_scratch.value())
            if hasattr(self, 'spin_thresh_upper_scratch'):
                self.local_params["max_threshold_scratch"] = int(self.spin_thresh_upper_scratch.value())
            if hasattr(self, 'lineEdit_scratch_length') and self.lineEdit_scratch_length.text().strip():
                self.local_params["scratch_length"] = float(self.lineEdit_scratch_length.text())
            
            # 偏移检测容差参数
            if hasattr(self, 'lineEdit_shift_x_tolerance') and self.lineEdit_shift_x_tolerance.text().strip():
                self.local_params["shift_x_tolerance"] = float(self.lineEdit_shift_x_tolerance.text())
            if hasattr(self, 'lineEdit_shift_y_tolerance') and self.lineEdit_shift_y_tolerance.text().strip():
                self.local_params["shift_y_tolerance"] = float(self.lineEdit_shift_y_tolerance.text())
            
            # 检测开关参数
            if hasattr(self, 'radioButton_size_check_enable'):
                self.local_params["size_check_enable"] = self.radioButton_size_check_enable.isChecked()
            if hasattr(self, 'radioButton_ball_check_enable'):
                self.local_params["ball_check_enable"] = self.radioButton_ball_check_enable.isChecked()
            if hasattr(self, 'radioButton_mark_check_enable'):
                self.local_params["mark_check_enable"] = self.radioButton_mark_check_enable.isChecked()
            if hasattr(self, 'radioButton_scratch_check_enable'):
                self.local_params["scratch_check_enable"] = self.radioButton_scratch_check_enable.isChecked()
            if hasattr(self, 'radioButton_shift_check_enable'):
                self.local_params["shift_check_enable"] = self.radioButton_shift_check_enable.isChecked()
            
            # ng_monitor 报警参数
            ng_monitor = {
                "monitor_enabled": self.checkBox_ng_monitor_enabled.isChecked(),
                "defect_limits": {},
                "min_yield_rate": 95.0,
                "min_yield_sample_size": 100,
            }
            defect_edit_map = [
                ("Mark", "lineEdit_defect_limit_Mark", 5),
                ("Size", "lineEdit_defect_limit_Size", 3),
                ("Ball_Area", "lineEdit_defect_limit_Area", 3),
                ("Ball Count", "lineEdit_defect_limit_BallCount", 3),
                ("Scratch", "lineEdit_defect_limit_Scratch", 3),
                ("Shift", "lineEdit_defect_limit_Shift", 3),
            ]
            for key, attr, default in defect_edit_map:
                try:
                    t = getattr(self, attr).text().strip()
                    ng_monitor["defect_limits"][key] = int(t) if t else default
                except ValueError:
                    ng_monitor["defect_limits"][key] = default
            try:
                t = self.lineEdit_min_yield_rate.text().strip()
                ng_monitor["min_yield_rate"] = float(t) if t else 95.0
            except ValueError:
                ng_monitor["min_yield_rate"] = 95.0
            try:
                t = self.lineEdit_min_yield_sample_size.text().strip()
                ng_monitor["min_yield_sample_size"] = int(t) if t else 100
            except ValueError:
                ng_monitor["min_yield_sample_size"] = 100
            self.local_params["ng_monitor"] = ng_monitor
            
            # ROI参数已经在create_roi中更新，这里不需要重复更新
            # 但确保它们存在
            if "roi_block" not in self.local_params:
                self.local_params["roi_block"] = []
            if "mark_rois" not in self.local_params:
                self.local_params["mark_rois"] = []
            if "mark_roi_min_areas" not in self.local_params:
                self.local_params["mark_roi_min_areas"] = []
            if "ball_search_roi" not in self.local_params:
                self.local_params["ball_search_roi"] = []
            if "scratch_roi" not in self.local_params:
                self.local_params["scratch_roi"] = []
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"更新参数失败: {str(e)}")
            import traceback
            traceback.print_exc()
   
    def create_roi(self, detect_type: str):
        """根据detect_type创建对应的ROI区域
        
        Args:
            detect_type: 检测类型
                - "ball" → 创建 ball_search_roi（单个列表 [x, y, w, h]）
                - "mark" → 向 mark_rois 追加 [x, y, w, h]（最多 4 个）
                - "scratch" → 创建 scratch_roi（单个列表 [x, y, w, h]）
                - "block" → 创建 roi_block（屏蔽区域，追加到列表）
        """
        # 检查模板图像是否已加载
        if self.template_image is None:
            QMessageBox.warning(self, "错误", "请先加载模板图像")
            return
        
        # 准备显示图像
        img_disp = ensure_bgr_u8(self.template_image, copy=True)
        
        try:
            # 根据detect_type设置窗口名称
            window_name_map = {
                "ball": "Ball Search ROI Selection",
                "mark": "Mark ROI Selection",
                "scratch": "Scratch ROI Selection",
                "block": "Block ROI Selection"
            }
            window_name = window_name_map.get(detect_type, "ROI Selection")
            
            # 创建窗口并设置为可调整大小模式
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            # 设置窗口大小为图像尺寸
            h, w = img_disp.shape[:2]
            cv.resizeWindow(window_name, w, h)
            
            # 使用selectROI让用户框选ROI
            roi = selectROI(window_name, img_disp, showCrosshair=True, fromCenter=False,
                           rect_color=(0, 255, 0), line_thickness=2)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"ROI选择失败: {str(e)}")
            return
        
        x, y, w, h = roi
        if w == 0 or h == 0:
            return
        
        # 根据detect_type保存到对应的参数位置
        if detect_type == "block":
            # 屏蔽区域：追加到列表
            if not isinstance(self.local_params.get("roi_block"), list):
                self.local_params["roi_block"] = []
            roi_tuple = (int(x), int(y), int(w), int(h))
            if roi_tuple not in self.local_params["roi_block"]:
                self.local_params["roi_block"].append(roi_tuple)
        elif detect_type == "ball":
            # 锡球搜索区域：单个列表
            self.local_params["ball_search_roi"] = [int(x), int(y), int(w), int(h)]
        elif detect_type == "mark":
            mrs = self.local_params.get("mark_rois") or []
            if not isinstance(mrs, list):
                mrs = []
            if len(mrs) >= 4:
                QMessageBox.warning(self, "提示", "Mark ROI 最多 4 个")
                return
            mrs.append([int(x), int(y), int(w), int(h)])
            self.local_params["mark_rois"] = mrs
            self._sync_mark_roi_min_areas_len()
        elif detect_type == "scratch":
            # 划痕检测区域：单个列表
            self.local_params["scratch_roi"] = [int(x), int(y), int(w), int(h)]
        else:
            QMessageBox.warning(self, "错误", f"未知的detect_type: {detect_type}")
            return
        
        # 更新显示（包含所有标记）
        self._update_template_display_with_markers()
    
    def clear_check_roi(self,detect_type):
        """清除检测区域"""
        # 根据detect_type映射到正确的参数名
        roi_key_map = {
            "mark": "mark_rois",
            "ball_search": "ball_search_roi",
            "scratch": "scratch_roi",
            "block": "roi_block"
        }
        roi_key = roi_key_map.get(detect_type)
        self.local_params[roi_key] = []
        if detect_type == "mark":
            self.local_params["mark_roi_min_areas"] = []

        # 更新显示（移除区域标记）
        self._update_template_display_with_markers()
    
    def confirm_load(self):
        """确认并保存参数到配置文件"""
        try:
            self._update_params_from_ui()
            if self.local_params.get("mark_check_enable") and len(
                self.local_params.get("mark_rois") or []
            ) < 1:
                QMessageBox.warning(
                    self,
                    "提示",
                    "已启用 Mark 检测，请至少框选 1 个 Mark ROI（最多 4 个）。",
                )
                return
            if len(self.local_params.get("mark_rois") or []) > 4:
                QMessageBox.warning(self, "提示", "Mark ROI 最多 4 个。")
                return
            mrs = self.local_params.get("mark_rois") or []
            if self.local_params.get("mark_check_enable") and len(mrs) >= 1:
                areas = self.local_params.get("mark_roi_min_areas") or []
                if len(areas) != len(mrs):
                    QMessageBox.warning(
                        self,
                        "提示",
                        "请在「检测区域管理」中为每个 Mark ROI 填写最小检出面积（像素）。",
                    )
                    return
                for i, a in enumerate(areas):
                    if not isinstance(a, int) or a < 0:
                        QMessageBox.warning(
                            self,
                            "提示",
                            f"ROI {i + 1} 的最小面积无效或未配置。",
                        )
                        return
            self.local_params.pop("min_mark_area", None)
            self.config_manager.set_section("work_dry_params", self.local_params)
            QMessageBox.information(self, "成功", "参数已成功保存")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存参数失败: {str(e)}")
            import traceback
            traceback.print_exc()
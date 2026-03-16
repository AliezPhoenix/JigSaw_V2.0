from threading import Thread
import ui.main_window_ui as main_window_ui
from PyQt5.QtWidgets import QMainWindow
from tools.MvImport.MvErrorDefine_const import MV_OK
import tools.hardware as HM
import src.config.config_manager as CM
from PyQt5.QtWidgets import QFileDialog,QMessageBox
import os
import cv2 as cv
import DryPramasSetDialog 
import TransferPramasSetDialog
from src.threads.thread_manager import ThreadManager
from PyQt5.QtGui import QPixmap, QImage, QBrush
from PyQt5.QtCore import QFile, Qt, QSettings, QTimer
from PyQt5.QtWidgets import QLabel, QApplication, QGraphicsScene, QTableWidgetItem, QHeaderView
import numpy as np
from src.support.support_funs import selectROI, execute_product_detection, draw_detection_results, fulltray_load_model, fulltray_predict_single_image, Bga_Strip
from src.support.InteractiveBgaLabel import InteractiveBgaLabel
from src.detectors.ball_detector import BallDetector
from src.detectors.size_detector import SizeDetector
from src.detectors.mark_detector import MarkDetector
from src.detectors.shift_detector import ShiftDetector
from src.detectors.scratch_detector import ScratchDetector
from src.detectors.template_detector import TemplateDetector
from ImageViewerWidget import ImageViewerWidget
from LogViewerWidget import LogViewerWidget
import traceback
import torch


class MainWindow(main_window_ui.Ui_MainWindow, QMainWindow):

    # =============================================================================
    # 1. 初始化
    # =============================================================================

    def __init__(self, progress_callback=None):
        super().__init__()
        _report = lambda p, m: progress_callback(p, m) if progress_callback else None

        _report(5, "加载界面...")
        self.setupUi(self)
        self.current_image = {
            "dry": None,
            "transfer": None,
            "sucker1": None,
            "sucker2": None,
            "fulltray": None
        }
        self.CAM_LIST=[
            {"alias": "dry_cam",        "port_ip": "192.168.1.200", "device_ip": "192.168.1.7"},
            {"alias": "transfer_cam",   "port_ip": "192.168.1.201", "device_ip": "192.168.1.8"},
            {"alias": "sucker1_cam",          "port_ip": "192.168.1.202", "device_ip": "192.168.1.9"},
            {"alias": "sucker2_cam",          "port_ip": "192.168.1.203", "device_ip": "192.168.1.10"},
            {"alias": "fulltray_cam",          "port_ip": "192.168.1.204", "device_ip": "192.168.1.6"}
        ] 

        self.MODBUS_INFO_LIST=[
            {"alias": "dry_modbus", "host_ip": "192.168.1.50", "port": 501},
            {"alias": "transfer_modbus", "host_ip": "192.168.1.50", "port": 502},
            {"alias": "sucker1_modbus", "host_ip": "192.168.1.50", "port": 503},
            {"alias": "sucker2_modbus", "host_ip": "192.168.1.50", "port": 504},
            {"alias": "fulltray_modbus", "host_ip": "192.168.1.50", "port": 505}
        ]
    
        _report(8, "初始化硬件与配置管理器...")
        self.hardware_manager = HM.Hardware_Manager(self.CAM_LIST)
        self.config_manager = CM.ConfigManager()
        self.modbus_manager = HM.ModBus_Manager(self.MODBUS_INFO_LIST)
        self.thread_manager:ThreadManager = None
        # 线程别名列表（所有线程共用同一个 hardware_manager、modbus_manager 和 config_manager）
        self.THREAD_INFO_LIST = [
            "dry_thread",
            "transfer_thread",
            "sucker_thread_1",
            "sucker_thread_2",
            "fulltray_thread"
        ]
        self.connection_status = {
        "camera":{
            "dry_cam":False,
            "transfer_cam":False,
            "sucker1_cam":False,
            "sucker2_cam":False,
            "fulltray_cam":False
        },
        "modbus":{
            "dry_modbus":False,
            "transfer_modbus":False,
            "sucker1_modbus":False,
            "sucker2_modbus":False,
            "fulltray_modbus":False
        }}
        # 初始化统计数据字典（按工位存储）
        self.statistics_data = {
            '干燥台': {
                'lot_id': '-',
                'total_count': 0,
                'ng_count': 0,
                'yield_rate': 0.0,
                'defect_counts': {'Mark': 0, 'Size': 0, 'Area': 0, 'Ball Count': 0, 'Scratch': 0, 'Shift': 0}
            },
            '移栽台': {
                'lot_id': '-',
                'total_count': 0,
                'ng_count': 0,
                'yield_rate': 0.0,
                'defect_counts': {'Mark': 0, 'Size': 0, 'Area': 0, 'Ball Count': 0, 'Scratch': 0, 'Shift': 0}
            }
        }
        self._devices_connect(progress_callback=_report)
        _report(50, "连接按钮信号...")
        self._all_button_connect()
        
        # 替换 label_mapping_dry 和 label_mapping_transfer 为 InteractiveBgaLabel
        _report(55, "初始化映射标签...")
        self._replace_mapping_label("dry")
        self._replace_mapping_label("transfer")

        # 创建日志查看和图像查看tab页面
        _report(65, "创建日志与图像查看器...")
        self.log_viewer_widget = LogViewerWidget(self)
        self.image_viewer_widget = ImageViewerWidget(self)

        # 添加到tabWidget（在现有tab之后）
        self.tabWidget.addTab(self.log_viewer_widget, "")
        self.tabWidget.addTab(self.image_viewer_widget, "")
        
        # 设置tab标题
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.log_viewer_widget), "日志查看")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.image_viewer_widget), "图像查看")
        
        # 延迟加载配方，确保窗口显示后 label 已获得正确尺寸，避免图像缩放异常
        QTimer.singleShot(3000, self._load_last_config)

        self.label_main_running_status.setText("待机")
        self.label_main_running_status.setStyleSheet(
            "color: orange ; font-weight: bold; font-size: 20pt; "
            "background-color: yellow; padding: 10px; border-radius: 5px;"
        )
        _report(85, "初始化统计表格...")
        self._init_statistics_table()
        _report(95, "启动完成")
        
    def _init_statistics_table(self):
        """初始化统计表格：设置表头、固定行标签、只读"""
        tbl = self.tableWidget_statistics
        tbl.setRowCount(9)
        tbl.setEditTriggers(tbl.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 表头
        tbl.setHorizontalHeaderLabels(["", "干燥台", "移栽台"])
        tbl.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        # 固定行标签（第0行为表头行，第1-9行为数据行）
        row_labels = ["总数：", "NG数：", "激光标记：", "尺寸：", "大小球(偏移)：", "缺球：", "划伤：", "偏移：", "良率："]
        for row, label in enumerate(row_labels):
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            tbl.setItem(row, 0, item)
        # 样式
        tbl.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                gridline-color: #555;
            }
            QHeaderView::section {
                background-color: #00bcd4;
                color: #000;
                font-weight: bold;
                padding: 5px;
            }
        """)
        self._update_statistics_display()

    def _init_all_tabs(self):
        tab_count = self.tabWidget.count()
        for i in range(tab_count):
            # 切换到每个tab页面
            self.tabWidget.setCurrentIndex(i)
            # 获取当前tab页面的widget
            current_widget = self.tabWidget.currentWidget()
            if current_widget:
                # 强制更新布局
                current_widget.updateGeometry()
                current_widget.adjustSize()
                # 更新整个tabWidget的布局
                self.tabWidget.updateGeometry()
                # 多次处理事件，确保布局计算完成
                for _ in range(10):
                    QApplication.processEvents()
                # 确保所有子widget的布局都已计算
                current_widget.update()
        
        # 切换回主界面（索引0）
        self.tabWidget.setCurrentIndex(0)
        # 再次处理事件，确保切换完成
        QApplication.processEvents()

    def _replace_mapping_label(self, work_position):
        """替换 label_mapping_dry 或 label_mapping_transfer 为 InteractiveBgaLabel"""
        attr_name = f"label_mapping_{work_position}"
        old_label = getattr(self, attr_name)
        parent = old_label.parent()
        layout = parent.layout()
        index = layout.indexOf(old_label)
        new_label = InteractiveBgaLabel(parent)
        new_label.setAlignment(Qt.AlignCenter)
        new_label.setObjectName(attr_name)
        new_label.regionClicked.connect(lambda pos, wp=work_position: self.on_bga_region_clicked(pos, wp))
        layout.takeAt(index).widget().deleteLater()
        layout.insertWidget(index, new_label)
        setattr(self, attr_name, new_label)

    # =============================================================================
    # 2. 配置管理
    # =============================================================================

    def _get_last_config_path(self):
        """从 QSettings 读取上次配方路径"""
        settings = QSettings("JigSaw", "JigSaw_v2")
        return settings.value("last_config_path", "", type=str)
    
    def _save_last_config_path(self, path):
        """将配方路径写入 QSettings"""
        if path:
            settings = QSettings("JigSaw", "JigSaw_v2")
            settings.setValue("last_config_path", path)
    
    def _load_last_config(self):
        """静默加载上次配方，若路径存在且文件有效则加载"""
        last_path = self._get_last_config_path()
        if not last_path or not os.path.exists(last_path):
            return
        try:
            ret, error_message = self.config_manager.load(last_path)
            if not ret:
                print(f"自动加载上次配方失败: {error_message}")
                return
            self.config_manager.get_section("work_dry_params")
            self._update_ui_from_config()
            self.label_13.setText(os.path.basename(last_path))
        except Exception as e:
            print(f"自动加载上次配方异常: {e}")
            traceback.print_exc()
        self.pushButton_start.setEnabled(True)
    def _load_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "./config", "配置文件 (*.json)")
        if not file_path:
            QMessageBox.warning(self, "警告", "未选择配置文件")
            return

        ret,error_message = self.config_manager.load(file_path)
        if not ret:
            QMessageBox.warning(self, "警告", f"加载配置文件失败: {error_message}❌")
            return
        self._save_last_config_path(file_path)
        self.label_13.setText(os.path.basename(file_path))
        self._update_ui_from_config()
        if self.thread_manager is not None:
            success_list = []
            for sub_station in ["dry_thread", "transfer_thread", "sucker_thread_1", "sucker_thread_2", "fulltray_thread"]:
                success_list.append(self.thread_manager.update_params(sub_station))
        QMessageBox.information(self, "提示", "配置文件加载成功✅")
        self.pushButton_start.setEnabled(True)
        
    def _save_config_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择配置文件", "./config", "配置文件 (*.json)")
        if not file_path:
            QMessageBox.warning(self, "警告", "未选择配置文件❌")
            return
        self._update_config_from_ui()
        ret,error_message = self.config_manager.save(file_path)
        if not ret:
            QMessageBox.warning(self, "警告", f"保存配置文件失败: {error_message}❌")
            return
        self._save_last_config_path(file_path)
        self.label_13.setText(os.path.basename(file_path))
        QMessageBox.information(self, "提示", "配置文件保存成功✅")

    def _update_config_from_ui(self):
        """从 UI 读取用户输入参数并保存到 config_manager"""
        try:
            size_x = float(self.lineEdit_product_size_x_mm.text())
            size_y = float(self.lineEdit_product_size_y_mm.text())
            total_rows = int(self.lineEdit_row_nums.text())
            total_cols = int(self.lineEdit_col_nums.text())
            product_count = int(self.lineEdit_product_count.text())
            product_type = "BGA" if self.radioButton_BGA_type.isChecked() else "QFN"

            for section in ["work_dry_params", "work_transfer_params"]:
                self.config_manager.set_key(section, "product_size", [size_x, size_y])
                self.config_manager.set_key(section, "product_type", product_type)
                self.config_manager.set_key(section, "total_rows", total_rows)
                self.config_manager.set_key(section, "total_cols", total_cols)
                self.config_manager.set_key(section, "product_count", product_count)

            self.config_manager.set_key("work_dry_params", "template_threshold", float(self.lineEdit_mathch_threshsold_dry.text()))
            self.config_manager.set_key("work_dry_params", "current_col", int(self.lineEdit_current_col_dry.text()))
            self.config_manager.set_key("work_dry_params", "current_row", int(self.lineEdit_current_row_dry.text()))
            self.config_manager.set_key("work_dry_params", "pixel_size", float(self.lineEdit_pixel_size_dry.text()))

            self.config_manager.set_key("work_transfer_params", "template_threshold", float(self.lineEdit_mathch_threshsold_transfer.text()))
            self.config_manager.set_key("work_transfer_params", "current_col", int(self.lineEdit_current_col_transfer.text()))
            self.config_manager.set_key("work_transfer_params", "current_row", int(self.lineEdit_current_row_transfer.text()))
            self.config_manager.set_key("work_transfer_params", "pixel_size", float(self.lineEdit_pixel_size_transfer.text()))

            self.config_manager.set_key("work_fulltray_params", "rows", int(self.lineEdit_fulltray_row_nums.text()))
            self.config_manager.set_key("work_fulltray_params", "cols", int(self.lineEdit_fulltray_col_nums.text()))
            self.config_manager.set_key("work_fulltray_params", "model_path", self.lineEdit_fulltray_model_path.text().strip())
        except (ValueError, AttributeError) as e:
            print(f"_update_config_from_ui 解析错误: {e}")
            raise

    def _update_ui_from_config(self):

        #————————————————————主界面参数上传——————————————————————————————————————————————————
        self.lineEdit_product_size_x_mm.setText(str(self.config_manager.get_key("work_dry_params","product_size")[0]))
        self.lineEdit_product_size_y_mm.setText(str(self.config_manager.get_key("work_dry_params","product_size")[1]))
        if self.config_manager.get_key("work_dry_params","product_type") == "BGA":
            self.radioButton_BGA_type.setChecked(True)
        elif self.config_manager.get_key("work_dry_params","product_type") == "QFN":
            self.radioButton_QFN_type.setChecked(True)
        self.lineEdit_row_nums.setText(str(self.config_manager.get_key("work_dry_params","total_rows")))
        self.lineEdit_col_nums.setText(str(self.config_manager.get_key("work_dry_params","total_cols")))
        self.lineEdit_product_count.setText(str(self.config_manager.get_key("work_dry_params","product_count")))
        #————————————————————————干燥台tab页面参数上传——————————————————————————————————

        self.lineEdit_mathch_threshsold_dry.setText(str(self.config_manager.get_key("work_dry_params","template_threshold")))
        self.lineEdit_current_col_dry.setText(str(self.config_manager.get_key("work_dry_params","current_col")))
        self.lineEdit_current_row_dry.setText(str(self.config_manager.get_key("work_dry_params","current_row")))
        self.lineEdit_pixel_size_dry.setText(str(self.config_manager.get_key("work_dry_params","pixel_size")))
        template_path_dry = self.config_manager.get_key("work_dry_params","golden_template_path")
        template_dry = cv.imread(template_path_dry)
        if template_dry is None:
            QMessageBox.warning(self, "错误", "无干燥台模板文件❌")
        else:
            self._update_label_from_image(self.label_template_display_dry,template_dry)
        

        #——————————————————————————移栽台tab页面参数上传————————————————————————————————
        self.lineEdit_current_col_transfer.setText(str(self.config_manager.get_key("work_transfer_params","current_col")))
        self.lineEdit_current_row_transfer.setText(str(self.config_manager.get_key("work_transfer_params","current_row")))
        self.lineEdit_mathch_threshsold_transfer.setText(str(self.config_manager.get_key("work_transfer_params","template_threshold")))
        self.lineEdit_pixel_size_transfer.setText(str(self.config_manager.get_key("work_transfer_params","pixel_size")))
        template_path_transfer = self.config_manager.get_key("work_transfer_params","golden_template_path")
        template_transfer = cv.imread(template_path_transfer)
        if template_transfer is None:
            QMessageBox.warning(self, "错误", "无移栽台模板文件❌")
        else:
            self._update_label_from_image(self.label_template_display_transfer,template_transfer)


        #——————————————————————————满盘tab页面参数上传
        self.lineEdit_fulltray_row_nums.setText(str(self.config_manager.get_key("work_fulltray_params","rows")))
        self.lineEdit_fulltray_col_nums.setText(str(self.config_manager.get_key("work_fulltray_params","cols")))
        model_path = self.config_manager.get_key("work_fulltray_params","model_path")
        model_name = os.path.basename(model_path)
        self.lineEdit_fulltray_model_path.setText(model_path)
        self.label_fulltray_current_model.setText(model_name)
        self.label_fulltray_current_model.setStyleSheet("color: green;")

        #——————————————————————————加载配置时显示初始 BGA mapping
        try:
            dry_params = self.config_manager.get_section("work_dry_params")
            dry_total_rows = dry_params.get("total_rows", 0)
            dry_total_cols = dry_params.get("total_cols", 0)
            if dry_total_rows > 0 and dry_total_cols > 0:
                bga_dry_front = Bga_Strip(station="dry",strip_side="front", strip_lot="", strip_sn="", strip_create_time="", params=dry_params)
                self.update_bga_display(bga_dry_front, "front", "dry")
            transfer_params = self.config_manager.get_section("work_transfer_params")
            transfer_total_rows = transfer_params.get("total_rows", 0)
            transfer_total_cols = transfer_params.get("total_cols", 0)
            if transfer_total_rows > 0 and transfer_total_cols > 0:
                bga_transfer_front = Bga_Strip(station="transfer",strip_side="front", strip_lot="", strip_sn="", strip_create_time="", params=transfer_params)
                self.update_bga_display(bga_transfer_front, "front", "transfer")
        except Exception as e:
            print(f"显示基础mapping图像失败: {str(e)}")
            traceback.print_exc()

    def _update_config_from_signal(self, config_name):
        """从信号更新配置"""
        config_path = os.path.join("./config", config_name + ".json")
        ret,error_message = self.config_manager.load(config_path)
        if not ret:
            QMessageBox.warning(self, "警告", f"加载配置文件失败: {error_message}❌")
            return
        self._update_ui_from_config()
        self.confirm_params("all", silent=True)
    # =============================================================================
    # 3. 信号与连接
    # =============================================================================

    def _all_button_connect(self):
        self.pushButton_start.clicked.connect(self.start_thread)
        self.pushButton_connect.clicked.connect(self._devices_connect)
        self.pushButton_stop.clicked.connect(self.stop)
        self.pushButton_read_cam_params_dry.clicked.connect(lambda: self._operate_hardware("read_cam_params","dry_cam"))
        self.pushButton_transfer_read_camera_param.clicked.connect(lambda: self._operate_hardware("read_cam_params","transfer_cam"))
        self.pushButton_sucker1_read_camera_param.clicked.connect(lambda: self._operate_hardware("read_cam_params","sucker1_cam"))
        self.pushButton_sucker2_read_camera_param.clicked.connect(lambda: self._operate_hardware("read_cam_params","sucker2_cam"))

        self.pushButton_write_cam_params_dry.clicked.connect(lambda: self._operate_hardware("write_cam_params","dry_cam"))
        self.pushButton_transfer_cam_params_save.clicked.connect(lambda: self._operate_hardware("write_cam_params","transfer_cam"))
        self.pushButton_sucker1_cam_params_save.clicked.connect(lambda: self._operate_hardware("write_cam_params","sucker1_cam"))
        self.pushButton_sucker2_cam_params_save.clicked.connect(lambda: self._operate_hardware("write_cam_params","sucker2_cam"))
        self.pushButton_capture_one_frame_dry.clicked.connect(lambda: self._operate_hardware("capture_one_frame","dry_cam"))
        self.pushButton_capture_one_frame_transfer.clicked.connect(lambda: self._operate_hardware("capture_one_frame","transfer_cam"))
        self.pushButton_capture_one_frame_sucker1.clicked.connect(lambda: self._operate_hardware("capture_one_frame","sucker1_cam"))
        self.pushButton_capture_one_frame_sucker2.clicked.connect(lambda: self._operate_hardware("capture_one_frame","sucker2_cam"))
        self.pushButton_capture_one_frame_fulltray.clicked.connect(lambda: self._operate_hardware("capture_one_frame","fulltray_cam"))

        self.pushButton_template_choose.clicked.connect(lambda: self._load_template("dry"))
        self.pushButton_transfer_template_choose.clicked.connect(lambda: self._load_template("transfer"))


        self.pushButton_product_prams_load.clicked.connect(self._load_config_file)
        self.pushButton_product_prams_save.clicked.connect(self._save_config_file)

        self.pushButton_confirm_params_all.clicked.connect(lambda: self.confirm_params("all"))
        self.pushButton_confirm_params_dry.clicked.connect(lambda: self.confirm_params("dry_thread"))
        self.pushButton_confirm_params_transfer.clicked.connect(lambda: self.confirm_params("transfer_thread"))
        self.pushButton_confirm_params_fulltray.clicked.connect(lambda: self.confirm_params("fulltray_thread"))

        self.PushButton_test_tempalte.clicked.connect(lambda:self.show_pramas_set_dialog("dry"))
        self.PushButton_transfer_test_template.clicked.connect(lambda:self.show_pramas_set_dialog("transfer"))

        self.btn_create_new_template.clicked.connect(lambda: self.create_new_tmepalte("dry"))
        self.btn_create_new_template_transfer.clicked.connect(lambda: self.create_new_tmepalte("transfer"))
        self.pushButton_tempaltematch.clicked.connect(lambda: self.template_validity_test("dry"))
        self.pushButton_transfer_template_match.clicked.connect(lambda: self.template_validity_test("transfer"))
        self.pushButton_current_image_select_dry.clicked.connect(lambda: self.load_current_image("dry"))
        self.pushButton_current_image_select_transfer.clicked.connect(lambda: self.load_current_image("transfer"))
        self.pushButton_create_checkable_roi_dry.clicked.connect(lambda: self.create_search_roi("dry"))
        self.pushButton_create_checkable_roi_transfer.clicked.connect(lambda: self.create_search_roi("transfer"))
        
        self.pushButton_fulltray_select_model.clicked.connect(self.select_fulltray_model)
        self.pushButton_fulltray_set_roi.clicked.connect(lambda: self.create_search_roi("fulltray"))
        self.pushButton_fulltray_select_image.clicked.connect(lambda: self.load_current_image("fulltray"))
        self.pushButton_fulltray_test.clicked.connect(self.manual_test_fulltray)
        self.radioButton_live_sucker1.toggled.connect(self._on_sucker1_live_toggled)
        self.radioButton_live_sucker2.toggled.connect(self._on_sucker2_live_toggled)


        self.pushButton_start.setEnabled(False)
        self.pushButton_stop.setEnabled(False)
        # self.pushButton_product_prams_load.clicked.connect(self._load_config_file())
    
    def _on_sucker1_live_toggled(self, checked):
        """主线程槽：更新 SuckerThread1 的实时显示标志（避免工作线程访问 UI 死锁）"""
        s1 = self.thread_manager.get_thread_obj("sucker_thread_1") if self.thread_manager else None
        if s1:
            s1.set_live_display_enabled(checked)

    def _on_sucker2_live_toggled(self, checked):
        """主线程槽：更新 SuckerThread2 的实时显示标志（避免工作线程访问 UI 死锁）"""
        s2 = self.thread_manager.get_thread_obj("sucker_thread_2") if self.thread_manager else None
        if s2:
            s2.set_live_display_enabled(checked)

    def _all_signal_connect(self):
        """连接各线程信号到 _update_display 及统计/消息占位"""
        t = getattr(self, "thread_manager", None)
        if t is None:
            return
        # fulltray_thread: 发射 np.ndarray，由 _update_display 统一处理；结果由 _update_fulltray_result 处理
        ft = t.get_thread_obj("fulltray_thread")
        if ft:
            ft._update_image_signal.connect(lambda img: self._update_display("fulltray", img, None))
            ft._update_result_signal.connect(self._update_fulltray_result)
            ft._update_config_changed_signal.connect(self._update_config_from_signal)
        # dry_thread: 发射 (np.ndarray, Bga_Strip|None)，第二参数为 None 表示实时图
        dt = t.get_thread_obj("dry_thread")
        if dt:
            dt._update_image_signal.connect(lambda img, bga: self._update_display("dry", img, bga))
            dt._update_statistics_signal.connect(lambda stats: self._update_statistics("dry", stats))
            dt._update_message_signal.connect(lambda msg: self._update_message("dry", msg))
        # transfer_thread
        tt = t.get_thread_obj("transfer_thread")
        if tt:
            tt._update_image_signal.connect(lambda img, bga: self._update_display("transfer", img, bga))
            tt._update_statistics_signal.connect(lambda stats: self._update_statistics("transfer", stats))
            tt._update_message_signal.connect(lambda msg: self._update_message("transfer", msg))
        # sucker_thread_1
        s1 = t.get_thread_obj("sucker_thread_1")
        if s1:
            s1._update_image_signal.connect(lambda img, bga: self._update_display("sucker_1", img, bga))
            s1._update_statistics_signal.connect(lambda stats: self._update_statistics("sucker_1", stats))
            s1._update_message_signal.connect(lambda msg: self._update_message("sucker_1", msg))
            s1.set_live_display_enabled(self.radioButton_live_sucker1.isChecked())
        # sucker_thread_2
        s2 = t.get_thread_obj("sucker_thread_2")
        if s2:
            s2._update_image_signal.connect(lambda img, bga: self._update_display("sucker_2", img, bga))
            s2._update_statistics_signal.connect(lambda stats: self._update_statistics("sucker_2", stats))
            s2._update_message_signal.connect(lambda msg: self._update_message("sucker_2", msg))
            s2.set_live_display_enabled(self.radioButton_live_sucker2.isChecked())

    # =============================================================================
    # 4. 硬件与设备
    # =============================================================================

    def _devices_connect(self, progress_callback=None):
        total = len(self.CAM_LIST) + len(self.MODBUS_INFO_LIST)
        step = 40 / max(1, total)  # 10% -> 50%
        idx = 0

        for each_cam in self.CAM_LIST:
            each_cam_alias = each_cam['alias']
            if progress_callback:
                progress_callback(min(50, int(10 + step * (idx + 1))), f"连接相机 {each_cam_alias}...")
            if self.connection_status["camera"].get(each_cam_alias) is not None and self.connection_status["camera"].get(each_cam_alias):
                idx += 1
                continue
            success,msg,_ = self.hardware_manager.connect(each_cam_alias)
            if not success:
                print( "警告", f"连接{each_cam_alias}失败: {msg}")
            else:
                self.connection_status["camera"][each_cam_alias] = True
            idx += 1

        for each_modbut in self.MODBUS_INFO_LIST:
            each_modbut_alias = each_modbut['alias']
            if progress_callback:
                progress_callback(min(50, int(10 + step * (idx + 1))), f"连接 Modbus {each_modbut_alias}...")
            if self.connection_status["modbus"].get(each_modbut_alias) is not None and self.connection_status["modbus"].get(each_modbut_alias):
                idx += 1
                continue
            success,msg = self.modbus_manager.connect(each_modbut_alias)
            if not success:
                print( "警告", f"连接{each_modbut_alias}失败: {msg}")
                return
            else:
                self.connection_status["modbus"][each_modbut_alias] = True
            idx += 1

        self._update_connection_status()

    def _update_connection_status(self):
        status_lable = {
            'modbus': self.label_status_plc,
            'dry_cam': self.label_status_cam1,
            'transfer_cam': self.label_status_cam2,
            'sucker1_cam': self.label_status_cam3_1,
            'sucker2_cam': self.label_status_cam3_2,
            'fulltray_cam': self.label_status_fulltray
        }

        # 检查所有modbus状态是否都为True
        all_modbus_connected = all(self.connection_status["modbus"].values())
        if all_modbus_connected:
            # 所有modbus已连接，设置为绿色
            status_lable['modbus'].setStyleSheet("color: green;")
        else:
            # 有modbus未连接，设置为红色
            status_lable['modbus'].setStyleSheet("color: red;")
        
        # 更新每个相机的label状态
        for cam_alias in self.connection_status["camera"].keys():
            if cam_alias in status_lable:
                cam_status = self.connection_status["camera"].get(cam_alias, False)
                if cam_status:
                    # 相机已连接，设置为绿色
                    status_lable[cam_alias].setStyleSheet("color: green;")
                else:
                    # 相机未连接，设置为红色
                    status_lable[cam_alias].setStyleSheet("color: red;")

    def _operate_hardware(self,action,hardware_alias:str):
        if action == "connect_all":
            for each_handle in self.hardware_manager.get_all_cameras():
                success,msg,_ = self.hardware_manager.connect(each_handle)
                if not success:
                    QMessageBox.warning(self, "警告", f"连接{each_handle}失败: {msg}")

        if action == "close_all":
            for each_handle in self.hardware_manager.get_all_cameras():
                success,msg,_ = self.hardware_manager.close_camera(each_handle)
                if not success:
                    QMessageBox.warning(self, "警告", f"关闭{each_handle}失败: {msg}")

        if action == "connect_single":
            success,msg,_  = self.hardware_manager.connect(hardware_alias)
            if not success:
                QMessageBox.warning(self, "警告", f"连接{hardware_alias}失败: {msg}")

        if action == "close_single":
            success,msg,_ = self.hardware_manager.close_camera(hardware_alias)
            if not success:
                QMessageBox.warning(self, "警告", f"关闭{hardware_alias}失败: {msg}")

        if action == "read_cam_params":
            try:
                param_names = ["ExposureTime", "Gain", "Gamma", "AcquisitionFrameRate"]
                params = {}
                param_errors = []
                
                # 批量获取参数
                for param_name in param_names:
                    success, msg, value = self.hardware_manager.get_parameter(hardware_alias, param_name)
                    if not success:
                        param_errors.append(f"{param_name}: {msg}")
                    else:
                        params[param_name] = value
                
                # 如果有参数获取失败，显示错误并返回
                if param_errors:
                    error_msg = "\n".join(param_errors)
                    QMessageBox.warning(self, "警告", f"读取{hardware_alias}相机参数失败：{error_msg}")
                    return
                
                # 更新UI
                prefix = hardware_alias.split('_')[0]
                lineEdit_exposure = getattr(self, f"lineEdit_ExposureTime_{prefix}")
                lineEdit_gain = getattr(self, f"lineEdit_Gain_{prefix}")
                lineEdit_gamma = getattr(self, f"lineEdit_Gamma_{prefix}")
                lineEdit_framerate = getattr(self, f"lineEdit_AcquisitionFrameRate_{prefix}")
                
                lineEdit_exposure.setText(str(round(params["ExposureTime"], 2)))
                lineEdit_gain.setText(str(round(params["Gain"], 2)))
                lineEdit_gamma.setText(str(round(params["Gamma"], 2)))
                lineEdit_framerate.setText(str(round(params["AcquisitionFrameRate"], 2)))
                
                QMessageBox.information(self, "成功", f"{hardware_alias}相机参数读取成功 ✅")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"读取{hardware_alias}相机参数失败：{str(e)}")

        if action == "write_cam_params":
            try:
                params = {
                    "ExposureTime": float(getattr(self,f"lineEdit_ExposureTime_{hardware_alias.split('_')[0]}").text()),
                    "Gain": float(getattr(self,f"lineEdit_Gain_{hardware_alias.split('_')[0]}").text()),
                    "Gamma": float(getattr(self,f"lineEdit_Gamma_{hardware_alias.split('_')[0]}").text()),
                    "AcquisitionFrameRate": float(getattr(self,f"lineEdit_AcquisitionFrameRate_{hardware_alias.split('_')[0]}").text())
                }
                param_errors = []
                
                # 批量设置参数
                for param_name, param_value in params.items():
                    success, msg, _ = self.hardware_manager.set_parameter(hardware_alias, param_name, param_value)
                    if not success:
                        param_errors.append(f"{param_name}: {msg}")
                        
                ## 如果有参数获取失败，显示错误并返回
                if param_errors:
                    error_msg = "\n".join(param_errors)
                    QMessageBox.warning(self, "警告", f"设置{hardware_alias}相机参数失败：\n{error_msg}")
                    return


                success, msg, _ = self.hardware_manager.save_to_userSet(hardware_alias)
                if not success:
                    QMessageBox.warning(self, "警告", f"保存{hardware_alias}相机参数到用户集失败：{msg}")
                    return
            except Exception as e:
                QMessageBox.warning(self, "错误", f"设置{hardware_alias}相机参数失败：{str(e)}")

        if action == "capture one frame":
            success, msg, image = self.hardware_manager.capture_image(hardware_alias)
            if not success:
                QMessageBox.warning(self, "警告", f"拍照{hardware_alias}失败: {msg}")
            else:
                label = getattr(self, f"label_current_cam_live_{hardware_alias.split('_')[0]}")
                self._update_label_from_image(label, image)
                self.current_image[hardware_alias.split('_')[0]] = image
                QApplication.processEvents()
    # =============================================================================
    # 5. 线程控制
    # =============================================================================

    def start_thread(self):
        """
        启动线程，检查对应的modbus和相机连接状态
        只有当对应的modbus和相机都连接成功时才启动线程
        """
        # 线程到硬件设备的映射关系
        thread_device_mapping = {
            "dry_thread": {
                "modbus": "dry_modbus",
                "camera": "dry_cam",
                "name": "干燥台"
            },
            "transfer_thread": {
                "modbus": "transfer_modbus",
                "camera": "transfer_cam",
                "name": "移栽台"
            },
            "sucker_thread_1": {
                "modbus": "sucker1_modbus",
                "camera": "sucker1_cam",
                "name": "吸嘴组1"
            },
            "sucker_thread_2": {
                "modbus": "sucker2_modbus",
                "camera": "sucker2_cam",
                "name": "吸嘴组2"
            },
            "fulltray_thread": {
                "modbus": "fulltray_modbus",
                "camera": "fulltray_cam",
                "name": "满盘"
            }
        }
        # 若尚无 ThreadManager 则创建并连接信号；已存在则复用，避免重复创建和重复连接
        if self.thread_manager is None:
            self.thread_manager = ThreadManager(
                thread_list=self.THREAD_INFO_LIST,
                hardware_manager=self.hardware_manager,
                modbus_manager=self.modbus_manager,
                config_manager=self.config_manager,
                ui=self
            )
            self._all_signal_connect()
        success_count = 0
        failed_threads = []
        
        for thread_alias in self.THREAD_INFO_LIST:
            if thread_alias not in thread_device_mapping:
                print(f"警告: 未找到线程 {thread_alias} 的设备映射")
                continue
            
            device_info = thread_device_mapping[thread_alias]
            modbus_alias = device_info["modbus"]
            camera_alias = device_info["camera"]
            thread_name = device_info["name"]
            
            # 检查连接状态
            modbus_status = self.connection_status["modbus"].get(modbus_alias, False)
            camera_status = self.connection_status["camera"].get(camera_alias, False)
            
            # 验证连接状态
            if not (modbus_status and camera_status):
                failed_reasons = []
                if not modbus_status:
                    failed_reasons.append(f"ModBus({modbus_alias})未连接")
                if not camera_status:
                    failed_reasons.append(f"相机({camera_alias})未连接")
                failed_threads.append(f"{thread_name}: {', '.join(failed_reasons)}")
                continue
            
            # 启动线程
            thread_obj = self.thread_manager.get_thread_obj(thread_alias)
            if not thread_obj:
                failed_threads.append(f"{thread_name}: 线程对象不存在")
                continue
            
            if thread_obj.isRunning():
                print(f"线程已运行: {thread_name} ({thread_alias})")
                continue
            
            thread_obj.start()
            print(f"启动线程: {thread_name} ({thread_alias})")
            success_count += 1
        
        # 打印结果摘要
        if success_count > 0:
            print(f"成功启动 {success_count} 个线程")
            self.label_main_running_status.setText("运行中")
            self.label_main_running_status.setStyleSheet(
                "color: white ; font-weight: bold; font-size: 20pt; "
                "background-color: green;"
            )
        if failed_threads:
            print("以下线程启动失败:")
            for failed_msg in failed_threads:
                print(f"  - {failed_msg}")
        elif success_count == 0:
            print("警告: 没有可启动的线程")
    
        self.pushButton_start.setEnabled(False)
        self.pushButton_stop.setEnabled(True)
    def stop(self):
        """停止所有线程"""
        if self.thread_manager is None:
            return
        self.thread_manager.stop_all_threads()
        print("所有线程已停止")
        self.label_main_running_status.setText("待机")
        self.label_main_running_status.setStyleSheet(
            "color: orange ; font-weight: bold; font-size: 20pt; "
            "background-color: yellow; padding: 10px; border-radius: 5px;"
        )
        self.pushButton_start.setEnabled(True)
        self.pushButton_stop.setEnabled(False)

    def _pause_thread(self):
        pass

    def _resume_thread(self):
        pass

    # =============================================================================
    # 6. 显示更新
    # =============================================================================

    def _update_display(self, station, image, bga_strip:Bga_Strip=None):
        """各工位图像显示，统一入口"""
        if station == "fulltray":
            if image is not None and isinstance(image, np.ndarray):
                try:
                    self._update_label_from_image(self.label_image_show_fulltray, image)
                    h, w = image.shape[:2]
                    bytes_per_line = 3 * w
                    q_image = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
                    pixmap = QPixmap.fromImage(q_image)
                    scene = QGraphicsScene()
                    scene.addPixmap(pixmap)
                    self.graphicsView_fulltray_cam_live.setScene(scene)
                    self.graphicsView_fulltray_cam_live.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
                except Exception as e:
                    print(f"满盘显示错误: {e}")
        elif station == "dry":
            if image is not None and isinstance(image, np.ndarray):
                try:
                    if bga_strip is None:
                        self._update_label_from_image(self.label_current_cam_live_dry, image)
                        self._update_label_from_image(self.label_image_show_dry, image)
                    else:
                        self._update_label_from_image(self.label_image_show_dry, image)
                        self._update_label_from_image(self.label_current_cam_live_dry, image)
                        self.update_bga_display(bga_strip, bga_strip.side, "dry")
                except Exception as e:
                    print(f"干燥台显示错误: {e}")
        elif station == "transfer":
            if image is not None and isinstance(image, np.ndarray):
                try:
                    if bga_strip is None:
                        self._update_label_from_image(self.label_current_cam_live_transfer, image)
                        self._update_label_from_image(self.label_image_show_transfer, image)
                    else:
                        self._update_label_from_image(self.label_image_show_transfer, image)
                        self._update_label_from_image(self.label_current_cam_live_transfer, image)
                        self.update_bga_display(bga_strip, bga_strip.side, "transfer")
                except Exception as e:
                    print(f"移栽台显示错误: {e}")
        elif station == "sucker_1":
            if image is not None and isinstance(image, np.ndarray):
                try:
                    self._update_label_from_image(self.label_sucker1_cam_live, image)
                except Exception as e:
                    print(f"吸嘴1显示错误: {e}")
        elif station == "sucker_2":
            if image is not None and isinstance(image, np.ndarray):
                try:
                    self._update_label_from_image(self.label_sucker2_cam_live, image)
                except Exception as e:
                    print(f"吸嘴2显示错误: {e}")

    def _update_label_from_image(self,label:QLabel,image:np.ndarray):
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        # 将numpy数组转换为bytes（QImage需要bytes类型，不能直接使用memoryview）
        q_image = QImage(image.tobytes(), width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        # 获取 label 的大小，保持宽高比缩放
        label_size = label.size()
        scaled_pixmap = pixmap.scaled(label_size.width(), label_size.height(), 
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setScaledContents(False)
        label.setPixmap(scaled_pixmap)

    def _update_fulltray_result(self, is_ok, product_count=0, total_cells=0, empty_count=0, avg_confidence=0.0):
        """更新满盘检测结果显示"""
        if hasattr(self, 'label_fulltray_result'):
            if is_ok:
                result_text = f"OK\n{product_count}/{total_cells}"
                self.label_fulltray_result.setText(result_text)
                self.label_fulltray_result.setStyleSheet(
                    "color: white; font-weight: bold; font-size: 50pt; "
                    "background-color: green; padding: 10px; border-radius: 5px;"
                )
            else:
                result_text = f"NG\n{product_count}/{total_cells}"
                self.label_fulltray_result.setText(result_text)
                self.label_fulltray_result.setStyleSheet(
                    "color: white; font-weight: bold; font-size: 50pt; "
                    "background-color: red; padding: 10px; border-radius: 5px;"
                )

    def _update_statistics_display(self):
        """根据 statistics_data 刷新表格显示"""
        dry = self.statistics_data['干燥台']
        transfer = self.statistics_data['移栽台']
        current_lot_id = dry['lot_id'] if dry['lot_id'] not in ('-', '') else transfer['lot_id']
        self.label_statistics_lot_id.setText(f"lot_id: {current_lot_id}")
        tbl = self.tableWidget_statistics
        # 行0: 总数, 1: NG数, 2: Mark, 3: Size, 4: Area, 5: Ball Count, 6: Scratch, 7: Shift, 8: 良率
        rows_data = [
            (dry['total_count'], transfer['total_count']),
            (dry['ng_count'], transfer['ng_count']),
            (dry['defect_counts'].get('Mark', 0), transfer['defect_counts'].get('Mark', 0)),
            (dry['defect_counts'].get('Size', 0), transfer['defect_counts'].get('Size', 0)),
            (dry['defect_counts'].get('Area', 0), transfer['defect_counts'].get('Area', 0)),
            (dry['defect_counts'].get('Ball Count', 0), transfer['defect_counts'].get('Ball Count', 0)),
            (dry['defect_counts'].get('Scratch', 0), transfer['defect_counts'].get('Scratch', 0)),
            (dry['defect_counts'].get('Shift', 0), transfer['defect_counts'].get('Shift', 0)),
            (f"{dry['yield_rate']:.2f}%", f"{transfer['yield_rate']:.2f}%"),
        ]
        for row, (dry_val, trans_val) in enumerate(rows_data):
            for col, val in enumerate([dry_val, trans_val], start=1):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if row == 1 and isinstance(val, int) and val > 0:  # NG数
                    item.setForeground(QBrush(Qt.red))
                elif row == 8:  # 良率
                    item.setForeground(QBrush(Qt.green))
                tbl.setItem(row, col, item)

    def _update_statistics(self, station: str, stats_info: dict):
        """根据线程上报的 stats_info 更新 statistics_data 并刷新表格"""
        if stats_info is None:
            return
        station_name = stats_info.get('station', '')
        if station_name not in self.statistics_data:
            return
        self.statistics_data[station_name]['lot_id'] = stats_info.get('lot_id', '-') or '-'
        self.statistics_data[station_name]['total_count'] = stats_info.get('total_count', 0)
        self.statistics_data[station_name]['ng_count'] = stats_info.get('ng_count', 0)
        self.statistics_data[station_name]['yield_rate'] = stats_info.get('yield_rate', 0.0)
        dc = stats_info.get('defect_counts', {})
        self.statistics_data[station_name]['defect_counts'] = dict(dc) if dc else {}
        self._update_statistics_display()

    def _update_message(self, station: str, msg: str):
        pass

    # =============================================================================
    # 7. BGA 显示
    # =============================================================================

    def update_bga_display(self, bga_instance, side, work_position="dry"):
        """
        更新BGA显示（在主线程中调用）
        参数:
            bga_instance: Bga_Strip 实例
            side: "front" 或 "back"
            work_position: "dry" 或 "transfer"，用于确定更新哪个 label
        """
        try:
            if work_position == "dry":
                label = self.label_mapping_dry
                label_mapping_show = self.label_image_show_mapping_1
            else:
                label = self.label_mapping_transfer
                label_mapping_show = self.label_image_show_mapping_2

            label.set_bga_data(bga_instance)
            animation = bga_instance.get_full_animation()
            self._update_label_from_image(label_mapping_show, animation)
        except Exception as e:
            print(f"更新BGA显示失败 ({work_position}): {str(e)}")
            traceback.print_exc()

    def on_bga_region_clicked(self, pos_start, work_position="dry"):
        """BGA区域点击回调，显示对应区域的图像"""
        try:
            if work_position == "dry":
                bga_instance = getattr(self.label_mapping_dry, 'bga', None)
            else:
                bga_instance = getattr(self.label_mapping_transfer, 'bga', None)
            if bga_instance is None:
                return
            current_image = bga_instance.get_pos_image(pos_start)
            if current_image is not None:
                label_name = getattr(self, f"label_current_cam_live_{work_position}")
                if current_image.dtype != np.uint8:
                    if np.issubdtype(current_image.dtype, np.floating) and current_image.max() <= 1.0:
                        current_image = (np.clip(current_image, 0, 1) * 255).astype(np.uint8)
                    else:
                        current_image = np.clip(current_image, 0, 255).astype(np.uint8)
                if len(current_image.shape) == 2:
                    current_image = cv.cvtColor(current_image, cv.COLOR_GRAY2BGR)
                self._update_label_from_image(label_name, current_image)
                self.current_image[work_position] = current_image
                self.template_validity_test(work_position)
        except Exception as e:
            print(f"BGA区域点击显示错误 ({work_position}): {str(e)}")
            traceback.print_exc()

    # =============================================================================
    # 8. 参数与对话框
    # =============================================================================

    def confirm_params(self, station, silent: bool = False):
        self._update_config_from_ui()
        self._update_ui_from_config()
        if self.thread_manager is None:
            return
        
        if station == "all":
            success_list =[]
            for sub_station in ["dry_thread","transfer_thread","sucker_thread_1","sucker_thread_2","fulltray_thread"]:
                success = self.thread_manager.update_params(sub_station)
                success_list.append(success)
            if all(success_list) and not silent:
                QMessageBox.information(self,"提示","参数更新成功✅")
            elif not all(success_list):
                QMessageBox.warning(self, "错误", "参数更新失败❌")
        else:
            success =self.thread_manager.update_params(station)
            if success and not silent:
                QMessageBox.information(self,"提示","参数更新成功✅")
            elif not success:
                QMessageBox.warning(self, "错误", "参数更新失败❌")

    def show_pramas_set_dialog(self,station):
        if self.config_manager.config_dict != {}:
            if station == "dry":
                dry_pramas_set_dialog = DryPramasSetDialog.DryPramasSetDialog(self.config_manager,self)
                dry_pramas_set_dialog.exec_()
            if station == "transfer":
                transfer_pramas_set_dialog = TransferPramasSetDialog.TransferPramasSetDialog(self.config_manager,self)
                transfer_pramas_set_dialog.exec_()

    # =============================================================================
    # 9. 模板操作
    # =============================================================================

    def _load_template(self,station):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模板", "", "模板文件类别： *.bmp"
        )
        if not file_path:
            QMessageBox.warning(self, "错误", "未选择模板❌")
            return
        
        template = cv.imread(file_path)
        if template is None:
            QMessageBox.warning(self, "错误", "无法读取模板文件❌")
            return

        self.config_manager.set_key(f"work_{station}_params","golden_template_path",file_path)
        self._update_label_from_image(getattr(self,f"label_template_display_{station}"),template)
        QMessageBox.information(self,"提示","模板已成功加载✅")

    def create_new_tmepalte(self,station):
        image = self.current_image[station].copy()
        cv.namedWindow("Create New Template", cv.WINDOW_NORMAL)
        roi = selectROI("Create New Template", image, showCrosshair=True, fromCenter=False, rect_color=(0,255,0), line_thickness=5)
        if roi is not None:
            x,y,w,h = roi
            template = image[y:y+h, x:x+w]
            template_path, _ = QFileDialog.getSaveFileName(self, "保存模板", "", "模板文件 (*.bmp)")
            if not template_path:
                QMessageBox.warning(self, "错误", "未选择模板保存路径❌")
                return
            cv.imwrite(template_path, template)
            self.config_manager.set_key(f"work_{station}_params","golden_template_path",template_path)
            QMessageBox.information(self, "提示", f"模板已成功创建✅，模板路径：{template_path}")
            self._update_label_from_image(getattr(self,f"label_template_display_{station}"),template)
        else:
            QMessageBox.warning(self, "错误", "模板创建失败❌")

    def template_validity_test(self, station):
        """
        1. template_detector 定位产品
        2. execute_product_detection 检测 
        3. 绘制结果 
        4. 显示到对应工位 tab
        """
        image = self.current_image.get(station)
        if image is None:
            QMessageBox.warning(self, "错误", "请先选择当前图像❌")
            return

        # 根据 config 直接创建临时检测器
        params_section = "work_dry_params" if station == "dry" else "work_transfer_params"
        try:
            params = self.config_manager.get_section(params_section)
        except KeyError:
            params = {}

        pixel_size = params.get("pixel_size", 0.008823)
        product_size = params.get("product_size", [10.0, 15.0])

        ball_detector = BallDetector()
        ball_detector.update_params({
            "min_threshold": params.get("min_threshold_ball", 0),
            "max_threshold": params.get("max_threshold_ball", 255),
            "ball_area_max_threshold": params.get("ball_area_max_threshold", 1500),
            "ball_area_min_threshold": params.get("ball_area_min_threshold", 500),
            "ball_radius_tolerance": params.get("ball_radius_tolerance", 0.05),
            "std_radius": params.get("std_radius", 0.17),
            "expected_ball_count": params.get("ball_count", 0),
            "ball_search_roi": params.get("ball_search_roi", []),
            "pixel_size": pixel_size,
        })
        size_detector = SizeDetector()
        size_detector.update_params({
            "min_threshold": params.get("min_threshold_size", 0),
            "max_threshold": params.get("max_threshold_size", 255),
            "allow_tolerance_x": params.get("product_size_tolerance_x", 0.1),
            "allow_tolerance_y": params.get("product_size_tolerance_y", 0.1),
            "roi_width": 80,
            "std_size": product_size,
            "pixel_size": pixel_size,
        })
        mark_detector = MarkDetector()
        mark_detector.update_params({
            "min_threshold": params.get("min_threshold_mark", 0),
            "max_threshold": params.get("max_threshold_mark", 255),
            "min_mark_area": params.get("min_mark_area", 2000),
            "pixel_size": pixel_size,
            "mark_detect_mode": params.get("mark_detect_mode", "manual"),
            "mark_roi": params.get("mark_roi", []),
        })
        shift_detector = ShiftDetector()
        shift_detector.update_params({
            "pixel_size": pixel_size,
            "error_correction_factor": 0.7,
            "allow_tolerance_x": params.get("shift_x_tolerance", 0.05),
            "allow_tolerance_y": params.get("shift_y_tolerance", 0.05),
        })
        scratch_detector = ScratchDetector()
        scratch_detector.update_params({
            "min_threshold": params.get("min_threshold_scratch", 0),
            "max_threshold": params.get("max_threshold_scratch", 255),
            "scratch_length_threshold": params.get("scratch_length", 5.0),
            "pixel_size": pixel_size,
            "scratch_roi": params.get("scratch_roi", []),
            "roi_blocks": params.get("roi_block", []),
        })
        template_detector = TemplateDetector()
        search_roi = params.get("search_roi") or []
        template_detector.update_params({
            "template_threshold": params.get("template_threshold", 0.7),
            "search_roi": search_roi,
        })

        detectors = {
            "ball_detector": ball_detector,
            "size_detector": size_detector,
            "mark_detector": mark_detector,
            "shift_detector": shift_detector,
            "scratch_detector": scratch_detector,
        }

        template_path = params.get("golden_template_path")
        if not template_path or not os.path.exists(template_path):
            QMessageBox.warning(self, "错误", "模板路径无效或文件不存在❌")
            return

        template = cv.imread(template_path)
        if template is None:
            QMessageBox.warning(self, "错误", "无法读取模板图像❌")
            return
        img_h, img_w = image.shape[:2]
        search_roi = params.get("search_roi") or []
        if not (isinstance(search_roi, (list, tuple)) and len(search_roi) >= 4):
            search_roi = [0, 0, img_w, img_h]
        template_detector.update_params({"template_threshold": params.get("template_threshold", 0.7), "search_roi": search_roi})

        detect_image = cv.blur(image, (3, 3)) if station == "transfer" else image
        template_pos_list = template_detector.detect(template, detect_image)
        if not template_pos_list:
            QMessageBox.warning(self, "提示", "未检测到产品位置❌")
            return

        template_h, template_w = template.shape[:2]
        image_result = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        detect_params = {
            "mark_check_enable": params.get("mark_check_enable", True),
            "size_check_enable": params.get("size_check_enable", True),
            "ball_check_enable": params.get("ball_check_enable", True),
            "shift_check_enable": params.get("shift_check_enable", True),
            "scratch_check_enable": params.get("scratch_check_enable", True),
            "allow_mark": params.get("allow_mark", False),
        }
        print(detect_params)
        for x, y in template_pos_list:
            if x + template_w > img_w or y + template_h > img_h:
                continue
            product_image = image[y:y + template_h, x:x + template_w]
            success, msg, product_info = execute_product_detection(
                image=product_image,
                detectors=detectors,
                params=detect_params,
                detect_type=None,
                early_return_on_ng=False,
                error_callback=None
            )
            product_info["x"], product_info["y"] = x, y
            if not success:
                continue
            _, _, drawn_patch = draw_detection_results(
                product_image.copy(),
                product_info,
                mark_color="green" if product_info.get("defect_type") == ["OK"] else "red"
            )
            if drawn_patch is not None:
                image_result[y:y + template_h, x:x + template_w] = drawn_patch
        cv.putText(image_result, f"Search ROI", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv.rectangle(image_result, (search_roi[0], search_roi[1]), (search_roi[0] + search_roi[2], search_roi[1] + search_roi[3]), (255, 255, 0), 5)
        self._update_label_from_image(getattr(self, f"label_current_cam_live_{station}"), image_result)

    # =============================================================================
    # 10. 图像与 ROI
    # =============================================================================

    def load_current_image(self, station):
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.bmp *.jpg *.png)")
        if not image_path:
            QMessageBox.warning(self, "错误", "未选择图像❌")
            return

        image = cv.imread(image_path)
        if image is None:
            QMessageBox.warning(self, "错误", "图像加载失败❌")
            return
        self.current_image[station] = image
        if station == "fulltray":
            image_bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
            self._update_label_from_image(self.label_image_show_fulltray, image_bgr)
            h, w = image_bgr.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(image_bgr.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView_fulltray_cam_live.setScene(scene)
            self.graphicsView_fulltray_cam_live.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            self._update_label_from_image(getattr(self, f"label_current_cam_live_{station}"), image)

    def create_search_roi(self, station):
        """使用 selectROI 在当前工位图像上框选 search_roi，并保存到配置。dry/transfer/fulltray 均使用 search_roi"""
        image = self.current_image.get(station)
        if image is None:
            QMessageBox.warning(self, "错误", "请先选择当前图像❌")
            return

        cv.namedWindow("创建 Search ROI", cv.WINDOW_NORMAL)
        roi = selectROI("创建 Search ROI", image, showCrosshair=True, fromCenter=False, rect_color=(255, 255, 0), line_thickness=5)
        if roi and roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            self.config_manager.set_key(f"work_{station}_params", "search_roi", [x, y, w, h])
            # 在图像上绘制 ROI 范围并更新显示
            display_image = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
            cv.putText(display_image, "Search ROI", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv.rectangle(display_image, (x, y), (x + w, y + h), (255, 255, 0), 5)
            if station == "fulltray":
                self._update_label_from_image(self.label_image_show_fulltray, display_image)
                h, w = display_image.shape[:2]
                bytes_per_line = 3 * w
                q_image = QImage(display_image.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                scene = QGraphicsScene()
                scene.addPixmap(pixmap)
                self.graphicsView_fulltray_cam_live.setScene(scene)
                self.graphicsView_fulltray_cam_live.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            else:
                self._update_label_from_image(getattr(self, f"label_current_cam_live_{station}"), display_image)
            QMessageBox.information(self, "提示", f"Search ROI 已保存 ✅\n区域: ({x}, {y}) 宽{w} 高{h}")
        else:
            QMessageBox.information(self, "提示", "已取消创建 Search ROI")

    # =============================================================================
    # 11. 满盘功能
    # =============================================================================

    def select_fulltray_model(self):
        """选择满盘检测模型文件并保存到配置"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择满盘检测模型", "", "PyTorch 模型 (*.pth *.pt)"
        )
        if not model_path:
            return
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "模型文件不存在❌")
            return
        self.config_manager.set_key("work_fulltray_params", "model_path", model_path)
        self.lineEdit_fulltray_model_path.setText(model_path)
        self.label_fulltray_current_model.setText(os.path.basename(model_path))
        self.label_fulltray_current_model.setStyleSheet("color: green;")
        # 若线程已创建，更新参数以触发模型重载
        if self.thread_manager:
            ft = self.thread_manager.get_thread_obj("fulltray_thread")
            if ft:
                params = self.config_manager.get_section("work_fulltray_params")
                ft.update_params(params)
        QMessageBox.information(self, "提示", f"模型已选择 ✅\n{model_path}")

    def manual_test_fulltray(self):
        """手动测试满盘检测，参考 fulltray_thread 的检测方式"""
        image = self.current_image.get("fulltray")
        if image is None:
            QMessageBox.warning(self, "错误", "请先选择当前图像❌")
            return
        try:
            params = self.config_manager.get_section("work_fulltray_params")
        except KeyError:
            params = {}
        model_path = params.get("model_path")
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请先选择有效的模型文件❌")
            return
        rows = int(params.get("rows", 8))
        cols = int(params.get("cols", 16))
        search_roi = params.get("search_roi", [])
        search_roi = search_roi if isinstance(search_roi, list) and len(search_roi) >= 4 else None
        input_size = int(params.get("input_size", 150))
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = fulltray_load_model(model_path, device=device)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"模型加载失败: {e}")
            return
        try:
            # 参考 fulltray_thread._detect_fulltray_dl
            if search_roi is not None and len(search_roi) >= 4:
                x, y, w, h = int(search_roi[0]), int(search_roi[1]), int(search_roi[2]), int(search_roi[3])
                roi_image = image[y:y+h, x:x+w]
            else:
                roi_image = image.copy()
            try:
                roi_gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)
            except Exception:
                roi_gray = roi_image.copy()
            result_image = cv.cvtColor(roi_gray, cv.COLOR_GRAY2BGR)
            roi_h, roi_w = roi_gray.shape
            cell_h = roi_h // rows
            cell_w = roi_w // cols
            result_matrix = np.zeros((rows, cols), dtype=bool)
            confidence_matrix = np.zeros((rows, cols), dtype=float)
            for i in range(rows):
                for j in range(cols):
                    y1 = i * cell_h
                    x1 = j * cell_w
                    y2 = (i + 1) * cell_h if i < rows - 1 else roi_h
                    x2 = (j + 1) * cell_w if j < cols - 1 else roi_w
                    cell = roi_gray[y1:y2, x1:x2]
                    try:
                        prediction, confidence = fulltray_predict_single_image(
                            model, cell, device=device, input_size=input_size
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
                            cv.line(result_image, (cell_center_x - offset, cell_center_y - offset),
                                    (cell_center_x + offset, cell_center_y + offset), (0, 0, 255), 2)
                            cv.line(result_image, (cell_center_x - offset, cell_center_y + offset),
                                    (cell_center_x + offset, cell_center_y - offset), (0, 0, 255), 2)
                    except Exception as e:
                        print(f"满盘Cell[{i},{j}] 预测失败: {e}")
                        result_matrix[i, j] = False
            if search_roi is not None and len(search_roi) >= 4:
                x, y, w, h = int(search_roi[0]), int(search_roi[1]), int(search_roi[2]), int(search_roi[3])
                image_result_full = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
                image_result_full[y:y+h, x:x+w] = result_image
                result_image = image_result_full
            is_ok = np.all(result_matrix)
            product_count = int(np.sum(result_matrix))
            total_cells = rows * cols
            empty_count = total_cells - product_count
            avg_confidence = float(np.mean(confidence_matrix))
            # 更新显示
            display_img = cv.cvtColor(result_image, cv.COLOR_GRAY2BGR) if len(result_image.shape) == 2 else result_image.copy()
            self._update_label_from_image(self.label_image_show_fulltray, display_img)
            h, w = display_img.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(display_img.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView_fulltray_cam_live.setScene(scene)
            self.graphicsView_fulltray_cam_live.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self._update_fulltray_result(is_ok, product_count, total_cells, empty_count, avg_confidence)
            QMessageBox.information(self, "提示", f"检测完成: {'OK' if is_ok else 'NG'} | {product_count}/{total_cells} | 置信度 {avg_confidence:.2%}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"满盘检测失败: {e}")
            traceback.print_exc()

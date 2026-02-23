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
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import QLabel,QApplication
import numpy as np
from src.support.support_funs import selectROI
from ImageViewerWidget import ImageViewerWidget
from LogViewerWidget import LogViewerWidget
class MainWindow(main_window_ui.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.current_image = {
            "dry": None,
            "transfer": None,
            "sucker1": None,
            "sucker2": None,
            "fulltray": None
        }
        self.CAM_LIST=[
            {"alias": "dry_cam",        "port_ip": "192.168.1.7", "device_ip": "192.168.1.200"},
            {"alias": "transfer_cam",   "port_ip": "192.168.1.8", "device_ip": "192.168.1.201"},
            {"alias": "sucker1_cam",          "port_ip": "192.168.1.9", "device_ip": "192.168.1.202"},
            {"alias": "sucker2_cam",          "port_ip": "192.168.1.10", "device_ip": "192.168.1.203"},
            {"alias": "fulltray_cam",          "port_ip": "192.168.1.6", "device_ip": "192.168.1.204"}
        ] 

        self.MODBUS_INFO_LIST=[
            {"alias": "dry_modbus", "host_ip": "192.168.1.50", "port": 501},
            {"alias": "transfer_modbus", "host_ip": "192.168.1.50", "port": 502},
            {"alias": "sucker1_modbus", "host_ip": "192.168.1.50", "port": 503},
            {"alias": "sucker2_modbus", "host_ip": "192.168.1.50", "port": 504},
            {"alias": "fulltray_modbus", "host_ip": "192.168.1.50", "port": 505}
        ]
    
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
        self._devices_connect()
        self._all_button_connect()
        
        # 创建日志查看和图像查看tab页面
        self.log_viewer_widget = LogViewerWidget(self)
        self.image_viewer_widget = ImageViewerWidget(self)
        
        # 添加到tabWidget（在现有tab之后）
        self.tabWidget.addTab(self.log_viewer_widget, "")
        self.tabWidget.addTab(self.image_viewer_widget, "")
        
        # 设置tab标题
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.log_viewer_widget), "日志查看")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.image_viewer_widget), "图像查看")
    
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
        
        self.pushButton_template_choose.clicked.connect(lambda: self._load_template("dry"))
        self.pushButton_transfer_template_choose.clicked.connect(lambda: self._load_template("transfer"))


        self.pushButton_product_prams_load.clicked.connect(self._load_config_file)
        self.pushButton_product_prams_save.clicked.connect(self._save_config_file)

        self.pushButton_confirm_params_all.clicked.connect(lambda: self.confirm_params("all"))
        self.pushButton_confirm_params_dry.clicked.connect(lambda: self.confirm_params("dry"))
        self.pushButton_confirm_params_transfer.clicked.connect(lambda: self.confirm_params("transfer"))

        self.PushButton_detect_pramas_set_dry.clicked.connect(lambda:self.show_pramas_set_dialog("dry"))
        self.PushButton_detect_pramas_set_transfer.clicked.connect(lambda:self.show_pramas_set_dialog("transfer"))

        self.btn_create_new_template_dry.clicked.connect(lambda: self.create_new_tmepalte("dry"))
        self.btn_create_new_template_transfer.clicked.connect(lambda: self.create_new_tmepalte("transfer"))
        # self.pushButton_tempaltematch.clicked.connect(self.dry_template_validity_test)
        # self.pushButton_transfer_template_match.clicked.connect(self.transfer_template_validity_test)
        self.pushButton_current_image_select_dry.clicked.connect(lambda: self.load_current_image("dry"))
        self.pushButton_current_image_select_transfer.clicked.connect(lambda: self.load_current_image("transfer"))
        # self.pushButton_create_checkable_roi_dry.clicked.connect(lambda: self.create_search_roi("dry"))
        # self.pushButton_create_checkable_roi_transfer.clicked.connect(lambda: self.create_search_roi("transfer"))
        # self.pushButton_fulltray_save.clicked.connect(self.save_fulltray_params)
        # self.pushButton_fulltray_select_model.clicked.connect(self.select_fulltray_model)
        # self.pushButton_fulltray_set_roi.clicked.connect(self.create_fulltray_grid_roi)
        # self.pushButton_fulltray_select_image.clicked.connect(self.select_fulltray_test_image)
        # self.pushButton_fulltray_test.clicked.connect(self.manual_test_fulltray)
        # self.pushButton_product_prams_load.clicked.connect(self._load_config_file())
    
    def _all_signal_connect(self):
        pass
    
    def _devices_connect(self):
        
        for each_cam in self.CAM_LIST:
            each_cam_alias = each_cam['alias']
            if self.connection_status["camera"].get(each_cam_alias) is not None and self.connection_status["camera"].get(each_cam_alias):
                continue
            success,msg,_ = self.hardware_manager.connect(each_cam_alias)
            if not success:
                print( "警告", f"连接{each_cam_alias}失败: {msg}")
            else:
                self.connection_status["camera"][each_cam_alias] = True


        for each_modbut in self.MODBUS_INFO_LIST:
            each_modbut_alias = each_modbut['alias']
            if self.connection_status["modbus"].get(each_modbut_alias) is not None and self.connection_status["modbus"].get(each_modbut_alias):
                continue
            success,msg = self.modbus_manager.connect(each_modbut_alias)
            if not success:
                print( "警告", f"连接{each_modbut_alias}失败: {msg}")
                return
            else:
                self.connection_status["modbus"][each_modbut_alias] = True

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

    def _update_display(self,station,image,animation):
        if station == "dry":
            pass
        if station == "transfer":
            pass
        if station == "suker_1":
            pass
        if station == "sucker_2":
            pass
        if station == "fulltray":
            pass

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
        self.thread_manager = ThreadManager(
            thread_list=self.THREAD_INFO_LIST,
            hardware_manager=self.hardware_manager,
            modbus_manager=self.modbus_manager,
            config_manager=self.config_manager,
            ui=self
        )
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
        if failed_threads:
            print("以下线程启动失败:")
            for failed_msg in failed_threads:
                print(f"  - {failed_msg}")
        elif success_count == 0:
            print("警告: 没有可启动的线程")
    
    def stop(self):
        """停止所有线程"""
        self.thread_manager.stop_all_threads()
        print("所有线程已停止")

    def _pause_thread(self):
        pass

    def _resume_thread(self):
        pass
    
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

    def _load_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "./config", "配置文件 (*.json)")
        if not file_path:
            QMessageBox.warning(self, "警告", "未选择配置文件")
            return

        ret,error_message = self.config_manager.load(file_path)
        if not ret:
            QMessageBox.warning(self, "警告", f"加载配置文件失败: {error_message}❌")
            return
        self.label_13.setText(os.path.basename(file_path))
        QMessageBox.information(self, "提示", "配置文件加载成功✅")

        self.config_manager.get_section("work_dry_params")
        self._update_ui_from_config()
        
    def _save_config_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择配置文件", "./config", "配置文件 (*.json)")
        if not file_path:
            QMessageBox.warning(self, "警告", "未选择配置文件❌")
            return
        ret,error_message = self.config_manager.save(file_path)
        if not ret:
            QMessageBox.warning(self, "警告", f"保存配置文件失败: {error_message}❌")
            return
        self.label_13.setText(os.path.basename(file_path))
        QMessageBox.information(self, "提示", "配置文件保存成功✅")
    
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
         
    def confirm_params(self,station):
        if station == "all":
            success_list =[]
            for sub_station in ["dry_thread","transfer_thread","sucker1_therad","sucker2_thread","fulltray_thread"]:
                success = self.thread_manager.update_params(sub_station)
                success_list.append(success)
            if all(success_list):
                QMessageBox.information(self,"提示","参数更新成功✅")
            else:
                QMessageBox.warning(self, "错误", "参数更新失败❌")
        else:
            success =self.thread_manager.update_params(station)
            if success:
                QMessageBox.information(self,"提示","参数更新成功✅")
            else:
                QMessageBox.warning(self, "错误", "参数更新失败❌")

    def show_pramas_set_dialog(self,station):
        if self.config_manager.config_dict != {}:
            if station == "dry":
                dry_pramas_set_dialog = DryPramasSetDialog.DryPramasSetDialog(self.config_manager,self)
                dry_pramas_set_dialog.exec_()
            if station == "transfer":
                transfer_pramas_set_dialog = TransferPramasSetDialog.TransferPramasSetDialog(self.config_manager,self)
                transfer_pramas_set_dialog.exec_()

    def _update_label_from_image(self,label:QLabel,image:np.ndarray):
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
                for _ in range(2):
                    QApplication.processEvents()
                # 确保所有子widget的布局都已计算
                current_widget.update()
        
        # 切换回主界面（索引0）
        self.tabWidget.setCurrentIndex(0)
        # 再次处理事件，确保切换完成
        QApplication.processEvents()

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

    def load_current_image(self,station):
        iamge_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.bmp)")
        if not iamge_path:
            QMessageBox.warning(self, "错误", "未选择图像❌")
            return

        image = cv.imread(iamge_path)
        self.current_image[station] = image
        self._update_label_from_image(getattr(self,f"label_current_cam_live_{station}"),image)
            
















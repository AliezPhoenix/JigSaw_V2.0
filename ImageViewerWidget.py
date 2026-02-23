import os
import re
import shutil
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QLineEdit, QDateEdit, QGroupBox,
    QCheckBox, QGraphicsPixmapItem
)
from PyQt5.QtCore import Qt, QDate
from ui.GraphicsView import ImageViewer
import cv2 as cv


class ImageViewerWidget(QWidget):
    """图像查看器组件 - 支持文件夹浏览、图像信息显示、搜索、删除、另存为等功能"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_folder = None
        self.image_files = []  # 存储所有图像文件路径
        self.filtered_files = []  # 存储过滤后的图像文件路径
        self.selected_files = set()  # 存储选中的文件索引
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI界面"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧：文件列表和控制面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右侧：图像预览和信息显示
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        # 移除自定义样式，使用主界面的 dark_teal 主题样式
        # 只保留必要的特殊样式（如图像预览区域的深色背景）
    
    def create_left_panel(self):
        """创建左侧面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # 文件夹选择
        folder_group = QGroupBox("文件夹选择")
        folder_layout = QVBoxLayout()
        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setWordWrap(True)
        folder_btn = QPushButton("选择文件夹")
        folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(folder_btn)
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)
        
        # 搜索功能
        search_group = QGroupBox("搜索")
        search_layout = QVBoxLayout()
        
        # 名称搜索
        name_label = QLabel("名称搜索:")
        self.name_search_edit = QLineEdit()
        self.name_search_edit.setPlaceholderText("输入lot_id、SN等关键词...")
        self.name_search_edit.textChanged.connect(self.apply_filters)
        
        # 日期范围搜索
        date_label = QLabel("日期范围:")
        date_range_layout = QHBoxLayout()
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate())
        self.start_date_edit.dateChanged.connect(self.apply_filters)
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.dateChanged.connect(self.apply_filters)
        date_range_layout.addWidget(self.start_date_edit)
        date_range_layout.addWidget(QLabel("至"))
        date_range_layout.addWidget(self.end_date_edit)
        
        # 启用日期过滤复选框
        self.enable_date_filter = QCheckBox("启用日期过滤")
        self.enable_date_filter.stateChanged.connect(self.apply_filters)
        
        search_layout.addWidget(name_label)
        search_layout.addWidget(self.name_search_edit)
        search_layout.addWidget(date_label)
        search_layout.addLayout(date_range_layout)
        search_layout.addWidget(self.enable_date_filter)
        search_group.setLayout(search_layout)
        left_layout.addWidget(search_group)
        
        # 文件列表
        list_group = QGroupBox("文件列表")
        list_layout = QVBoxLayout()
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_item_clicked)
        self.file_list.itemSelectionChanged.connect(self.update_selection_buttons)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)  # 支持多选
        list_layout.addWidget(self.file_list)
        list_group.setLayout(list_layout)
        left_layout.addWidget(list_group, 1)
        
        # 操作按钮
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout()
        
        # 全选/取消全选
        select_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn = QPushButton("取消全选")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        select_layout.addWidget(self.select_all_btn)
        select_layout.addWidget(self.deselect_all_btn)
        
        # 删除按钮
        self.delete_btn = QPushButton("删除选中")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        
        # 另存为按钮
        self.save_as_btn = QPushButton("另存为选中")
        self.save_as_btn.clicked.connect(self.save_as_selected)
        self.save_as_btn.setEnabled(False)
        
        action_layout.addLayout(select_layout)
        action_layout.addWidget(self.delete_btn)
        action_layout.addWidget(self.save_as_btn)
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        return left_widget
    
    def create_right_panel(self):
        """创建右侧面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        
        # 图像预览区域
        preview_group = QGroupBox("图像预览")
        preview_layout = QVBoxLayout()
        
        # 使用ImageViewer（QGraphicsView）支持缩放和拖动
        self.image_viewer = ImageViewer()
        self.image_viewer.setStyleSheet("background-color: #2b2b2b; border: 1px solid gray;")
        self.image_viewer.setMinimumSize(400, 300)
        preview_layout.addWidget(self.image_viewer)
        
        # 缩放控制
        zoom_layout = QHBoxLayout()
        zoom_layout.addStretch()
        zoom_out_btn = QPushButton("缩小")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_reset_btn = QPushButton("重置")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        zoom_in_btn = QPushButton("放大")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_reset_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addStretch()
        preview_layout.addLayout(zoom_layout)
        
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group, 1)
        
        # 图像信息显示区域
        info_group = QGroupBox("图像信息")
        info_layout = QVBoxLayout()
        
        self.info_label = QLabel("未选择图像")
        self.info_label.setWordWrap(True)
        # 信息标签使用适当的样式，与主题保持一致
        self.info_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                border-radius: 4px;
                font-size: 11pt;
                line-height: 1.6;
            }
        """)
        self.info_label.setMinimumHeight(150)
        
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        return right_widget
    
    def select_folder(self):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self.current_folder = folder
            self.folder_label.setText(f"当前文件夹:\n{folder}")
            self.load_images()
    
    def load_images(self):
        """加载文件夹中的所有图像"""
        if not self.current_folder:
            return
        
        self.image_files = []
        image_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif')
        
        for file in os.listdir(self.current_folder):
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(self.current_folder, file)
                self.image_files.append(file_path)
        
        self.image_files.sort(reverse=True)  # 按文件名倒序排列（最新的在前）
        self.apply_filters()
    
    def parse_filename(self, filepath):
        """解析文件名，提取信息
        
        格式: {lot_id}_{sn_number}_{工位}_{图像类别}_{缺陷类型}_{时间戳}.bmp
        统一为6个部分：
        - NG图像: LOT147258_SN158_Dry_NG_Mark_20260112192801.bmp (parts[3]=NG, parts[4]=缺陷类型)
        - OK图像: LOT147258_SN158_Dry_OK_NONE_20260112192801.bmp (parts[3]=OK, parts[4]=NONE)
        - ORI图像: LOT147258_SN158_Dry_ORI_NONE_20260112192801.bmp (parts[3]=ORI, parts[4]=NONE)
        """
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        
        # 使用下划线分隔文件名
        parts = name_without_ext.split('_')
        
        # 默认返回值
        result = {
            'lot_id': '未知',
            'sn_number': '未知',
            'station': '未知',
            'sequence': '未知',
            'image_type': '未知',
            'defect_type': None,
            'timestamp': '未知',
            'raw_timestamp': ''
        }
        
        # 只处理6个部分的情况
        if len(parts) != 6:
            return result
        
        # 提取基本信息（前3个部分）
        result['lot_id'] = parts[0]
        result['sn_number'] = parts[1]
        result['station'] = parts[2]
        
        # 最后一部分应该是时间戳（14位数字）
        last_part = parts[5]
        if len(last_part) != 14 or not last_part.isdigit():
            return result
        
        result['raw_timestamp'] = last_part
        timestamp = last_part
        
        # parts[3] 是图像类别，parts[4] 是缺陷类型
        image_type = parts[3]
        defect_type_part = parts[4]
        
        if image_type == 'NG':
            # NG图像：parts[4] 是缺陷类型
            result['image_type'] = 'NG'
            result['defect_type'] = defect_type_part
        elif image_type in ['OK', 'ORI']:
            # OK/ORI图像：parts[4] 应该是 "NONE"
            result['image_type'] = image_type
            if defect_type_part == 'NONE':
                result['defect_type'] = None
            else:
                # 如果不是NONE，也设置为None（兼容处理）
                result['defect_type'] = None
        else:
            # 图像类别不符合预期
            return result
        
        # 解析时间戳
        try:
            dt = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
            result['timestamp'] = dt.strftime('%Y/%m/%d %H:%M:%S')
        except:
            result['timestamp'] = timestamp
        
        return result
    
    def apply_filters(self):
        """应用搜索过滤"""
        self.filtered_files = []
        
        # 名称搜索
        name_keyword = self.name_search_edit.text().strip().lower()
        
        # 日期过滤
        use_date_filter = self.enable_date_filter.isChecked()
        start_date = self.start_date_edit.date().toPyDate()
        end_date = self.end_date_edit.date().toPyDate()
        
        for filepath in self.image_files:
            # 名称过滤
            filename = os.path.basename(filepath).lower()
            if name_keyword and name_keyword not in filename:
                continue
            
            # 日期过滤
            if use_date_filter:
                info = self.parse_filename(filepath)
                raw_timestamp = info.get('raw_timestamp', '')
                if raw_timestamp:
                    try:
                        file_date = datetime.strptime(raw_timestamp[:8], '%Y%m%d').date()
                        if file_date < start_date or file_date > end_date:
                            continue
                    except:
                        continue
            
            self.filtered_files.append(filepath)
        
        # 更新列表显示
        self.update_file_list()
    
    def update_file_list(self):
        """更新文件列表显示"""
        self.file_list.clear()
        
        for filepath in self.filtered_files:
            filename = os.path.basename(filepath)
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, filepath)
            self.file_list.addItem(item)
        
        # 更新统计信息
        total_count = len(self.image_files)
        filtered_count = len(self.filtered_files)
        if total_count > 0:
            self.folder_label.setText(
                f"当前文件夹:\n{self.current_folder}\n"
                f"总计: {total_count} 张 | 显示: {filtered_count} 张"
            )
    
    def on_item_clicked(self, item):
        """列表项点击事件"""
        filepath = item.data(Qt.UserRole)
        self.display_image(filepath)
        self.update_selection_buttons()
    
    def display_image(self, filepath):
        """显示图像和信息"""
        if not os.path.exists(filepath):
            return
        
        self.current_image_path = filepath  # 保存当前显示的图像路径
        
        # 加载图像（OpenCV读取为BGR格式）
        image = cv.imread(filepath)
        if image is None:
            return
        
        # 使用ImageViewer显示图像（setImage会自动处理BGR到RGB的转换）
        self.image_viewer.setImage(image)
        
        # 显示图像信息
        info = self.parse_filename(filepath)
        info_text = f"""
        <div style="font-family: 'Microsoft YaHei', Arial, sans-serif;">
            <p><b>Lot ID:</b> {info['lot_id']}</p>
            <p><b>SN Number:</b> {info['sn_number']}</p>
            <p><b>拍摄工位:</b> {info['station']}</p>
            <p><b>图像类别:</b> {info['image_type']}</p>
            <p><b>缺陷类型:</b> {info['defect_type'] if info['defect_type'] else '无'}</p>
            <p><b>拍摄时间:</b> {info['timestamp']}</p>
        </div>
        """
        self.info_label.setText(info_text)
    
    def zoom_in(self):
        """放大图像"""
        if hasattr(self.image_viewer, 'scene_image') and self.image_viewer.scene_image.items():
            # 获取当前变换矩阵
            transform = self.image_viewer.transform()
            # 以视口中心为缩放点
            center = self.image_viewer.mapToScene(self.image_viewer.viewport().rect().center())
            # 缩放
            self.image_viewer.scale(self.image_viewer.zoom_factor, self.image_viewer.zoom_factor)
            # 居中显示
            self.image_viewer.centerOn(center)
    
    def zoom_out(self):
        """缩小图像"""
        if hasattr(self.image_viewer, 'scene_image') and self.image_viewer.scene_image.items():
            # 获取当前变换矩阵
            transform = self.image_viewer.transform()
            # 以视口中心为缩放点
            center = self.image_viewer.mapToScene(self.image_viewer.viewport().rect().center())
            # 缩放
            self.image_viewer.scale(1.0 / self.image_viewer.zoom_factor, 1.0 / self.image_viewer.zoom_factor)
            # 居中显示
            self.image_viewer.centerOn(center)
    
    def zoom_reset(self):
        """重置缩放（自适应显示）"""
        if hasattr(self.image_viewer, 'scene_image') and self.image_viewer.scene_image.items():
            # 重置变换矩阵
            self.image_viewer.resetTransform()
            # 自适应显示
            self.image_viewer.fitInView(self.image_viewer.scene_image.sceneRect(), Qt.KeepAspectRatio)
    
    def select_all(self):
        """全选"""
        self.file_list.selectAll()
        self.update_selection_buttons()
    
    def deselect_all(self):
        """取消全选"""
        self.file_list.clearSelection()
        self.update_selection_buttons()
    
    def update_selection_buttons(self):
        """更新选择按钮状态"""
        selected_count = len(self.file_list.selectedItems())
        self.delete_btn.setEnabled(selected_count > 0)
        self.save_as_btn.setEnabled(selected_count > 0)
    
    def delete_selected(self):
        """删除选中的文件"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        count = len(selected_items)
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除选中的 {count} 个文件吗？\n此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            deleted_files = []
            deleted_count = 0
            for item in selected_items:
                filepath = item.data(Qt.UserRole)
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        deleted_files.append(filepath)
                        deleted_count += 1
                except Exception as e:
                    QMessageBox.warning(self, "删除失败", f"无法删除文件:\n{filepath}\n错误: {str(e)}")
            
            if deleted_count > 0:
                QMessageBox.information(self, "删除成功", f"已成功删除 {deleted_count} 个文件")
                # 如果当前显示的图像被删除，清空显示
                if hasattr(self, 'current_image_path') and self.current_image_path in deleted_files:
                    self.info_label.setText("未选择图像")
                    # 清空ImageViewer显示
                    if hasattr(self.image_viewer, 'scene_image'):
                        for item in self.image_viewer.scene_image.items():
                            if isinstance(item, QGraphicsPixmapItem):
                                self.image_viewer.scene_image.removeItem(item)
                self.load_images()  # 重新加载图像列表
    
    def save_as_selected(self):
        """另存为选中的文件"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        # 选择目标文件夹
        target_folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not target_folder:
            return
        
        saved_count = 0
        for item in selected_items:
            filepath = item.data(Qt.UserRole)
            filename = os.path.basename(filepath)
            target_path = os.path.join(target_folder, filename)
            
            try:
                shutil.copy2(filepath, target_path)
                saved_count += 1
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"无法保存文件:\n{filename}\n错误: {str(e)}")
        
        if saved_count > 0:
            QMessageBox.information(self, "保存成功", f"已成功保存 {saved_count} 个文件到:\n{target_folder}")

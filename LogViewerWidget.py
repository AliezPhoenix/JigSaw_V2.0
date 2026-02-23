# -*- coding: utf-8 -*-
"""
日志查看器Widget
用于查看、搜索和打开日志文件
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QListWidget, QTextEdit,
                             QFileDialog, QMessageBox, QGroupBox, QDateEdit,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QDate, QDateTime
from PyQt5.QtGui import QFont
import os
import glob
import traceback
from datetime import datetime
from openpyxl import load_workbook


class LogViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_dir = "Log"
        self.init_ui()
        self.refresh_log_list()
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 搜索区域
        search_group = QGroupBox("搜索")
        search_layout = QHBoxLayout()
        
        # 文件名搜索
        self.search_label = QLabel("文件名搜索:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入文件名关键字...")
        self.search_input.textChanged.connect(self.filter_logs)
        
        # 日期搜索
        self.date_label = QLabel("日期搜索:")
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        self.date_input.dateChanged.connect(self.filter_logs)
        
        # 清除搜索按钮
        self.clear_search_btn = QPushButton("清除搜索")
        self.clear_search_btn.clicked.connect(self.clear_search)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.refresh_log_list)
        
        search_layout.addWidget(self.search_label)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.date_label)
        search_layout.addWidget(self.date_input)
        search_layout.addWidget(self.clear_search_btn)
        search_layout.addWidget(self.refresh_btn)
        search_layout.addStretch()
        
        search_group.setLayout(search_layout)
        main_layout.addWidget(search_group)
        
        # 文件列表和内容查看区域
        content_layout = QHBoxLayout()
        
        # 左侧：文件列表
        list_group = QGroupBox("日志文件列表")
        list_layout = QVBoxLayout()
        
        self.log_list = QListWidget()
        self.log_list.setMinimumWidth(400)
        self.log_list.itemDoubleClicked.connect(self.open_log_file)
        self.log_list.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 打开按钮
        self.open_btn = QPushButton("打开选中文件")
        self.open_btn.clicked.connect(self.open_selected_log)
        
        # 删除按钮
        self.delete_btn = QPushButton("删除选中文件")
        self.delete_btn.clicked.connect(self.delete_selected_log)
        self.delete_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        
        button_layout.addWidget(self.open_btn)
        button_layout.addWidget(self.delete_btn)
        
        list_layout.addWidget(self.log_list)
        list_layout.addLayout(button_layout)
        list_group.setLayout(list_layout)
        
        # 右侧：日志内容显示
        content_group = QGroupBox("日志内容")
        content_view_layout = QVBoxLayout()
        
        # 使用QTableWidget显示Excel内容，支持单元格居中
        self.log_table = QTableWidget()
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 只读
        self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.log_table.setAlternatingRowColors(True)  # 交替行颜色
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.verticalHeader().setVisible(False)
        
        content_view_layout.addWidget(self.log_table)
        content_group.setLayout(content_view_layout)
        
        content_layout.addWidget(list_group, 1)
        content_layout.addWidget(content_group, 2)
        
        main_layout.addLayout(content_layout)
    
    def refresh_log_list(self):
        """刷新日志文件列表"""
        self.log_list.clear()
        
        if not os.path.exists(self.log_dir):
            return
        
        # 获取所有.xlsx文件
        log_files = glob.glob(os.path.join(self.log_dir, "*.xlsx"))
        log_files.sort(reverse=True)  # 按时间倒序排列
        
        for log_file in log_files:
            filename = os.path.basename(log_file)
            self.log_list.addItem(filename)
        
        if self.log_list.count() > 0:
            self.log_list.setCurrentRow(0)
    
    def filter_logs(self):
        """根据搜索条件过滤日志列表"""
        search_text = self.search_input.text().lower()
        search_date = self.date_input.date().toString("yyyyMMdd")
        
        for i in range(self.log_list.count()):
            item = self.log_list.item(i)
            filename = item.text().lower()
            
            # 检查文件名是否包含搜索文本
            text_match = search_text == "" or search_text in filename
            
            # 检查日期是否匹配
            date_match = search_date in filename or search_text != ""
            
            item.setHidden(not (text_match and date_match))
    
    def clear_search(self):
        """清除搜索条件"""
        self.search_input.clear()
        self.date_input.setDate(QDate.currentDate())
        self.filter_logs()
    
    def on_selection_changed(self):
        """选中项改变时的处理"""
        current_item = self.log_list.currentItem()
        if current_item:
            self.open_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
        else:
            self.open_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
    
    def open_selected_log(self):
        """打开选中的日志文件"""
        current_item = self.log_list.currentItem()
        if current_item:
            self.open_log_file(current_item)
    
    def delete_selected_log(self):
        """删除选中的日志文件"""
        current_item = self.log_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个日志文件")
            return
        
        filename = current_item.text()
        filepath = os.path.join(self.log_dir, filename)
        
        # 确认删除
        reply = QMessageBox.question(
            self, 
            "确认删除", 
            f"确定要删除日志文件吗？\n\n文件名: {filename}\n\n此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    QMessageBox.information(self, "成功", f"已成功删除文件: {filename}")
                    # 清空表格显示
                    self.log_table.setRowCount(0)
                    self.log_table.setColumnCount(0)
                    # 刷新列表
                    self.refresh_log_list()
                else:
                    QMessageBox.warning(self, "错误", f"文件不存在: {filepath}")
            except Exception as e:
                error_msg = f"删除文件失败: {str(e)}\n"
                error_msg += f"文件路径: {filepath}\n"
                error_msg += f"错误详情: {traceback.format_exc()}"
                QMessageBox.critical(self, "错误", error_msg)
    
    def open_log_file(self, item):
        """打开日志文件并显示内容"""
        if item is None:
            return
        
        filename = item.text()
        filepath = os.path.join(self.log_dir, filename)
        
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "错误", f"文件不存在: {filepath}")
            return
        
        try:
            # 读取Excel文件
            wb = load_workbook(filepath, data_only=True)
            ws = wb.active
            
            # 获取数据范围
            max_row = ws.max_row
            max_col = ws.max_column
            
            # 读取第一行作为表头
            header_items = []
            first_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
            for col_idx, header_value in enumerate(first_row):
                if header_value is not None:
                    header_items.append(str(header_value))
                else:
                    header_items.append(f"列{col_idx + 1}")
            
            # 设置表格行列数（排除表头行）
            data_row_count = max_row - 1 if max_row > 1 else 0
            self.log_table.setRowCount(data_row_count)
            self.log_table.setColumnCount(max_col)
            
            # 设置表头
            self.log_table.setHorizontalHeaderLabels(header_items)
            
            # 读取数据行（从第二行开始）
            if data_row_count > 0:
                for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=0):
                    for col_idx, cell_value in enumerate(row, start=0):
                        if col_idx >= max_col:
                            break
                        if cell_value is not None:
                            cell_str = str(cell_value)
                        else:
                            cell_str = ""
                        
                        # 创建表格项并设置内容居中
                        item = QTableWidgetItem(cell_str)
                        item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)  # 水平和垂直居中
                        self.log_table.setItem(row_idx, col_idx, item)
            
            # 调整列宽
            self.log_table.resizeColumnsToContents()
            # 设置最小列宽
            for col_idx in range(max_col):
                current_width = self.log_table.columnWidth(col_idx)
                if current_width < 80:
                    self.log_table.setColumnWidth(col_idx, 80)
            
            # 设置表头样式
            header = self.log_table.horizontalHeader()
            header.setDefaultAlignment(Qt.AlignCenter)
            
        except Exception as e:
            error_msg = f"读取文件错误: {str(e)}\n"
            error_msg += f"文件路径: {filepath}\n"
            error_msg += f"错误详情: {traceback.format_exc()}"
            QMessageBox.critical(self, "错误", error_msg)
            # 清空表格并显示错误信息
            self.log_table.setRowCount(1)
            self.log_table.setColumnCount(1)
            error_item = QTableWidgetItem(error_msg)
            error_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.log_table.setItem(0, 0, error_item)

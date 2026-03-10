# -*- coding: utf-8 -*-
"""启动加载进度条窗口"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class LoadingSplashScreen(QWidget):
    """程序启动时显示的加载进度窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(420, 160)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # 标题
        title = QLabel("JigSaw 正在启动...")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 进度条
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setMinimumHeight(24)
        layout.addWidget(self._progress_bar)

        # 状态标签
        self._status_label = QLabel("初始化中...")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self._status_label)

        self.setStyleSheet("""
            LoadingSplashScreen {
                background-color: #2b2b2b;
                border: 1px solid #00bcd4;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 3px;
            }
        """)

    def update_progress(self, percent: int, message: str = ""):
        """更新进度条和状态。仅在关键进度点处理事件，减少 MainWindow 初始化期间的布局计算，避免影响 tab 缩放。"""
        self._progress_bar.setValue(min(100, max(0, percent)))
        if message:
            self._status_label.setText(message)
        # 节流：每 10% 或完成时处理事件，降低对 tab 布局的干扰
        if percent % 10 == 0 or percent >= 99:
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

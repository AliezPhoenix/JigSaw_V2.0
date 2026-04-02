"""
线程模块共享导入文件
统一管理所有线程类需要的导入，避免重复导入
"""

# 标准库导入
import os
import time
import gc
import threading
import traceback
from datetime import datetime
from queue import Queue

# 第三方库导入
import cv2 as cv
import numpy as np
import modbus_tk.defines as cst
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

# PyQt5 导入
from PyQt5.QtCore import QThread, pyqtSignal

# 类型检查导入（避免循环导入）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main_window import MainWindow

# 项目内部导入
from tools.hardware import Hardware_Manager, ModBus_Manager
from src.config.config_manager import ConfigManager
from src.detectors.ball_detector import BallDetector
from src.detectors.size_detector import SizeDetector
from src.detectors.template_detector import TemplateDetector
from src.detectors.shift_detector import ShiftDetector
from src.detectors.mark_detector import MarkDetector
from src.detectors.scratch_detector import ScratchDetector
from src.support.support_funs import (
    Bga_Strip,
    assign_matches_to_grid,
    hex_to_string,
    value_transmit,
    draw_detection_results,
    execute_product_detection,
    sanitize_filename_part,
    ensure_gray_u8,
    ensure_bgr_u8,
    map_product_type_to_sector,
)
from src.support.ng_monitor import check_ng_alarm, wait_for_strip_plc_choice
import typing
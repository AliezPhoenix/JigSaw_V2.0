"""
基于QLabel的交互式BGA显示组件
实现与OpenCV版本相同的功能：鼠标移动到区域时显示框选，点击时输出pos起始位置
"""

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPixmap, QImage
import cv2 as cv

from src.support.support_funs import Bga_Strip, get_bga_animation_grid_layout


class InteractiveBgaLabel(QLabel):
    """交互式BGA显示标签，支持 pos 区域交互"""

    # 信号：当点击区域时发出，参数为pos起始位置 (row, col)
    regionClicked = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # 启用鼠标跟踪
        self.setAlignment(Qt.AlignCenter)  # 图像居中显示
        self.setScaledContents(False)  # 不自动拉伸，保持宽高比

        # 区域信息
        self.regions = []
        self.original_image = None  # 原始numpy图像
        self.original_pixmap = None  # 原始QPixmap
        self.bga = None

        # 当前鼠标命中的 pos 区域
        self.current_region = None

        # 图像参数（与get_full_animation保持一致）
        self.margin = 2
        # block_size 将根据BGA实例的行列数动态计算
        self.block_height = None
        self.block_width = None

    def _get_display_info(self):
        """
        获取当前显示信息（缩放比例、显示区域等）
        当setScaledContents(False)时，手动计算保持宽高比的缩放比例
        返回: (scale_x, scale_y, display_rect) 或 None
        """
        # 获取原始pixmap
        if not self.original_pixmap or self.original_pixmap.isNull():
            return None

        pixmap_w = self.original_pixmap.width()
        pixmap_h = self.original_pixmap.height()

        if pixmap_w <= 0 or pixmap_h <= 0:
            return None

        # 获取实际显示区域（contentsRect考虑了padding和alignment）
        contents_rect = self.contentsRect()
        contents_w = contents_rect.width()
        contents_h = contents_rect.height()

        if contents_w <= 0 or contents_h <= 0:
            return None

        # 计算保持宽高比的缩放比例
        scale_x = contents_w / pixmap_w
        scale_y = contents_h / pixmap_h
        scale = min(scale_x, scale_y)  # 使用较小的缩放比例以保持宽高比

        # 计算实际显示的图像尺寸（保持宽高比）
        display_w = int(pixmap_w * scale)
        display_h = int(pixmap_h * scale)

        # 计算居中显示的矩形区域
        display_x = contents_rect.x() + (contents_w - display_w) // 2
        display_y = contents_rect.y() + (contents_h - display_h) // 2
        display_rect = QRect(display_x, display_y, display_w, display_h)

        return (scale, scale, display_rect)

    def set_bga_data(self, bga_strip_log_instance: 'Bga_Strip'):
        """
        设置BGA数据并初始化显示

        参数:
            bga_strip_log_instance: bga_strip_log类的实例
        """
        self.bga = bga_strip_log_instance

        # 获取图像
        img = self.bga.get_full_animation()
        self.original_image = img

        # 转换numpy数组为QPixmap
        # OpenCV使用BGR格式，Qt使用RGB格式，需要转换
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qt_image)

        # 计算 pos 区域信息
        self._calculate_regions()
        self.current_region = None

        # 清除当前pixmap，让paintEvent处理缩放显示
        self.setPixmap(QPixmap())

        # 触发重绘，paintEvent会根据label大小自动缩放
        self.update()

    def resizeEvent(self, event):
        """当label大小改变时，触发重绘以更新缩放"""
        super().resizeEvent(event)
        if self.original_pixmap and not self.original_pixmap.isNull():
            self.update()

    def _calculate_regions(self):
        """计算所有区域在图像中的像素坐标"""
        self.regions = []

        # 根据整体图像比例计算方块尺寸（与get_full_animation保持一致）
        h, w = self.bga.full_value.shape
        margin, block_h, block_w, _, _ = get_bga_animation_grid_layout(
            h, w, margin=self.margin, canvas_h=480, canvas_w=150
        )
        self.margin = margin
        self.block_height = block_h
        self.block_width = block_w

        for pos in self.bga.position_list:
            row_start, col_start, row_end, col_end, \
            actual_row_end, actual_col_end, \
            source_row_size, source_col_size = pos

            # 计算区域在图像中的像素坐标
            x_start = self.margin + col_start * (self.block_width + self.margin)
            y_start = self.margin + row_start * (self.block_height + self.margin)
            x_end = self.margin + actual_col_end * (self.block_width + self.margin)
            y_end = self.margin + actual_row_end * (self.block_height + self.margin)

            self.regions.append({
                'pos_start': (row_start, col_start),
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end
            })

    def _label_to_image_coords(self, label_point):
        """
        将QLabel中的坐标转换为原始图像坐标
        支持setScaledContents(False)时保持宽高比的缩放情况

        参数:
            label_point: QPoint，QLabel中的坐标

        返回:
            QPoint: 原始图像中的坐标，如果不在图像范围内则返回None
        """
        display_info = self._get_display_info()
        if not display_info:
            return None

        scale_x, scale_y, display_rect = display_info

        # 获取原始pixmap的尺寸
        if not self.original_pixmap:
            return None

        pixmap_w = self.original_pixmap.width()
        pixmap_h = self.original_pixmap.height()

        # 将label坐标转换为相对于显示区域的坐标
        relative_x = label_point.x() - display_rect.x()
        relative_y = label_point.y() - display_rect.y()

        # 检查是否在显示区域内
        if relative_x < 0 or relative_x >= display_rect.width() or \
           relative_y < 0 or relative_y >= display_rect.height():
            return None

        # 将显示区域中的坐标转换为原始图像坐标
        # 需要将坐标除以缩放比例
        img_x = int(relative_x / scale_x)
        img_y = int(relative_y / scale_y)

        # 确保坐标在有效范围内
        if 0 <= img_x < pixmap_w and 0 <= img_y < pixmap_h:
            return QPoint(img_x, img_y)
        return None

    def _find_region_at_point(self, image_point):
        """
        查找指定图像坐标点所在的区域

        参数:
            image_point: QPoint，原始图像中的坐标

        返回:
            dict: 区域信息，如果不在任何区域内则返回None
        """
        if image_point is None:
            return None

        x, y = image_point.x(), image_point.y()

        for region in self.regions:
            if (region['x_start'] <= x < region['x_end'] and
                region['y_start'] <= y < region['y_end']):
                return region

        return None

    def _create_overlay_pixmap(self):
        """绘制当前高亮区域叠加层并返回 QPixmap。"""
        if self.original_image is None:
            return None

        display_img = self.original_image.copy()

        if self.current_region is not None:
            cv.rectangle(
                display_img,
                (self.current_region["x_start"], self.current_region["y_start"]),
                (self.current_region["x_end"], self.current_region["y_end"]),
                (255, 0, 255), 3,
            )

        # 转换numpy数组为QPixmap
        img_rgb = cv.cvtColor(display_img, cv.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        image_point = self._label_to_image_coords(event.pos())
        new_region = self._find_region_at_point(image_point)
        if new_region != self.current_region:
            self.current_region = new_region
            pixmap = self._create_overlay_pixmap()
            if pixmap:
                self.setPixmap(pixmap)
                self.update()

    def leaveEvent(self, event):
        """鼠标离开事件处理，清除矩形框"""
        if self.current_region is not None:
            self.current_region = None
            if self.original_pixmap:
                self.setPixmap(self.original_pixmap)
                self.update()

    def mousePressEvent(self, event):
        """鼠标点击事件处理"""
        if event.button() != Qt.LeftButton:
            return

        image_point = self._label_to_image_coords(event.pos())
        clicked_region = self._find_region_at_point(image_point)
        if clicked_region is not None:
            pos_start = clicked_region['pos_start']
            print(f"点击区域 - pos起始位置: (row={pos_start[0]}, col={pos_start[1]})")
            self.regionClicked.emit(pos_start)
        else:
            print("点击位置不在任何区域内")

    def paintEvent(self, event):
        """绘制事件，手动绘制图像以保持宽高比"""
        if not self.original_pixmap or self.original_pixmap.isNull():
            super().paintEvent(event)
            return

        display_info = self._get_display_info()
        if not display_info:
            super().paintEvent(event)
            return

        scale_x, scale_y, display_rect = display_info

        # 获取要显示的pixmap（可能包含矩形框）
        current_pixmap = self.pixmap()
        if not current_pixmap or current_pixmap.isNull():
            current_pixmap = self.original_pixmap

        # 缩放pixmap以保持宽高比
        scaled_pixmap = current_pixmap.scaled(
            display_rect.width(),
            display_rect.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # 创建QPainter并绘制图像
        painter = QPainter(self)
        painter.drawPixmap(display_rect.x(), display_rect.y(), scaled_pixmap)


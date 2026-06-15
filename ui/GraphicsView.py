from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2 as cv

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 初始化缩放因子
        self.zoom_factor = 1.25
        self.zoom_in_pos = None

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.scene_image = QGraphicsScene()

    def setImage(self, image, reset_view=False):
        """
        设置图像
        
        Args:
            image: numpy数组图像，支持BGR、RGB或灰度格式
            reset_view: 为 True 时重置缩放并 fitInView；默认 False 时保留当前缩放（适合实时刷新）
        """
        if image is None:
            print("Error loading image")
            return
        
        # 处理不同格式的图像
        if len(image.shape) == 2:
            # 灰度图
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif len(image.shape) == 3:
            h, w, c = image.shape
            bytes_per_line = c * w
            
            # 判断是BGR还是RGB格式（OpenCV默认BGR）
            # 如果图像是BGR格式，转换为RGB
            if c == 3:
                # 假设是BGR格式，转换为RGB
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            print(f"Unsupported image shape: {image.shape}")
            return
        
        pix_image = QPixmap.fromImage(q_image.copy())

        if (
            not reset_view
            and hasattr(self, "item_image")
            and self.item_image is not None
            and self.scene_image is not None
        ):
            self.item_image.setPixmap(pix_image)
            return

        for item in self.scene_image.items():
            if isinstance(item, QGraphicsPixmapItem):
                self.scene_image.removeItem(item)
        
        self.item_image = QGraphicsPixmapItem(pix_image)
        
        self.scene_image.addItem(self.item_image)
        
        self.setScene(self.scene_image)

        self.resetTransform()
        self.fitInView(self.scene_image.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        # 检查滚动的方向
        if event.angleDelta().y() < 0:
            factor = 1 / self.zoom_factor
        else:
            factor = self.zoom_factor

        # 获取当前视口中心点作为缩放参考点
        zoom_in_pos = event.pos()
        self.zoom_in_pos = self.mapToScene(zoom_in_pos)

        # 缩放视图
        self.scale(factor, factor)

        # 更新下次缩放的参照点
        self.centerOn(self.zoom_in_pos)

    def isZoomed(self):
        return self.transform().m11() != 1.0 or self.transform().m22() != 1.0

    def get_visible_image_rect(self):
        """
        将当前视口映射到原图像素坐标，返回 (x, y, w, h)。
        无有效图像时返回 None。
        """
        if not hasattr(self, "item_image") or self.item_image is None:
            return None
        pixmap = self.item_image.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        img_w = pixmap.width()
        img_h = pixmap.height()
        if img_w <= 0 or img_h <= 0:
            return None

        scene_poly = self.mapToScene(self.viewport().rect())
        bounds = scene_poly.boundingRect()
        x0 = max(0, int(bounds.left()))
        y0 = max(0, int(bounds.top()))
        x1 = min(img_w, int(bounds.right()) + 1)
        y1 = min(img_h, int(bounds.bottom()) + 1)
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            return (0, 0, img_w, img_h)
        return (x0, y0, w, h)
    

    
    # def mousePressEvent(self, event: QMouseEvent):
    #     if event.button() == Qt.LeftButton:
    #         self.is_dragging = True
    #         self.start_pos = event.pos()

    # def mouseMoveEvent(self, event: QMouseEvent):
    #     if event.button() == Qt.LeftButton:
    #         delta = event.pos() - self.start_pos
    #         self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
    #         self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
    #         self.start_pos = event.pos()

    # def mouseReleaseEvent(self, event: QMouseEvent):
    #     if event.button() == Qt.LeftButton:
    #         self.is_dragging = False
    #         endPos = event.pos()
    #         x = endPos.x() - self.start_pos.x()
    #         y = endPos.y() - self.start_pos.y()

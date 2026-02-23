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

    def setImage(self, image):
        """
        设置图像
        
        Args:
            image: numpy数组图像，支持BGR、RGB或灰度格式
        """
        for item in self.scene_image.items():
            if isinstance(item, QGraphicsPixmapItem):
                self.scene_image.removeItem(item)
        
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
        
        pix_image = QPixmap.fromImage(q_image)
        
        self.item_image = QGraphicsPixmapItem(pix_image)
        
        self.scene_image.addItem(self.item_image)
        
        self.setScene(self.scene_image)

        self.fitInView(self.scene_image.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        # 检查滚动的方向
        if event.angleDelta().y() > 0:
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

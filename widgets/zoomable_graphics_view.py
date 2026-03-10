# -*- coding: utf-8 -*-
"""支持滚轮缩放的 QGraphicsView"""
from PyQt5.QtWidgets import QGraphicsView


class ZoomableGraphicsView(QGraphicsView):
    """支持滚轮缩放的 QGraphicsView，以鼠标位置为中心缩放"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom_factor = 1.15

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self._zoom_factor, self._zoom_factor)
        else:
            self.scale(1 / self._zoom_factor, 1 / self._zoom_factor)
        event.accept()

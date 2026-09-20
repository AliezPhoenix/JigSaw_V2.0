from PyQt5 import QtCore, QtGui, QtWidgets
from ui.theme import PALETTE


class WestTabBar(QtWidgets.QTabBar):
    """West 位置专用 TabBar：tabSizeHint 是唯一有效控制 tab 尺寸的方式（样式表 min-width/min-height 无效）"""

    BAR_THICKNESS = 130
    TAB_HEIGHT = 80

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        try:
            painter.fillRect(self.rect(), QtGui.QColor(PALETTE["tab_bg"]))
            option = QtWidgets.QStyleOptionTab()
            for index in range(self.count()):
                self.initStyleOption(option, index)
                rect = self.tabRect(index)
                selected = bool(option.state & QtWidgets.QStyle.State_Selected)
                fill = QtGui.QColor(PALETTE["tab_selected"] if selected else PALETTE["tab_bg"])
                text = QtGui.QColor(PALETTE["text"] if selected else PALETTE["text_muted"])
                painter.fillRect(rect, fill)
                painter.setPen(QtGui.QColor(PALETTE["border_lo"]))
                painter.drawLine(rect.bottomLeft(), rect.bottomRight())
                if selected:
                    painter.setPen(QtGui.QColor(PALETTE["border_hi"]))
                    painter.drawLine(rect.topLeft(), rect.topRight())
                    edge = QtCore.QRect(rect.right() - 4, rect.top(), 4, rect.height())
                    painter.fillRect(edge, QtGui.QColor(PALETTE["accent"]))
                font = painter.font()
                font.setBold(selected)
                font.setPointSize(11)
                font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 106)
                painter.setFont(font)
                painter.setPen(text)
                painter.drawText(
                    rect.adjusted(0, 0, -6, 0),
                    QtCore.Qt.AlignCenter | QtCore.Qt.TextDontClip,
                    self.tabText(index),
                )
        finally:
            painter.end()

    def tabSizeHint(self, index):
        return QtCore.QSize(self.BAR_THICKNESS, self.TAB_HEIGHT)


class TabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        QtWidgets.QTabWidget.__init__(self, parent)
        bar = WestTabBar()
        self.setTabBar(bar)
        bar.setMinimumWidth(WestTabBar.BAR_THICKNESS)
        bar.setMinimumHeight(WestTabBar.TAB_HEIGHT)

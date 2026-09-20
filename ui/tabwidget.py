from PyQt5 import QtCore, QtGui, QtWidgets
from ui.theme import PALETTE


class WestTabBar(QtWidgets.QTabBar):
    """West 位置专用 TabBar：tabSizeHint 是唯一有效控制 tab 尺寸的方式（样式表 min-width/min-height 无效）"""

    # West 下：width=条厚度(水平)，height=每 tab 高度(垂直)。仅修改此处即可调整尺寸。
    BAR_THICKNESS = 130   # 条厚度，容纳「干燥台参数」等 5 字
    TAB_HEIGHT = 80      # 每 tab 高度

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        try:
            option = QtWidgets.QStyleOptionTab()
            for index in range(self.count()):
                self.initStyleOption(option, index)
                rect = self.tabRect(index)
                selected = bool(option.state & QtWidgets.QStyle.State_Selected)
                fill = QtGui.QColor(PALETTE["tab_selected"] if selected else PALETTE["tab_bg"])
                text = QtGui.QColor(PALETTE["bg"] if selected else PALETTE["text"])
                painter.fillRect(rect, fill)
                if selected:
                    edge = QtCore.QRect(rect.right() - 3, rect.top(), 3, rect.height())
                    painter.fillRect(edge, QtGui.QColor(PALETTE["accent_hi"]))
                painter.setPen(text)
                painter.drawText(
                    rect,
                    QtCore.Qt.AlignCenter | QtCore.Qt.TextDontClip,
                    self.tabText(index),
                )
        finally:
            painter.end()

    def tabSizeHint(self, index):
        # West 位置：width=条厚度(水平)，height=每 tab 高度(垂直)
        return QtCore.QSize(self.BAR_THICKNESS, self.TAB_HEIGHT)


class TabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        QtWidgets.QTabWidget.__init__(self, parent)
        bar = WestTabBar()
        self.setTabBar(bar)
        # 直接设置 TabBar 最小宽度，确保条厚度生效（West 下 TabBar 的 width=条厚度）
        bar.setMinimumWidth(WestTabBar.BAR_THICKNESS)
        bar.setMinimumHeight(WestTabBar.TAB_HEIGHT)

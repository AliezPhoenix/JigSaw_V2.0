"""Mark 检测区域管理子对话框（干燥 / 移栽共用）。"""
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QAbstractItemView,
    QHeaderView,
)
from PyQt5.QtCore import Qt


class MarkRoiManageDialog(QDialog):
    def __init__(self, parent_dialog, title="检测区域管理"):
        super().__init__(parent_dialog)
        self.setWindowTitle(title)
        self._parent = parent_dialog
        self._lp = parent_dialog.local_params
        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["#", "x", "y", "w", "h", "最小面积(像素)"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)

        btn_row = QHBoxLayout()
        self._btn_pick = QPushButton("框选检测区域")
        self._btn_pick.clicked.connect(self._on_pick_roi)
        self._btn_del = QPushButton("删除选中")
        self._btn_del.clicked.connect(self._on_remove_selected)
        self._btn_clear = QPushButton("清除检测区域")
        self._btn_clear.clicked.connect(self._on_clear_all)
        btn_row.addWidget(self._btn_pick)
        btn_row.addWidget(self._btn_del)
        btn_row.addWidget(self._btn_clear)

        ok_row = QHBoxLayout()
        self._btn_ok = QPushButton("确定")
        self._btn_cancel = QPushButton("取消")
        self._btn_ok.clicked.connect(self._on_ok)
        self._btn_cancel.clicked.connect(self.reject)
        ok_row.addStretch()
        ok_row.addWidget(self._btn_ok)
        ok_row.addWidget(self._btn_cancel)

        root = QVBoxLayout(self)
        root.addWidget(self._table)
        root.addLayout(btn_row)
        root.addLayout(ok_row)
        self.resize(560, 320)
        self._reload_table()

    def _reload_table(self):
        if hasattr(self._parent, "_sync_mark_roi_min_areas_len"):
            self._parent._sync_mark_roi_min_areas_len()
        mrs = self._lp.get("mark_rois") or []
        areas = self._lp.get("mark_roi_min_areas") or []
        self._table.setRowCount(len(mrs))
        for i, roi in enumerate(mrs):
            if not isinstance(roi, (list, tuple)) or len(roi) < 4:
                continue
            for c, val in enumerate([i + 1, roi[0], roi[1], roi[2], roi[3]]):
                it = QTableWidgetItem(str(int(val)))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(i, c, it)
            amin = areas[i] if i < len(areas) else -1
            show = "" if (amin is None or amin < 0) else str(int(amin))
            self._table.setItem(i, 5, QTableWidgetItem(show))

    def _on_pick_roi(self):
        self._parent.create_roi("mark")
        self._reload_table()

    def _on_remove_selected(self):
        row = self._table.currentRow()
        if row < 0:
            return
        mrs = list(self._lp.get("mark_rois") or [])
        areas = list(self._lp.get("mark_roi_min_areas") or [])
        if 0 <= row < len(mrs):
            mrs.pop(row)
            if row < len(areas):
                areas.pop(row)
            self._lp["mark_rois"] = mrs
            self._lp["mark_roi_min_areas"] = areas
        self._reload_table()
        if hasattr(self._parent, "_update_template_display_with_markers"):
            self._parent._update_template_display_with_markers()

    def _on_clear_all(self):
        self._parent.clear_check_roi("mark")
        self._reload_table()

    def _on_ok(self):
        mrs = self._lp.get("mark_rois") or []
        if not mrs:
            self.accept()
            return
        areas = []
        for i in range(len(mrs)):
            it = self._table.item(i, 5)
            text = (it.text().strip() if it else "") or ""
            if not text:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"请为 ROI {i + 1} 填写最小检出面积（像素）。",
                )
                return
            try:
                v = int(float(text))
            except ValueError:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"ROI {i + 1} 的最小面积不是有效整数。",
                )
                return
            if v < 0:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"ROI {i + 1} 的最小面积不能为负数。",
                )
                return
            areas.append(v)
        self._lp["mark_roi_min_areas"] = areas
        self.accept()

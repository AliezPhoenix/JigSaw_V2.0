# -*- coding: utf-8 -*-
"""App theme: one interface, Fusion + QSS implementation.

Callers use apply(app) and bind(root). Palette, radio/checkbox indicators,
preview surfaces, and tab chrome stay inside this module.
"""
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QTabWidget, QWidget

PALETTE = {
    "bg": "#263238",
    "bg_alt": "#2b2b2b",
    "surface": "#37474f",
    "preview": "#1a1a1a",
    "border": "#78909c",
    "accent": "#00bcd4",
    "accent_hi": "#4dd0e1",
    "accent_dim": "#80cbc4",
    "text": "#eceff1",
    "text_muted": "#90a4ae",
    "ok": "#4caf50",
    "ng": "#f44336",
    "tab_bg": "#1c313a",
    "tab_selected": "#00bcd4",
    "input_bg": "#102027",
    "button_bg": "#37474f",
    "button_hover": "#455a64",
    "button_press": "#00838f",
}


def stylesheet():
    p = PALETTE
    return f"""
QWidget {{
    color: {p["text"]};
    background-color: {p["bg"]};
    font-size: 10pt;
}}
QMainWindow, QDialog, QFrame {{
    background-color: {p["bg"]};
    color: {p["text"]};
}}
QGroupBox {{
    border: 1px solid {p["border"]};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
    color: {p["text"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {p["accent_hi"]};
}}
QLabel {{
    background-color: transparent;
    color: {p["text"]};
}}
QPushButton {{
    background-color: {p["button_bg"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 24px;
}}
QPushButton:hover {{
    background-color: {p["button_hover"]};
    border-color: {p["accent"]};
}}
QPushButton:pressed {{
    background-color: {p["button_press"]};
}}
QPushButton:disabled {{
    color: {p["text_muted"]};
    background-color: {p["tab_bg"]};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
QComboBox, QDateEdit, QAbstractSpinBox {{
    background-color: {p["input_bg"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
    border-radius: 3px;
    padding: 4px;
    selection-background-color: {p["accent"]};
    selection-color: {p["bg"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {p["surface"]};
    color: {p["text"]};
    selection-background-color: {p["accent"]};
    selection-color: {p["bg"]};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {p["input_bg"]};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {p["accent"]};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QHeaderView::section {{
    background-color: {p["accent"]};
    color: {p["bg"]};
    font-weight: bold;
    padding: 5px;
    border: none;
}}
QTableWidget, QTableView, QListWidget, QTreeView {{
    background-color: {p["bg_alt"]};
    color: {p["text"]};
    gridline-color: {p["surface"]};
    alternate-background-color: {p["surface"]};
    border: 1px solid {p["border"]};
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background-color: {p["accent"]};
    color: {p["bg"]};
}}
QScrollBar:vertical {{
    background: {p["tab_bg"]};
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {p["surface"]};
    min-height: 24px;
    border-radius: 4px;
}}
QScrollBar:horizontal {{
    background: {p["tab_bg"]};
    height: 12px;
}}
QScrollBar::handle:horizontal {{
    background: {p["surface"]};
    min-width: 24px;
    border-radius: 4px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QMenuBar {{
    background-color: {p["tab_bg"]};
    color: {p["text"]};
}}
QMenuBar::item:selected {{
    background-color: {p["accent"]};
    color: {p["bg"]};
}}
QMenu {{
    background-color: {p["surface"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
}}
QMenu::item:selected {{
    background-color: {p["accent"]};
    color: {p["bg"]};
}}
QToolTip {{
    background-color: {p["surface"]};
    color: {p["text"]};
    border: 1px solid {p["accent"]};
}}
QProgressBar {{
    border: 1px solid {p["border"]};
    border-radius: 4px;
    text-align: center;
    background-color: {p["input_bg"]};
    color: {p["text"]};
}}
QProgressBar::chunk {{
    background-color: {p["accent"]};
    border-radius: 3px;
}}
QTabWidget::pane {{
    border: 1px solid {p["border"]};
    background-color: {p["bg"]};
}}
QTabBar::tab {{
    background-color: {p["tab_bg"]};
    color: {p["text"]};
    padding: 8px 12px;
    border: 1px solid {p["border"]};
}}
QTabBar::tab:left {{
    padding: 8px 12px;
}}
QTabBar::tab:selected {{
    background-color: {p["tab_selected"]};
    color: {p["bg"]};
    font-weight: 600;
}}
QTabBar::tab:hover {{
    border-color: {p["accent_hi"]};
}}
QRadioButton, QCheckBox {{
    spacing: 8px;
    color: {p["text"]};
    background-color: transparent;
}}
QRadioButton:checked, QCheckBox:checked {{
    color: {p["accent_hi"]};
    font-weight: 600;
}}
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid {p["accent_dim"]};
    background-color: {p["input_bg"]};
}}
QRadioButton::indicator:hover {{
    border-color: {p["accent_hi"]};
}}
QRadioButton::indicator:checked {{
    border: 3px solid {p["accent_hi"]};
    background-color: {p["accent"]};
}}
QRadioButton::indicator:disabled {{
    border-color: {p["surface"]};
    background-color: {p["tab_bg"]};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 2px solid {p["accent_dim"]};
    background-color: {p["input_bg"]};
}}
QCheckBox::indicator:hover {{
    border-color: {p["accent_hi"]};
}}
QCheckBox::indicator:checked {{
    border: 3px solid {p["accent_hi"]};
    background-color: {p["accent"]};
}}
QCheckBox::indicator:disabled {{
    border-color: {p["surface"]};
    background-color: {p["tab_bg"]};
}}
*[jigsawRole="preview"] {{
    border: 1px solid {p["border"]};
    background-color: {p["preview"]};
}}
LoadingSplashScreen {{
    background-color: {p["bg_alt"]};
    border: 1px solid {p["accent"]};
    border-radius: 8px;
    color: {p["text"]};
}}
LoadingSplashScreen QLabel {{
    color: {p["text"]};
}}
LoadingSplashScreen QLabel#splashStatus {{
    color: {p["text_muted"]};
    font-size: 12px;
}}
""".strip()


def _app_palette():
    pal = QPalette()
    bg = QColor(PALETTE["bg"])
    text = QColor(PALETTE["text"])
    accent = QColor(PALETTE["accent"])
    pal.setColor(QPalette.Window, bg)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, QColor(PALETTE["input_bg"]))
    pal.setColor(QPalette.AlternateBase, QColor(PALETTE["surface"]))
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, QColor(PALETTE["button_bg"]))
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, QColor(PALETTE["accent_hi"]))
    pal.setColor(QPalette.Highlight, accent)
    pal.setColor(QPalette.HighlightedText, QColor(PALETTE["bg"]))
    pal.setColor(QPalette.ToolTipBase, QColor(PALETTE["surface"]))
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Link, accent)
    pal.setColor(QPalette.PlaceholderText, QColor(PALETTE["text_muted"]))
    return pal


def apply(app):
    """Install Fusion, palette, and app QSS. Replaces qt_material."""
    app.setStyle("Fusion")
    app.setPalette(_app_palette())
    app.setStyleSheet(stylesheet())


_PREVIEW_OBJECT_NAMES = {
    "processed_image_label",
    "template_image_label",
    "label_sucker1_cam_live",
    "label_sucker2_cam_live",
    "graphicsView_fulltray_cam_live",
    "image_viewer_preview",
}
_PREVIEW_OBJECT_PREFIXES = (
    "label_image_show_",
    "label_template_display_",
    "label_current_cam_live_",
)


def _compact(ss):
    return "".join((ss or "").split()).lower()


def _is_preview_object(name):
    if not name:
        return False
    if name in _PREVIEW_OBJECT_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in _PREVIEW_OBJECT_PREFIXES)


def _is_preview_hole(ss, name=""):
    if "status" in (name or "").lower():
        return False
    compact = _compact(ss)
    if "background-color:#2b2b2b" not in compact:
        return False
    if any(token in compact for token in ("qtable", "qheaderview", "qprogressbar", "qmessagebox")):
        return False
    return True


def _is_tab_padding_hole(ss):
    compact = _compact(ss)
    return "qtabbar::tab" in compact and "background" not in compact


def _is_graphicsview_border_hole(ss):
    compact = _compact(ss)
    return "qgraphicsview" in compact and "border:0" in compact and "background" not in compact


def _repolish(widget):
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def bind(root):
    """Clear widget-level QSS holes so app Theme QSS reaches radios and previews."""
    if root is None:
        return
    widgets = [root]
    widgets.extend(root.findChildren(QWidget))
    seen = set()
    for widget in widgets:
        key = id(widget)
        if key in seen:
            continue
        seen.add(key)
        ss = widget.styleSheet() or ""
        name = widget.objectName()
        tagged = False
        if _is_preview_object(name):
            widget.setProperty("jigsawRole", "preview")
            tagged = True
        if _is_preview_hole(ss, name):
            widget.setProperty("jigsawRole", "preview")
            widget.setStyleSheet("")
            tagged = True
        elif isinstance(widget, QTabWidget) and _is_tab_padding_hole(ss):
            widget.setStyleSheet("")
            tagged = True
        elif _is_graphicsview_border_hole(ss):
            widget.setStyleSheet("")
            tagged = True
        if tagged:
            _repolish(widget)

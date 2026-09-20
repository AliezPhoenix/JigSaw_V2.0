# -*- coding: utf-8 -*-
"""App theme: one interface, Fusion + QSS implementation.

Design read: industrial AOI HMI for station operators, Tactical Telemetry
language (dark CRT / instrument panel). Dials: VARIANCE 3, MOTION 1, DENSITY 8.
Accent is aviation red only. Terminal green is reserved for the running badge.
"""
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QTabWidget, QWidget

PALETTE = {
    "bg": "#121212",
    "bg_alt": "#1a1a1a",
    "surface": "#1f1f1f",
    "preview": "#0a0a0a",
    "border": "#3a3a3a",
    "accent": "#E61919",
    "accent_hi": "#FF2A2A",
    "text": "#EAEAEA",
    "text_muted": "#8A8A8A",
    "ok": "#4AF626",
    "tab_bg": "#0a0a0a",
    "tab_selected": "#1f1f1f",
    "input_bg": "#0a0a0a",
    "button_bg": "#1f1f1f",
    "button_hover": "#2a2a2a",
    "button_press": "#3a1010",
}

_UI_FONT = '"Bahnschrift", "Segoe UI", "Microsoft YaHei UI", sans-serif'
_DATA_FONT = 'Consolas, "Cascadia Mono", "Microsoft YaHei UI", monospace'


def stylesheet():
    p = PALETTE
    return f"""
QWidget {{
    color: {p["text"]};
    background-color: {p["bg"]};
    font-family: {_UI_FONT};
    font-size: 10pt;
}}
QMainWindow, QDialog, QFrame {{
    background-color: {p["bg"]};
    color: {p["text"]};
    border-radius: 0px;
}}
QGroupBox {{
    border: 1px solid {p["border"]};
    border-radius: 0px;
    margin-top: 14px;
    padding-top: 10px;
    color: {p["text"]};
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 6px;
    color: {p["text"]};
}}
QLabel {{
    background-color: transparent;
    color: {p["text"]};
}}
QPushButton {{
    background-color: {p["button_bg"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
    border-radius: 0px;
    padding: 6px 12px;
    min-height: 26px;
}}
QPushButton:hover {{
    background-color: {p["button_hover"]};
    border-color: {p["accent"]};
}}
QPushButton:pressed {{
    background-color: {p["button_press"]};
    border-color: {p["accent"]};
}}
QPushButton:disabled {{
    color: {p["text_muted"]};
    background-color: {p["tab_bg"]};
    border-color: {p["border"]};
}}
QPushButton:focus {{
    border: 2px solid {p["accent"]};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
QComboBox, QDateEdit, QAbstractSpinBox {{
    background-color: {p["input_bg"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
    border-radius: 0px;
    padding: 4px;
    font-family: {_DATA_FONT};
    selection-background-color: {p["accent"]};
    selection-color: {p["text"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {p["surface"]};
    color: {p["text"]};
    selection-background-color: {p["accent"]};
    selection-color: {p["text"]};
    border: 1px solid {p["border"]};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {p["input_bg"]};
    border: 1px solid {p["border"]};
    border-radius: 0px;
}}
QSlider::handle:horizontal {{
    background: {p["accent"]};
    width: 14px;
    height: 18px;
    margin: -8px 0;
    border-radius: 0px;
}}
QHeaderView::section {{
    background-color: {p["surface"]};
    color: {p["text"]};
    font-weight: 700;
    padding: 6px;
    border: 1px solid {p["border"]};
    border-radius: 0px;
}}
QTableWidget, QTableView, QListWidget, QTreeView {{
    background-color: {p["bg_alt"]};
    color: {p["text"]};
    gridline-color: {p["border"]};
    alternate-background-color: {p["surface"]};
    border: 1px solid {p["border"]};
    border-radius: 0px;
    font-family: {_DATA_FONT};
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background-color: {p["accent"]};
    color: {p["text"]};
}}
QScrollBar:vertical {{
    background: {p["tab_bg"]};
    width: 10px;
    margin: 0;
    border: 1px solid {p["border"]};
}}
QScrollBar::handle:vertical {{
    background: {p["surface"]};
    min-height: 24px;
    border-radius: 0px;
}}
QScrollBar:horizontal {{
    background: {p["tab_bg"]};
    height: 10px;
    border: 1px solid {p["border"]};
}}
QScrollBar::handle:horizontal {{
    background: {p["surface"]};
    min-width: 24px;
    border-radius: 0px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QMenuBar {{
    background-color: {p["tab_bg"]};
    color: {p["text"]};
    border-bottom: 1px solid {p["border"]};
}}
QMenuBar::item:selected {{
    background-color: {p["accent"]};
    color: {p["text"]};
}}
QMenu {{
    background-color: {p["surface"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
    border-radius: 0px;
}}
QMenu::item:selected {{
    background-color: {p["accent"]};
    color: {p["text"]};
}}
QToolTip {{
    background-color: {p["surface"]};
    color: {p["text"]};
    border: 1px solid {p["accent"]};
    border-radius: 0px;
}}
QProgressBar {{
    border: 1px solid {p["border"]};
    border-radius: 0px;
    text-align: center;
    background-color: {p["input_bg"]};
    color: {p["text"]};
    font-family: {_DATA_FONT};
}}
QProgressBar::chunk {{
    background-color: {p["accent"]};
    border-radius: 0px;
}}
QTabWidget::pane {{
    border: 1px solid {p["border"]};
    background-color: {p["bg"]};
    border-radius: 0px;
}}
QTabBar::tab {{
    background-color: {p["tab_bg"]};
    color: {p["text_muted"]};
    padding: 8px 12px;
    border: 1px solid {p["border"]};
    border-radius: 0px;
}}
QTabBar::tab:left {{
    padding: 8px 12px;
}}
QTabBar::tab:selected {{
    background-color: {p["tab_selected"]};
    color: {p["text"]};
    font-weight: 700;
    border-right: 3px solid {p["accent"]};
}}
QTabBar::tab:hover {{
    color: {p["text"]};
    border-color: {p["accent"]};
}}
QRadioButton, QCheckBox {{
    spacing: 8px;
    color: {p["text"]};
    background-color: transparent;
}}
QRadioButton:checked, QCheckBox:checked {{
    color: {p["text"]};
    font-weight: 700;
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid {p["text"]};
    background-color: {p["input_bg"]};
}}
QRadioButton::indicator:hover {{
    border-color: {p["accent"]};
}}
QRadioButton::indicator:checked {{
    border: 2px solid {p["text"]};
    background-color: {p["accent"]};
}}
QRadioButton::indicator:disabled {{
    border-color: {p["border"]};
    background-color: {p["tab_bg"]};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 0px;
    border: 2px solid {p["text"]};
    background-color: {p["input_bg"]};
}}
QCheckBox::indicator:hover {{
    border-color: {p["accent"]};
}}
QCheckBox::indicator:checked {{
    border: 2px solid {p["text"]};
    background-color: {p["accent"]};
}}
QCheckBox::indicator:disabled {{
    border-color: {p["border"]};
    background-color: {p["tab_bg"]};
}}
*[jigsawRole="preview"] {{
    border: 1px solid {p["border"]};
    background-color: {p["preview"]};
    border-radius: 0px;
}}
LoadingSplashScreen {{
    background-color: {p["bg"]};
    border: 2px solid {p["accent"]};
    border-radius: 0px;
    color: {p["text"]};
}}
LoadingSplashScreen QLabel {{
    color: {p["text"]};
}}
LoadingSplashScreen QLabel#splashStatus {{
    color: {p["text_muted"]};
    font-family: {_DATA_FONT};
    font-size: 11px;
}}
""".strip()


def role_qss(role):
    """Semantic chrome for status surfaces. Running is the only terminal-green use."""
    p = PALETTE
    roles = {
        "idle": (
            f"color: {p['text']}; font-weight: 700; font-size: 18pt; "
            f"background-color: {p['bg']}; border: 2px solid {p['accent']}; padding: 8px;"
        ),
        "running": (
            f"color: {p['bg']}; font-weight: 700; font-size: 18pt; "
            f"background-color: {p['ok']}; border: 2px solid {p['ok']}; padding: 8px;"
        ),
        "connected": f"color: {p['text']}; font-weight: 600;",
        "disconnected": f"color: {p['accent']}; font-weight: 700;",
        "ok": (
            f"color: {p['text']}; font-weight: 700; font-size: 50pt; "
            f"background-color: {p['bg']}; border: 2px solid {p['text']}; padding: 8px;"
        ),
        "ng": (
            f"color: {p['text']}; font-weight: 700; font-size: 50pt; "
            f"background-color: {p['accent']}; border: 2px solid {p['accent']}; padding: 8px;"
        ),
        "model": f"color: {p['text']};",
        "alert": (
            f"QMessageBox {{ background-color: {p['bg']}; border: 2px solid {p['accent']}; }}"
            f"QMessageBox QLabel {{ color: {p['text']}; font-size: 20px; font-weight: 700; padding: 28px; }}"
        ),
    }
    return roles[role]


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

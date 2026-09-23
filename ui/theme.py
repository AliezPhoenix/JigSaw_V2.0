# -*- coding: utf-8 -*-
"""App theme: one interface, Fusion + QSS implementation.

Design read: industrial AOI HMI for station operators, qt-material dark_amber
language. Equipment-panel gray, bright amber for selection/focus. Red stays
reserved for NG, disconnect, and destructive actions.
Dials: VARIANCE 3, MOTION 1, DENSITY 8.
Tokens follow qt-material dark_amber.xml (primary #FFD740, secondary #232629).
"""
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QPushButton, QTabWidget, QWidget

PALETTE = {
    "bg": "#232629",
    "bg_alt": "#2C3136",
    "surface": "#31363B",
    "preview": "#1B1E21",
    "border": "#4F5B62",
    "border_hi": "#8A99A3",
    "border_lo": "#15181A",
    "accent": "#FFD740",
    "accent_hi": "#FFFF74",
    "text": "#FFFFFF",
    "text_muted": "#F2F4F6",
    "ok": "#3DAA6A",
    "alarm": "#C62828",
    "tab_bg": "#1B1E21",
    "tab_selected": "#31363B",
    "input_bg": "#1B1E21",
    "button_bg": "#3C464D",
    "button_hover": "#4F5B62",
    "button_press": "#5C4A12",
    "ink": "#000000",
}

_UI_FONT = '"Microsoft YaHei UI", "Bahnschrift", "Segoe UI", sans-serif'
_DATA_FONT = '"Microsoft YaHei UI", Consolas, "Cascadia Mono", monospace'


def _raised_border():
    p = PALETTE
    return (
        f"border-top: 1px solid {p['border_hi']};\n"
        f"    border-left: 1px solid {p['border_hi']};\n"
        f"    border-right: 1px solid {p['border_lo']};\n"
        f"    border-bottom: 1px solid {p['border_lo']};"
    )


def _sunken_border():
    p = PALETTE
    return (
        f"border-top: 1px solid {p['border_lo']};\n"
        f"    border-left: 1px solid {p['border_lo']};\n"
        f"    border-right: 1px solid {p['border_hi']};\n"
        f"    border-bottom: 1px solid {p['border_hi']};"
    )


def stylesheet():
    p = PALETTE
    raised = _raised_border()
    sunken = _sunken_border()
    return f"""
QWidget {{
    color: {p["text"]};
    background-color: {p["bg"]};
    font-family: {_UI_FONT};
    font-size: 11pt;
    font-weight: 500;
}}
QMainWindow, QDialog {{
    background-color: {p["bg"]};
    color: {p["text"]};
    border-radius: 0px;
}}
QWidget#centralwidget {{
    background-color: {p["bg"]};
}}
QGroupBox {{
    background-color: {p["bg_alt"]};
    border: 1px solid {p["border"]};
    border-radius: 0px;
    margin-top: 12px;
    padding-top: 10px;
    color: {p["text"]};
    font-size: 11pt;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 8px;
    color: {p["accent"]};
    font-size: 11pt;
    font-weight: 700;
    background-color: {p["bg"]};
}}
QLabel {{
    background-color: transparent;
    color: {p["text"]};
    font-size: 11pt;
}}
QLabel#label_time {{
    font-family: {_DATA_FONT};
    font-weight: 700;
    font-size: 12pt;
    padding: 4px 10px;
    background-color: {p["input_bg"]};
    {sunken}
}}
QPushButton {{
    background-color: {p["button_bg"]};
    color: {p["text"]};
    {raised}
    border-radius: 0px;
    padding: 6px 12px;
    min-height: 32px;
    font-size: 11pt;
    font-weight: 700;
}}
QPushButton:hover {{
    background-color: {p["button_hover"]};
    border-color: {p["accent"]};
}}
QPushButton:pressed {{
    background-color: {p["button_press"]};
    {sunken}
    padding-top: 7px;
    padding-bottom: 5px;
}}
QPushButton:disabled, QPushButton#pushButton_connect:disabled,
QPushButton[jigsawRole="primary"]:disabled, QPushButton[jigsawRole="pass"]:disabled,
QPushButton[jigsawRole="danger"]:disabled {{
    color: {p["text"]};
    background-color: {p["tab_bg"]};
    border: 1px solid {p["border"]};
}}
QPushButton:focus {{
    border: 2px solid {p["accent"]};
}}
QPushButton:default, QPushButton#pushButton_connect, QPushButton[jigsawRole="primary"] {{
    background-color: {p["accent"]};
    color: {p["ink"]};
    font-weight: 700;
    border-top: 1px solid {p["accent_hi"]};
    border-left: 1px solid {p["accent_hi"]};
    border-right: 1px solid {p["border_lo"]};
    border-bottom: 1px solid {p["border_lo"]};
}}
QPushButton:default:hover, QPushButton#pushButton_connect:hover, QPushButton[jigsawRole="primary"]:hover {{
    background-color: {p["accent_hi"]};
    color: {p["ink"]};
}}
QPushButton:default:pressed, QPushButton#pushButton_connect:pressed, QPushButton[jigsawRole="primary"]:pressed {{
    background-color: {p["button_press"]};
    color: {p["text"]};
    {sunken}
}}
QPushButton[jigsawRole="pass"] {{
    background-color: {p["ok"]};
    color: {p["ink"]};
    font-weight: 700;
    border-top: 1px solid {p["ok"]};
    border-left: 1px solid {p["ok"]};
    border-right: 1px solid {p["border_lo"]};
    border-bottom: 1px solid {p["border_lo"]};
}}
QPushButton[jigsawRole="pass"]:hover {{
    background-color: {p["ok"]};
    color: {p["ink"]};
    border-color: {p["text"]};
}}
QPushButton[jigsawRole="danger"] {{
    background-color: {p["alarm"]};
    color: {p["text"]};
    font-weight: 700;
    border: 1px solid {p["alarm"]};
}}
QPushButton[jigsawRole="danger"]:hover {{
    border-color: {p["text"]};
}}
QToolButton {{
    background-color: {p["button_bg"]};
    color: {p["text"]};
    {raised}
    border-radius: 0px;
    padding: 4px 8px;
    min-height: 28px;
    font-size: 11pt;
    font-weight: 700;
}}
QToolButton:hover {{
    background-color: {p["button_hover"]};
    border-color: {p["accent"]};
}}
QToolButton:pressed {{
    background-color: {p["button_press"]};
    {sunken}
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
QComboBox, QDateEdit, QAbstractSpinBox {{
    background-color: {p["input_bg"]};
    color: {p["text"]};
    {sunken}
    border-radius: 0px;
    padding: 4px 8px;
    min-height: 28px;
    font-size: 11pt;
    font-family: {_DATA_FONT};
    selection-background-color: {p["accent"]};
    selection-color: {p["ink"]};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus, QAbstractSpinBox:focus {{
    border: 2px solid {p["accent"]};
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    background-color: {p["surface"]};
    width: 22px;
    border: 1px solid {p["border"]};
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background-color: {p["button_hover"]};
    border-color: {p["accent"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
    background-color: {p["surface"]};
}}
QComboBox QAbstractItemView {{
    background-color: {p["surface"]};
    color: {p["text"]};
    selection-background-color: {p["accent"]};
    selection-color: {p["ink"]};
    border: 1px solid {p["border"]};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {p["input_bg"]};
    {sunken}
    border-radius: 0px;
}}
QSlider::handle:horizontal {{
    background: {p["accent"]};
    width: 16px;
    height: 22px;
    margin: -9px 0;
    border-radius: 0px;
    border-top: 1px solid {p["accent_hi"]};
    border-left: 1px solid {p["accent_hi"]};
    border-right: 1px solid {p["border_lo"]};
    border-bottom: 1px solid {p["border_lo"]};
}}
QHeaderView::section {{
    background-color: {p["surface"]};
    color: {p["text"]};
    font-weight: 700;
    font-size: 11pt;
    padding: 6px 6px;
    {raised}
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
    font-size: 11pt;
}}
QTableWidget::item, QTableView::item, QListWidget::item, QTreeView::item {{
    color: {p["text"]};
    padding: 4px 8px;
    min-height: 24px;
}}
QTableCornerButton::section {{
    background-color: {p["surface"]};
    border: 1px solid {p["border"]};
}}
QTableWidget::item:selected, QListWidget::item:selected, QTreeView::item:selected {{
    background-color: {p["accent"]};
    color: {p["ink"]};
}}
QListWidget::item:hover {{
    background-color: {p["button_hover"]};
}}
QScrollBar:vertical {{
    background: {p["tab_bg"]};
    width: 14px;
    margin: 0;
    border: 1px solid {p["border"]};
}}
QScrollBar::handle:vertical {{
    background: {p["surface"]};
    min-height: 28px;
    border-radius: 0px;
    border: 1px solid {p["border"]};
}}
QScrollBar::handle:vertical:hover {{
    background: {p["button_hover"]};
    border-color: {p["accent"]};
}}
QScrollBar:horizontal {{
    background: {p["tab_bg"]};
    height: 14px;
    border: 1px solid {p["border"]};
}}
QScrollBar::handle:horizontal {{
    background: {p["surface"]};
    min-width: 28px;
    border-radius: 0px;
    border: 1px solid {p["border"]};
}}
QScrollBar::handle:horizontal:hover {{
    background: {p["button_hover"]};
    border-color: {p["accent"]};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollArea {{
    border: none;
    background-color: {p["bg"]};
}}
QSplitter::handle {{
    background-color: {p["border"]};
}}
QSplitter::handle:horizontal {{
    width: 4px;
}}
QSplitter::handle:vertical {{
    height: 4px;
}}
QMenuBar {{
    background-color: {p["tab_bg"]};
    color: {p["text"]};
    border-bottom: 1px solid {p["border"]};
    font-size: 11pt;
    font-weight: 600;
}}
QMenuBar::item {{
    padding: 8px 14px;
    background-color: transparent;
}}
QMenuBar::item:selected {{
    background-color: {p["accent"]};
    color: {p["ink"]};
}}
QMenu {{
    background-color: {p["surface"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
    border-radius: 0px;
    padding: 4px;
    font-size: 11pt;
}}
QMenu::item {{
    padding: 8px 20px;
}}
QMenu::item:selected {{
    background-color: {p["accent"]};
    color: {p["ink"]};
}}
QToolTip {{
    background-color: {p["surface"]};
    color: {p["text"]};
    border: 1px solid {p["accent"]};
    border-radius: 0px;
    padding: 6px 8px;
}}
QProgressBar {{
    border: 1px solid {p["border"]};
    border-radius: 0px;
    text-align: center;
    background-color: {p["input_bg"]};
    color: {p["text"]};
    font-family: {_DATA_FONT};
    font-size: 11pt;
    min-height: 26px;
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
    color: {p["text"]};
    padding: 8px 12px;
    border: 1px solid {p["border"]};
    border-radius: 0px;
    font-size: 11pt;
    font-weight: 600;
}}
QTabBar::tab:left {{
    padding: 8px 12px;
}}
QTabBar::tab:selected {{
    background-color: {p["tab_selected"]};
    color: {p["accent"]};
    font-weight: 700;
    border-right: 4px solid {p["accent"]};
}}
QTabBar::tab:hover {{
    color: {p["text"]};
    border-color: {p["accent"]};
}}
QRadioButton, QCheckBox {{
    spacing: 10px;
    color: {p["text"]};
    background-color: transparent;
    font-size: 11pt;
    min-height: 28px;
}}
QRadioButton:checked, QCheckBox:checked {{
    color: {p["accent"]};
    font-weight: 700;
}}
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 9px;
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
    width: 18px;
    height: 18px;
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
QCalendarWidget {{
    background-color: {p["surface"]};
    color: {p["text"]};
    border: 1px solid {p["border"]};
}}
QCalendarWidget QAbstractItemView:enabled {{
    background-color: {p["bg_alt"]};
    color: {p["text"]};
    selection-background-color: {p["accent"]};
    selection-color: {p["ink"]};
}}
QCalendarWidget QWidget#qt_calendar_navigationbar {{
    background-color: {p["tab_bg"]};
}}
QStatusBar {{
    background-color: {p["tab_bg"]};
    color: {p["text"]};
    border-top: 1px solid {p["border"]};
    font-size: 11pt;
    font-weight: 600;
}}
QFrame {{
    background-color: transparent;
    color: {p["text"]};
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {p["border_hi"]};
}}
*[jigsawRole="preview"] {{
    {sunken}
    background-color: {p["preview"]};
    border-radius: 0px;
}}
QLabel#info_label {{
    color: {p["text"]};
    padding: 6px 8px;
    background-color: {p["input_bg"]};
    {sunken}
    font-family: {_DATA_FONT};
    font-size: 11pt;
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
LoadingSplashScreen QLabel#splashTitle {{
    font-size: 16pt;
    font-weight: 700;
    color: {p["accent"]};
}}
LoadingSplashScreen QLabel#splashStatus {{
    color: {p["text"]};
    font-family: {_DATA_FONT};
    font-size: 11pt;
    font-weight: 600;
}}
LoadingSplashScreen QFrame#splashRule {{
    background-color: {p["accent"]};
    max-height: 2px;
    min-height: 2px;
    border: none;
}}
""".strip()


def role_qss(role):
    """Status chrome. Amber = selected/standby. Green = running/OK. Red = NG/disconnect only."""
    p = PALETTE
    roles = {
        "idle": (
            f"color: {p['accent']}; font-weight: 700; font-size: 18pt; "
            f"background-color: {p['preview']}; border: 2px solid {p['accent']}; padding: 8px;"
        ),
        "running": (
            f"color: {p['ink']}; font-weight: 700; font-size: 18pt; "
            f"background-color: {p['ok']}; border: 2px solid {p['ok']}; padding: 8px;"
        ),
        "connected": f"color: {p['ok']}; font-weight: 700; font-size: 11pt; font-family: {_DATA_FONT};",
        "disconnected": f"color: {p['alarm']}; font-weight: 700; font-size: 11pt; font-family: {_DATA_FONT};",
        "ok": (
            f"color: {p['ink']}; font-weight: 700; font-size: 50pt; "
            f"background-color: {p['ok']}; border: 2px solid {p['ok']}; padding: 8px;"
        ),
        "ng": (
            f"color: {p['text']}; font-weight: 700; font-size: 50pt; "
            f"background-color: {p['alarm']}; border: 2px solid {p['alarm']}; padding: 8px;"
        ),
        "model": f"color: {p['text']}; font-weight: 700;",
        "alert": (
            f"QMessageBox {{ background-color: {p['surface']}; border: 2px solid {p['accent']}; }}"
            f"QMessageBox QLabel {{ color: {p['text']}; font-size: 16pt; font-weight: 700; padding: 20px; }}"
        ),
    }
    return roles[role]


def _app_palette():
    pal = QPalette()
    roles = {
        QPalette.Window: PALETTE["bg"],
        QPalette.WindowText: PALETTE["text"],
        QPalette.Base: PALETTE["input_bg"],
        QPalette.AlternateBase: PALETTE["surface"],
        QPalette.Text: PALETTE["text"],
        QPalette.Button: PALETTE["button_bg"],
        QPalette.ButtonText: PALETTE["text"],
        QPalette.BrightText: PALETTE["accent_hi"],
        QPalette.Highlight: PALETTE["accent"],
        QPalette.HighlightedText: PALETTE["ink"],
        QPalette.ToolTipBase: PALETTE["surface"],
        QPalette.ToolTipText: PALETTE["text"],
        QPalette.Link: PALETTE["accent"],
        QPalette.PlaceholderText: PALETTE["text_muted"],
        QPalette.Light: PALETTE["border_hi"],
        QPalette.Midlight: PALETTE["surface"],
        QPalette.Mid: PALETTE["border"],
        QPalette.Dark: PALETTE["border_lo"],
        QPalette.Shadow: PALETTE["ink"],
    }
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role, value in roles.items():
            pal.setColor(group, role, QColor(value))
    muted = QColor(PALETTE["text"])
    pal.setColor(QPalette.Disabled, QPalette.WindowText, muted)
    pal.setColor(QPalette.Disabled, QPalette.Text, muted)
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, muted)
    pal.setColor(QPalette.Disabled, QPalette.HighlightedText, muted)
    return pal


def apply(app):
    """Install Fusion, palette, and app QSS. Replaces qt_material."""
    app.setStyle("Fusion")
    font = QFont("Microsoft YaHei UI", 11)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    app.setPalette(_app_palette())
    app.setStyleSheet(stylesheet())


_PREVIEW_OBJECT_NAMES = {
    "processed_image_label",
    "template_image_label",
    "label_sucker1_cam_live",
    "label_sucker2_cam_live",
    "graphicsView_fulltray_cam_live",
    "image_viewer_preview",
    "label_image_show_cell",
}
_PREVIEW_OBJECT_PREFIXES = (
    "label_image_show_",
    "label_template_display_",
    "label_current_cam_live_",
)
_PRIMARY_OBJECT_NAMES = {
    "pushButton_connect",
}
_DANGER_OBJECT_NAMES = {
    "delete_btn",
}


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


def _is_info_label_hole(ss, name=""):
    if name != "info_label":
        return False
    compact = _compact(ss)
    return "color:white" in compact or "border-radius:4px" in compact


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
        tagged = bool(widget.property("jigsawRole"))
        if _is_preview_object(name):
            widget.setProperty("jigsawRole", "preview")
            tagged = True
        if name in _PRIMARY_OBJECT_NAMES and isinstance(widget, QPushButton):
            widget.setProperty("jigsawRole", "primary")
            tagged = True
        if name in _DANGER_OBJECT_NAMES and isinstance(widget, QPushButton):
            widget.setProperty("jigsawRole", "danger")
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
        elif _is_info_label_hole(ss, name):
            widget.setStyleSheet("")
            tagged = True
        if tagged:
            _repolish(widget)

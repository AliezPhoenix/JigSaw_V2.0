"""Theme module: geometric radio indicators survive hide/show; preview bind clears local QSS."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QRadioButton,
    QStyle,
    QStyleOptionButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.theme import PALETTE, apply, bind, stylesheet


def _app():
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


def _accent_pixel_count(radio):
    option = QStyleOptionButton()
    radio.initStyleOption(option)
    rect = radio.style().subElementRect(QStyle.SE_RadioButtonIndicator, option, radio)
    if not rect.isValid() or rect.width() < 4:
        rect = radio.rect()
    image = radio.grab(rect).toImage()
    accent = QColor(PALETTE["accent"])
    hits = 0
    for x in range(image.width()):
        for y in range(image.height()):
            pixel = QColor(image.pixel(x, y))
            if (
                abs(pixel.red() - accent.red()) < 50
                and abs(pixel.green() - accent.green()) < 50
                and abs(pixel.blue() - accent.blue()) < 80
            ):
                hits += 1
    return hits


def test_stylesheet_paints_radio_indicator_without_svg_image():
    qss = stylesheet()
    assert "QRadioButton::indicator:checked" in qss
    assert "QCheckBox::indicator:checked" in qss
    assert "image:" not in qss.replace(" ", "").lower()
    assert PALETTE["accent"].lower() in qss.lower()


def test_apply_does_not_use_qt_material():
    app = _app()
    apply(app)
    qss = app.styleSheet()
    assert "QRadioButton::indicator:checked" in qss
    assert "qt_material" not in qss.lower()


def test_checked_radio_indicator_survives_hide_and_show():
    app = _app()
    apply(app)
    host = QWidget()
    radio = QRadioButton("实时画面", host)
    radio.setChecked(True)
    host.resize(240, 48)
    radio.resize(220, 36)
    host.show()
    QApplication.processEvents()

    before = _accent_pixel_count(radio)
    assert before > 0

    host.hide()
    QApplication.processEvents()
    host.show()
    QApplication.processEvents()

    assert radio.isChecked()
    after = _accent_pixel_count(radio)
    assert after > 0


def test_checked_radio_survives_tab_switch():
    app = _app()
    apply(app)
    tabs = QTabWidget()
    page_a = QWidget()
    page_b = QWidget()
    layout = QVBoxLayout(page_a)
    radio = QRadioButton("QFN")
    radio.setChecked(True)
    layout.addWidget(radio)
    tabs.addTab(page_a, "工位A")
    tabs.addTab(page_b, "工位B")
    tabs.resize(320, 120)
    tabs.show()
    QApplication.processEvents()

    assert _accent_pixel_count(radio) > 0
    tabs.setCurrentIndex(1)
    QApplication.processEvents()
    tabs.setCurrentIndex(0)
    QApplication.processEvents()

    assert radio.isChecked()
    assert _accent_pixel_count(radio) > 0


def test_bind_tags_preview_by_object_name():
    app = _app()
    apply(app)
    host = QWidget()
    viewer = QLabel(host)
    viewer.setObjectName("label_current_cam_live_dry")
    bind(host)
    assert viewer.property("jigsawRole") == "preview"


def test_bind_clears_preview_local_stylesheet():
    app = _app()
    apply(app)
    host = QWidget()
    preview = QLabel(host)
    preview.setObjectName("label_image_show_dry")
    preview.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
    tabs = QTabWidget(host)
    tabs.setStyleSheet("QTabBar::tab:left { padding: 8px 12px; }")

    bind(host)

    assert preview.styleSheet() == ""
    assert preview.property("jigsawRole") == "preview"
    assert tabs.styleSheet() == ""


def test_main_entry_does_not_import_qt_material():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "main.py"), encoding="utf-8") as handle:
        source = handle.read()
    assert "qt_material" not in source
    assert "from ui.theme import apply" in source

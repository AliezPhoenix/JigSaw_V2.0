"""满盘 Cell 标注模态对话框。"""

import json
import os
from pathlib import Path

import cv2 as cv
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.fulltray_dl.cell_extract import extract_cells_from_roi

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class CellLabelDialog(QDialog):
  """模态 Cell 标注工具；参数来自打开时的 config 快照。"""

  def __init__(self, parent=None, params_snapshot=None, get_current_fulltray_image=None):
    super().__init__(parent)
    self.setWindowTitle("满盘 Cell 标注工具")
    self.setModal(True)
    self.resize(1100, 750)

    self.params = dict(params_snapshot or {})
    self.get_current_fulltray_image = get_current_fulltray_image

    self.rows = int(self.params.get("rows", 8))
    self.cols = int(self.params.get("cols", 16))
    self.search_roi = self.params.get("search_roi") or []
    self.input_size = int(self.params.get("input_size", 150))

    self.image_dir = None
    self.image_files = []
    self.current_index = 0
    self.labels = {}

    self._build_ui()
    self._update_extract_button_state()

  def _build_ui(self):
    layout = QVBoxLayout(self)

    top = QHBoxLayout()
    self.btn_extract = QPushButton("从当前满盘图提取 Cell")
    self.btn_extract.clicked.connect(self.extract_from_current_image)
    top.addWidget(self.btn_extract)

    self.btn_select_dir = QPushButton("选择 Cell 目录")
    self.btn_select_dir.clicked.connect(self.select_image_dir)
    top.addWidget(self.btn_select_dir)

    roi_text = self.search_roi if isinstance(self.search_roi, list) and len(self.search_roi) >= 4 else "未设置"
    self.label_params = QLabel(
      f"行={self.rows}  列={self.cols}  ROI={roi_text}  input_size={self.input_size}"
    )
    self.label_params.setStyleSheet("font-weight: bold;")
    top.addWidget(self.label_params)

    self.label_stats = QLabel("")
    self.label_stats.setStyleSheet("color: blue;")
    top.addWidget(self.label_stats)
    layout.addLayout(top)

    mid = QHBoxLayout()
    image_col = QVBoxLayout()
    self.label_image = QLabel("图像将显示在这里")
    self.label_image.setMinimumSize(560, 400)
    self.label_image.setAlignment(Qt.AlignCenter)
    self.label_image.setStyleSheet("border: 1px solid gray; background: #f0f0f0;")
    image_col.addWidget(self.label_image)
    self.label_image_info = QLabel("")
    self.label_image_info.setAlignment(Qt.AlignCenter)
    image_col.addWidget(self.label_image_info)
    mid.addLayout(image_col)

    right = QVBoxLayout()
    self.btn_has_product = QPushButton("有产品 (1)")
    self.btn_has_product.setMinimumHeight(50)
    self.btn_has_product.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
    self.btn_has_product.clicked.connect(lambda: self.label_current("has_product"))
    right.addWidget(self.btn_has_product)

    self.btn_empty = QPushButton("无产品 (2)")
    self.btn_empty.setMinimumHeight(50)
    self.btn_empty.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
    self.btn_empty.clicked.connect(lambda: self.label_current("empty"))
    right.addWidget(self.btn_empty)

    self.btn_skip = QPushButton("跳过")
    self.btn_skip.clicked.connect(self.next_image)
    right.addWidget(self.btn_skip)

    self.progress_bar = QProgressBar()
    right.addWidget(self.progress_bar)

    right.addWidget(QLabel("文件列表:"))
    self.list_files = QListWidget()
    self.list_files.setMaximumWidth(280)
    self.list_files.itemClicked.connect(self.on_file_selected)
    right.addWidget(self.list_files)
    mid.addLayout(right)
    layout.addLayout(mid)

    bottom = QHBoxLayout()
    self.btn_prev = QPushButton("上一张 (←)")
    self.btn_prev.clicked.connect(self.prev_image)
    bottom.addWidget(self.btn_prev)
    self.btn_next = QPushButton("下一张 (→)")
    self.btn_next.clicked.connect(self.next_image)
    bottom.addWidget(self.btn_next)
    self.btn_save = QPushButton("保存标注")
    self.btn_save.clicked.connect(self.save_labels)
    bottom.addWidget(self.btn_save)
    self.btn_export = QPushButton("导出到训练集")
    self.btn_export.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
    self.btn_export.clicked.connect(self.export_to_dataset)
    bottom.addWidget(self.btn_export)
    layout.addLayout(bottom)

    self.btn_has_product.setShortcut("1")
    self.btn_empty.setShortcut("2")
    self.btn_next.setShortcut(Qt.Key_Right)
    self.btn_prev.setShortcut(Qt.Key_Left)

  def _update_extract_button_state(self):
    has_image = False
    if self.get_current_fulltray_image:
      img = self.get_current_fulltray_image()
      has_image = img is not None
    has_roi = isinstance(self.search_roi, list) and len(self.search_roi) >= 4
    self.btn_extract.setEnabled(has_image and has_roi)
    if not has_image:
      self.btn_extract.setToolTip("请先在满盘 tab 选择或拍摄图像")
    elif not has_roi:
      self.btn_extract.setToolTip("请先在满盘 tab 设置 ROI")
    else:
      self.btn_extract.setToolTip("")

  def extract_from_current_image(self):
    if not self.get_current_fulltray_image:
      QMessageBox.warning(self, "错误", "无法获取当前满盘图像")
      return
    image = self.get_current_fulltray_image()
    if image is None:
      QMessageBox.warning(self, "提示", "请先选择/拍摄满盘图像")
      return
    if not (isinstance(self.search_roi, list) and len(self.search_roi) >= 4):
      QMessageBox.warning(self, "提示", "请先在满盘 tab 设置有效的 search_roi")
      return

    output_dir = QFileDialog.getExistingDirectory(self, "选择 Cell 保存目录")
    if not output_dir:
      return
    try:
      count = extract_cells_from_roi(
        image,
        self.search_roi,
        self.rows,
        self.cols,
        self.input_size,
        output_dir,
        image_stem="fulltray",
      )
      QMessageBox.information(self, "完成", f"已提取 {count} 个 cell 到:\n{output_dir}")
      self.image_dir = output_dir
      self.load_images()
    except Exception as e:
      QMessageBox.critical(self, "错误", f"提取失败:\n{e}")

  def select_image_dir(self):
    dir_path = QFileDialog.getExistingDirectory(self, "选择包含 cell 图像的目录")
    if dir_path:
      self.image_dir = dir_path
      self.load_images()

  def load_images(self):
    if not self.image_dir:
      return
    self.image_files = []
    for ext in IMAGE_EXTENSIONS:
      self.image_files.extend(Path(self.image_dir).glob(f"*{ext}"))
      self.image_files.extend(Path(self.image_dir).glob(f"*{ext.upper()}"))
    self.image_files = sorted({str(f) for f in self.image_files})
    if not self.image_files:
      QMessageBox.warning(self, "警告", "目录中没有找到图像文件")
      return
    self.current_index = 0
    self.progress_bar.setMaximum(len(self.image_files))
    self.load_labels()
    self.update_file_list()
    self.update_image_display()
    self.update_stats()

  def update_file_list(self):
    self.list_files.clear()
    for i, file_path in enumerate(self.image_files):
      filename = os.path.basename(file_path)
      item = QListWidgetItem(f"{i + 1}. {filename}")
      if filename in self.labels:
        label = self.labels[filename]
        if label == "has_product":
          item.setBackground(Qt.green)
          item.setText(f"{i + 1}. {filename} [有产品]")
        elif label == "empty":
          item.setBackground(Qt.red)
          item.setText(f"{i + 1}. {filename} [无产品]")
      if i == self.current_index:
        item.setBackground(Qt.yellow)
      self.list_files.addItem(item)

  def update_image_display(self):
    if not self.image_files or self.current_index >= len(self.image_files):
      return
    file_path = self.image_files[self.current_index]
    filename = os.path.basename(file_path)
    image = cv.imread(file_path)
    if image is None:
      self.label_image.setText(f"无法读取:\n{filename}")
      return
    if len(image.shape) == 3:
      image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
      image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    h, w = image_rgb.shape[:2]
    max_size = 560
    if max(w, h) > max_size:
      scale = max_size / max(w, h)
      image_rgb = cv.resize(image_rgb, (int(w * scale), int(h * scale)))
    h, w, ch = image_rgb.shape
    q_image = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    self.label_image.setPixmap(QPixmap.fromImage(q_image.copy()))
    label_text = self.labels.get(filename, "未标注")
    if label_text == "has_product":
      label_text = "有产品"
    elif label_text == "empty":
      label_text = "无产品"
    self.label_image_info.setText(
      f"{filename}  |  {label_text}  |  {self.current_index + 1}/{len(self.image_files)}"
    )
    self.progress_bar.setValue(self.current_index + 1)
    self.list_files.setCurrentRow(self.current_index)

  def label_current(self, label):
    if not self.image_files or self.current_index >= len(self.image_files):
      return
    filename = os.path.basename(self.image_files[self.current_index])
    self.labels[filename] = label
    self.save_labels()
    self.next_image()

  def next_image(self):
    if self.image_files and self.current_index < len(self.image_files) - 1:
      self.current_index += 1
      self.update_image_display()
      self.update_file_list()

  def prev_image(self):
    if self.current_index > 0:
      self.current_index -= 1
      self.update_image_display()
      self.update_file_list()

  def on_file_selected(self, item):
    row = self.list_files.row(item)
    if 0 <= row < len(self.image_files):
      self.current_index = row
      self.update_image_display()
      self.update_file_list()

  def update_stats(self):
    if not self.image_files:
      return
    total = len(self.image_files)
    labeled = len(self.labels)
    has_product = sum(1 for v in self.labels.values() if v == "has_product")
    empty = sum(1 for v in self.labels.values() if v == "empty")
    self.label_stats.setText(
      f"总计 {total} | 已标注 {labeled} | 有产品 {has_product} | 无产品 {empty}"
    )

  def save_labels(self):
    if not self.image_dir:
      return
    labels_file = os.path.join(self.image_dir, "labels.json")
    with open(labels_file, "w", encoding="utf-8") as f:
      json.dump(self.labels, f, ensure_ascii=False, indent=2)
    self.update_stats()

  def load_labels(self):
    if not self.image_dir:
      return
    labels_file = os.path.join(self.image_dir, "labels.json")
    if os.path.isfile(labels_file):
      try:
        with open(labels_file, "r", encoding="utf-8") as f:
          self.labels = json.load(f)
      except (json.JSONDecodeError, OSError):
        self.labels = {}

  def export_to_dataset(self):
    if not self.labels:
      QMessageBox.warning(self, "警告", "没有标注数据可导出")
      return
    output_dir = QFileDialog.getExistingDirectory(self, "选择训练集根目录 <dataset_root>")
    if not output_dir:
      return
    train_has = os.path.join(output_dir, "train", "has_product")
    train_empty = os.path.join(output_dir, "train", "empty")
    os.makedirs(train_has, exist_ok=True)
    os.makedirs(train_empty, exist_ok=True)
    copied = 0
    skipped = 0
    for filename, label in self.labels.items():
      src_path = os.path.join(self.image_dir, filename)
      if not os.path.isfile(src_path):
        skipped += 1
        continue
      img = cv.imread(src_path)
      if img is None:
        skipped += 1
        continue
      img_resized = cv.resize(
        img, (self.input_size, self.input_size), interpolation=cv.INTER_AREA
      )
      if label == "has_product":
        dst_path = os.path.join(train_has, filename)
      elif label == "empty":
        dst_path = os.path.join(train_empty, filename)
      else:
        continue
      cv.imwrite(dst_path, img_resized)
      copied += 1
    msg = f"已导出 {copied} 张图像到训练集（{self.input_size}x{self.input_size}）"
    if skipped:
      msg += f"\n跳过 {skipped} 个缺失或无效文件"
    QMessageBox.information(self, "完成", msg)

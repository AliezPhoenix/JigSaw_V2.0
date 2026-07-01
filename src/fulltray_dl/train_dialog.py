"""满盘模型 CPU 训练对话框与后台线程。"""

import os
import shutil
import tempfile

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
)

from src.fulltray_dl.dataset import count_class_images
from src.fulltray_dl.split import split_dataset, val_dir_has_data
from src.fulltray_dl.trainer import train as run_train


class FulltrayTrainWorker(QThread):
    batch_progress = pyqtSignal(int, int, int, int, str, int, int, float)
    epoch_done = pyqtSignal(int, int, float, float, float, float)
    preparing = pyqtSignal()
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        train_dir,
        val_dir,
        save_dir,
        input_size=150,
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        parent=None,
    ):
        super().__init__(parent)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.save_dir = save_dir
        self.input_size = input_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def run(self):
        try:
            self.preparing.emit()
            best_path = run_train(
                self.train_dir,
                self.val_dir,
                self.save_dir,
                device="cpu",
                pretrained=True,
                input_size=self.input_size,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                num_workers=0,
                should_stop=self.isInterruptionRequested,
                on_batch=lambda step, total, epoch, total_epochs, phase, batch, num_batches, acc: (
                    self.batch_progress.emit(
                        step, total, epoch, total_epochs, phase, batch, num_batches, acc
                    )
                ),
                on_epoch=lambda e, total, tl, ta, vl, va: self.epoch_done.emit(
                    e, total, tl, ta, vl, va
                ),
            )
            self.finished_ok.emit(best_path)
        except Exception as e:
            self.failed.emit(str(e))


class FulltrayTrainDialog(QDialog):
    """选择数据集、划分 val、CPU 训练并手动保存模型。"""

    def __init__(self, parent=None, params_snapshot=None):
        super().__init__(parent)
        self.setWindowTitle("满盘模型训练 (CPU)")
        self.setModal(True)
        self.params = dict(params_snapshot or {})
        self.input_size = int(self.params.get("input_size", 150))
        self.dataset_root = ""
        self._worker = None
        self._temp_dir = None
        self._progress_dialog = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.edit_dataset = QLineEdit()
        self.edit_dataset.setReadOnly(True)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._pick_dataset)
        row = QHBoxLayout()
        row.addWidget(self.edit_dataset)
        row.addWidget(btn_browse)
        form.addRow("训练集根目录:", row)

        self.edit_epochs = QLineEdit("50")
        self.edit_batch = QLineEdit("16")
        self.edit_lr = QLineEdit("0.001")
        form.addRow("Epochs:", self.edit_epochs)
        form.addRow("Batch size:", self.edit_batch)
        form.addRow("学习率:", self.edit_lr)
        form.addRow("input_size:", QLabel(str(self.input_size)))
        layout.addLayout(form)

        self.label_status = QLabel("请选择包含 train/has_product 与 train/empty 的数据集目录")
        layout.addWidget(self.label_status)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_start)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _pick_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练集根目录 <dataset_root>")
        if path:
            self.dataset_root = path
            self.edit_dataset.setText(path)

    def _validate_dataset(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "提示", "请先选择训练集根目录")
            return None
        train_dir = os.path.join(self.dataset_root, "train")
        counts = count_class_images(train_dir)
        if counts.get("has_product", 0) < 1 or counts.get("empty", 0) < 1:
            QMessageBox.warning(
                self,
                "数据不足",
                "train/has_product 与 train/empty 均须至少 1 张图像。\n"
                "请先完成双类标注并导出训练集。",
            )
            return None
        return train_dir

    def _validate_val_dir(self, val_dir: str) -> bool:
        counts = count_class_images(val_dir)
        if counts.get("has_product", 0) < 1 or counts.get("empty", 0) < 1:
            QMessageBox.warning(
                self,
                "验证集不足",
                "val/has_product 与 val/empty 均须至少 1 张图像。\n"
                "请重新导出训练集，或清空 val/ 后再次训练以自动划分。",
            )
            return False
        return True

    def _start_training(self):
        train_dir = self._validate_dataset()
        if not train_dir:
            return
        val_dir = os.path.join(self.dataset_root, "val")
        if not val_dir_has_data(val_dir):
            reply = QMessageBox.question(
                self,
                "划分验证集",
                "将从 train 中移动约 15% 样本到 val（train 样本会减少）。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            try:
                moved = split_dataset(train_dir, val_dir, val_ratio=0.15)
                self.label_status.setText(f"已划分验证集，移动 {moved} 张图像到 val/")
            except Exception as e:
                QMessageBox.critical(self, "划分失败", str(e))
                return
            train_counts = count_class_images(train_dir)
            if train_counts.get("has_product", 0) < 1 or train_counts.get("empty", 0) < 1:
                QMessageBox.warning(
                    self,
                    "划分后训练集不足",
                    "自动划分后 train 某一类样本为 0。\n"
                    "请增加标注样本或清空 val/ 后重新导出训练集。",
                )
                return
        else:
            QMessageBox.information(self, "提示", "val/ 已有完整双类数据，跳过自动划分")

        if not self._validate_val_dir(val_dir):
            return

        try:
            epochs = int(self.edit_epochs.text())
            batch_size = int(self.edit_batch.text())
            lr = float(self.edit_lr.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "Epochs/Batch/学习率须为有效数字")
            return

        self._temp_dir = tempfile.mkdtemp(prefix="fulltray_train_")
        self.btn_start.setEnabled(False)
        progress = QProgressDialog("正在初始化...", "取消", 0, 0, self)
        progress.setWindowTitle("CPU 训练")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumWidth(360)
        self._progress_dialog = progress

        self._worker = FulltrayTrainWorker(
            train_dir,
            val_dir,
            self._temp_dir,
            input_size=self.input_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            parent=self,
        )

        _phase_labels = {"train": "训练", "val": "验证"}

        def on_preparing():
            progress.setRange(0, 0)
            progress.setLabelText("正在加载数据与预训练权重文件...")
            QApplication.processEvents()

        def on_batch(step, total_steps, epoch, total_epochs, phase, batch, num_batches, acc):
            progress.setRange(0, total_steps)
            progress.setValue(step)
            phase_label = _phase_labels.get(phase, phase)
            progress.setLabelText(
                f"Epoch {epoch}/{total_epochs}  {phase_label} {batch}/{num_batches}  "
                f"acc={acc:.1f}%"
            )
            QApplication.processEvents()

        def on_epoch_done(epoch, total_epochs, tl, ta, vl, va):
            progress.setLabelText(
                f"Epoch {epoch}/{total_epochs} 完成  "
                f"train_acc={ta:.1f}%  val_acc={va:.1f}%"
            )
            QApplication.processEvents()

        def on_ok(best_path):
            progress.close()
            self._progress_dialog = None
            self.btn_start.setEnabled(True)
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存模型", "best_model.pth", "PyTorch (*.pth)"
            )
            if save_path:
                shutil.copy2(best_path, save_path)
                QMessageBox.information(self, "完成", f"模型已保存至:\n{save_path}")
            self._cleanup_temp()

        def on_fail(msg):
            progress.close()
            self._progress_dialog = None
            self.btn_start.setEnabled(True)
            if msg == "训练已取消":
                self.label_status.setText("训练已取消")
            else:
                QMessageBox.critical(self, "训练失败", msg)
            self._cleanup_temp()

        def on_cancel():
            if self._worker and self._worker.isRunning():
                self._worker.requestInterruption()

        progress.canceled.connect(on_cancel)
        self._worker.preparing.connect(on_preparing)
        self._worker.batch_progress.connect(on_batch)
        self._worker.epoch_done.connect(on_epoch_done)
        self._worker.finished_ok.connect(on_ok)
        self._worker.failed.connect(on_fail)
        progress.show()
        QApplication.processEvents()
        self._worker.start()

    def _cleanup_temp(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(3000)
        self._cleanup_temp()
        super().closeEvent(event)

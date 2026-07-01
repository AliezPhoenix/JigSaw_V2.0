"""训练集 / 验证集划分。"""

import random
import shutil
from pathlib import Path

from .dataset import IMAGE_EXTENSIONS


def _list_images(directory: Path):
    if not directory.is_dir():
        return []
    return [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


def val_dir_has_data(val_dir: str) -> bool:
    """val 目录两类子目录均至少有一张图像时视为已划分。"""
    val_path = Path(val_dir)
    for category in ("has_product", "empty"):
        if not _list_images(val_path / category):
            return False
    return True


def split_dataset(
    train_dir: str,
    val_dir: str,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> int:
    """
    从 train 按类移动部分样本到 val（非复制）。

    Returns:
        移动到 val 的样本总数。
    """
    random.seed(seed)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    val_path.mkdir(parents=True, exist_ok=True)
    moved_total = 0

    for category in ("has_product", "empty"):
        category_train_dir = train_path / category
        files = _list_images(category_train_dir)
        if not files:
            continue
        random.shuffle(files)
        n = len(files)
        if n <= 1:
            continue
        n_val = max(1, int(n * val_ratio))
        if n_val >= n:
            n_val = n - 1

        val_category_dir = val_path / category
        val_category_dir.mkdir(parents=True, exist_ok=True)
        for f in files[:n_val]:
            dst = val_category_dir / f.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(f), str(dst))
            moved_total += 1
    return moved_total

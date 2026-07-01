"""满盘 MobileNetV2 CPU 训练。"""

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.support.support_funs import MobileNetV2ClassifierFulltray

from .dataset import get_dataloaders


def _train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    should_stop=None,
    *,
    epoch=0,
    total_epochs=0,
    step_offset=0,
    total_steps=0,
    on_batch=None,
):
    should_stop = should_stop or (lambda: False)
    num_batches = len(train_loader)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        if should_stop():
            raise RuntimeError("训练已取消")
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if on_batch:
            running_acc = 100.0 * correct / max(total, 1)
            on_batch(
                step_offset + batch_idx,
                total_steps,
                epoch,
                total_epochs,
                "train",
                batch_idx,
                num_batches,
                running_acc,
            )
    epoch_loss = running_loss / max(num_batches, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


def _validate(
    model,
    val_loader,
    criterion,
    device,
    should_stop=None,
    *,
    epoch=0,
    total_epochs=0,
    step_offset=0,
    total_steps=0,
    on_batch=None,
):
    should_stop = should_stop or (lambda: False)
    num_batches = len(val_loader)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 1):
            if should_stop():
                raise RuntimeError("训练已取消")
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if on_batch:
                running_acc = 100.0 * correct / max(total, 1)
                on_batch(
                    step_offset + batch_idx,
                    total_steps,
                    epoch,
                    total_epochs,
                    "val",
                    batch_idx,
                    num_batches,
                    running_acc,
                )
    epoch_loss = running_loss / max(num_batches, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


def train(
    train_dir: str,
    val_dir: str,
    save_dir: str,
    *,
    device: str = "cpu",
    pretrained: bool = True,
    input_size: int = 150,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    should_stop: Optional[Callable[[], bool]] = None,
    on_batch: Optional[
        Callable[[int, int, int, int, str, int, int, float], None]
    ] = None,
    on_epoch: Optional[Callable[[int, int, float, float, float, float], None]] = None,
) -> str:
    """
    训练并保存最佳 checkpoint。

    Returns:
        best_model.pth 的绝对路径。

    Raises:
        RuntimeError: 用户取消训练。
    """
    should_stop = should_stop or (lambda: False)
    torch_device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        train_dir,
        val_dir,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
    )

    model = MobileNetV2ClassifierFulltray(num_classes=2, pretrained=pretrained)
    model = model.to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pth")
    steps_per_epoch = len(train_loader) + len(val_loader)
    total_steps = max(epochs * steps_per_epoch, 1)

    for epoch in range(epochs):
        if should_stop():
            raise RuntimeError("训练已取消")

        epoch_no = epoch + 1
        step_base = epoch * steps_per_epoch
        train_loss, train_acc = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            torch_device,
            should_stop,
            epoch=epoch_no,
            total_epochs=epochs,
            step_offset=step_base,
            total_steps=total_steps,
            on_batch=on_batch,
        )
        val_loss, val_acc = _validate(
            model,
            val_loader,
            criterion,
            torch_device,
            should_stop,
            epoch=epoch_no,
            total_epochs=epochs,
            step_offset=step_base + len(train_loader),
            total_steps=total_steps,
            on_batch=on_batch,
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                best_model_path,
            )

        if on_epoch:
            on_epoch(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)

        if should_stop():
            raise RuntimeError("训练已取消")

    if not os.path.isfile(best_model_path):
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
            },
            best_model_path,
        )
    return os.path.abspath(best_model_path)

# -*- coding: utf-8 -*-
"""
人机操作审计日志（与检测批次 Excel 日志职责分离）。

落盘目录默认为工作目录下 ``Log/operations/``，按周滚动；首启自动创建目录。
现场可直接用文本编辑器打开；程序内通过 ``log_operation`` 写入。

格式（单行 UTF-8）::

    YYYY-MM-DD HH:MM:SS | 等级 | 来源 | 事件简述 | key=value key2=value2

敏感键名（不区分大小写，子串匹配）对应的值会被 redact，不写盘。
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

LOGGER_NAME = "jigsaw.operation"

# 键名包含以下子串之一则整键值对记为 [redacted]（值不写出）
_SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "passwd",
    "pwd",
)

_initialized = False
_lock = threading.RLock()
_fallback_handler: logging.Handler | None = None
_setup_failed = False


def _default_log_dir() -> Path:
    return Path("Log") / "operations"


def _sanitize_message(message: str) -> str:
    if not message:
        return ""
    s = message.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_sensitive_key(key: str) -> bool:
    lower = key.lower()
    return any(frag in lower for frag in _SENSITIVE_KEY_FRAGMENTS)


def _stringify_value(value: Any) -> str:
    if value is None:
        return "未设置"
    s = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 500:
        s = s[:497] + "..."
    return s


def _format_context(context: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(context.keys()):
        if _is_sensitive_key(key):
            parts.append(f"{key}=[redacted]")
        else:
            parts.append(f"{key}={_stringify_value(context[key])}")
    return " ".join(parts)


def _level_label(level: int) -> str:
    name = logging.getLevelName(level)
    if isinstance(name, str):
        return name
    return str(name)


def _build_line(
    source: str,
    message: str,
    context: Mapping[str, Any],
    *,
    level: int = logging.INFO,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lvl = _level_label(level)
    src = _sanitize_message(source) or "未命名来源"
    msg = _sanitize_message(message) or "未命名事件"
    ctx_str = _format_context(context)
    if ctx_str:
        return f"{ts} | {lvl} | {src} | {msg} | {ctx_str}"
    return f"{ts} | {lvl} | {src} | {msg}"


def _make_file_handler(log_dir: Path) -> TimedRotatingFileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "operation.log"
    h = TimedRotatingFileHandler(
        filename=str(log_path),
        when="D",
        interval=7,
        backupCount=100,
        encoding="utf-8",
        delay=False,
    )
    h.setFormatter(logging.Formatter("%(message)s"))
    return h


def _make_stderr_handler() -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[operation_log] %(message)s"))
    return h


def configure_operation_log(
    log_dir: str | Path | None = None,
    *,
    reset: bool = False,
) -> None:
    """
    配置操作日志 logger。应用启动时可显式调用一次；未调用时首次 ``log_operation`` 会按默认目录初始化。

    Args:
        log_dir: 日志目录，默认 ``Log/operations``（相对当前工作目录）。
        reset: 为 True 时清空已有 handler（供单元测试复用）。
    """
    global _initialized, _fallback_handler, _setup_failed

    with _lock:
        logger = logging.getLogger(LOGGER_NAME)
        # 允许写入 DEBUG 及以上，具体是否落盘由每条 log_operation 的 level 决定
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if reset:
            logger.handlers.clear()
            _initialized = False
            _fallback_handler = None
            _setup_failed = False

        if _initialized:
            return

        base = Path(log_dir) if log_dir is not None else _default_log_dir()
        try:
            fh = _make_file_handler(base)
            logger.addHandler(fh)
            _initialized = True
            _setup_failed = False
        except OSError:
            _setup_failed = True
            _fallback_handler = _make_stderr_handler()
            logger.addHandler(_fallback_handler)
            _initialized = True


def log_operation(
    source: str,
    message: str,
    *,
    level: int = logging.INFO,
    **context: Any,
) -> None:
    """
    写入一条操作审计记录。

    Args:
        source: 来源标识，如 ``主界面``、``DryThread``。
        message: 事件简述（换行会被压成空格）。
        level: 事件等级（须用关键字传入，避免与上下文键混淆）。
            常用 ``logging.INFO`` / ``WARNING`` / ``ERROR`` / ``CRITICAL``。
        **context: 可选键值上下文，键名宜稳定（如 lot_id、项目）。
    """
    configure_operation_log()
    line = _build_line(source, message, context, level=level)
    logger = logging.getLogger(LOGGER_NAME)
    try:
        logger.log(level, line)
    except Exception:
        try:
            sys.stderr.write(f"[operation_log] {line}\n")
        except Exception:
            pass


def reset_operation_log_for_tests() -> None:
    """仅测试：清空 handler 与状态，不在磁盘上创建默认日志目录。"""
    global _initialized, _fallback_handler, _setup_failed
    with _lock:
        logger = logging.getLogger(LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        _initialized = False
        _fallback_handler = None
        _setup_failed = False

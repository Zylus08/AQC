"""
aqc/utils/logger.py
===================
Centralised logging configuration for AQC.

Call :func:`setup_logging` once at application start-up (in ``main.py``).
All subsequent ``logging.getLogger(name)`` calls across the codebase will
inherit this configuration.

Author: AQC Team
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_filename: str = "backtest.log",
    log_to_file: bool = True,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure the root AQC logger with console and optional file handlers.

    Parameters
    ----------
    level:
        Minimum log level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
    log_dir:
        Directory in which log files are stored (created if it does not exist).
    log_filename:
        Name of the rotating log file.
    log_to_file:
        If ``True``, attach a :class:`~logging.handlers.RotatingFileHandler`.
    fmt:
        Log record format string.
    datefmt:
        Date format for the log timestamp.
    max_bytes:
        Maximum size of a single log file before rotation.
    backup_count:
        Number of rotated log files to retain.

    Returns
    -------
    logging.Logger
        The configured ``aqc`` root logger.

    Examples
    --------
    >>> from aqc.utils.logger import setup_logging
    >>> logger = setup_logging(level="DEBUG")
    >>> logger.info("AQC framework initialised")
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Get the AQC root logger (all submodule loggers inherit from this)
    root_logger = logging.getLogger("aqc")
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers on re-initialisation
    root_logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --- Rotating file handler ---
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info(
        "AQC logging initialised — level=%s, file=%s",
        level.upper(),
        (Path(log_dir) / log_filename).as_posix() if log_to_file else "disabled",
    )
    return root_logger

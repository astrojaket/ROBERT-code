"""Logging helpers for ROBERT."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a package logger without configuring global logging."""

    return logging.getLogger(name)

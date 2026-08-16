"""Shared utility helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create directory when missing and return Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

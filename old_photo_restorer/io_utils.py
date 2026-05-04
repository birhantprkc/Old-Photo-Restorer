"""File-system, download, and validation helpers."""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from .config import ALLOWED_EXTS, DEFAULT_TIMEOUT_SEC, MIN_MODEL_BYTES

ProgressCallback = Callable[[float, str], None]


def safe_remove(path: str | os.PathLike[str] | None) -> None:
    """Best-effort file cleanup."""

    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def safe_rmtree(path: str | os.PathLike[str] | None) -> None:
    """Best-effort directory cleanup."""

    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def is_allowed_file(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in ALLOWED_EXTS


def file_size_mb(path: str | os.PathLike[str]) -> float:
    try:
        return Path(path).stat().st_size / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def sanitize_stem(stem: str, max_len: int = 48) -> str:
    """Return a filesystem-safe compact filename stem."""

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return (cleaned[:max_len] or "img").strip("_") or "img"


def file_to_path(file_obj: Any) -> str:
    if hasattr(file_obj, "name"):
        return str(file_obj.name)
    return str(file_obj)


def download_file(
    url: str,
    dst_path: str | os.PathLike[str],
    timeout: int = DEFAULT_TIMEOUT_SEC,
    retries: int = 3,
    progress: ProgressCallback | None = None,
) -> None:
    """Download a file with retries and an atomic final move."""

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > MIN_MODEL_BYTES:
        return

    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    headers = {"User-Agent": "old-photo-restorer/1.1"}
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            if progress:
                progress(0.02, "Downloading model weights")

            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", "0") or "0")
                bytes_read = 0
                last_update = time.time()

                with tmp_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        file.write(chunk)
                        bytes_read += len(chunk)

                        if progress and total > 0 and (time.time() - last_update) > 0.25:
                            fraction = min(0.9, 0.02 + 0.88 * (bytes_read / total))
                            progress(fraction, "Downloading model weights")
                            last_update = time.time()

            if tmp_path.stat().st_size <= MIN_MODEL_BYTES:
                raise RuntimeError("Downloaded model file is unexpectedly small.")

            tmp_path.replace(dst)
            if progress:
                progress(0.95, "Finalizing")
            return
        except Exception as exc:
            last_error = exc
            safe_remove(tmp_path)
            if progress:
                progress(0.02, "Retrying download")
            time.sleep(1.25 * (attempt + 1))

    raise RuntimeError(f"Model download failed after {retries} attempts: {last_error}")

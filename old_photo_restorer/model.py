"""GFPGAN model loading and weight management."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

try:
    import sys

    import torchvision.transforms.functional as F

    sys.modules["torchvision.transforms.functional_tensor"] = F
except Exception:
    pass

from gfpgan import GFPGANer

from .config import MIN_MODEL_BYTES, MODEL_NAME, MODEL_URL, resolve_weights_dir
from .io_utils import ProgressCallback, download_file

_RESTORER_LOCK = threading.Lock()
_WEIGHTS_LOCK = threading.Lock()


def restorer_lock() -> threading.Lock:
    return _RESTORER_LOCK


def ensure_weights(progress: ProgressCallback | None = None) -> Path:
    """Ensure GFPGAN weights exist locally and return the model path."""

    weights_dir = resolve_weights_dir()
    model_path = weights_dir / MODEL_NAME

    with _WEIGHTS_LOCK:
        if model_path.exists() and model_path.stat().st_size > MIN_MODEL_BYTES:
            return model_path
        download_file(MODEL_URL, model_path, progress=progress)
        return model_path


@lru_cache(maxsize=1)
def load_restorer_cached(model_path: str) -> GFPGANer:
    """Load GFPGAN once per process."""

    return GFPGANer(
        model_path=model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def get_restorer(progress: ProgressCallback | None = None) -> GFPGANer:
    model_path = ensure_weights(progress=progress)
    return load_restorer_cached(str(model_path))

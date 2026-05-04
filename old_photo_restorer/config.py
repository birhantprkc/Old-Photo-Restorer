"""Application configuration and runtime constants."""

from __future__ import annotations

import os
from pathlib import Path

APP_TITLE = "Old Photo Restorer"
APP_VERSION = "1.1.1"

MODEL_NAME = "GFPGANv1.3.pth"
MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
MIN_MODEL_BYTES = 10_000_000

DEFAULT_TIMEOUT_SEC = 120
MAX_MEGAPIXELS = 4.0
MAX_UPLOAD_MB = 30
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

IMG_HEIGHT_FALLBACK = 560
CMP_HEIGHT_FALLBACK = 300

IS_HF_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))


def resolve_weights_dir() -> Path:
    """Return the first writable directory suitable for model weights."""

    candidates = [Path("/data")]
    for env_name in ("HF_HOME", "HOME"):
        env_value = os.getenv(env_name)
        if env_value:
            candidates.append(Path(env_value))
    candidates.append(Path.cwd())

    for base in candidates:
        try:
            weights_dir = base / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            probe = weights_dir / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return weights_dir
        except Exception:
            continue

    fallback = Path.cwd() / "weights"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

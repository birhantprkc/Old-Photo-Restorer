"""Image conversion, resizing, blending, and export helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import MAX_MEGAPIXELS


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to OpenCV BGR uint8, preserving EXIF orientation."""

    image = ImageOps.exif_transpose(pil_img)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA")

    arr = np.asarray(image)
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return bgr.astype(np.uint8)


def maybe_downscale(bgr: np.ndarray, max_megapixels: float = MAX_MEGAPIXELS) -> tuple[np.ndarray, float]:
    """Downscale large images before inference to avoid memory spikes."""

    height, width = bgr.shape[:2]
    megapixels = (height * width) / 1_000_000.0
    if megapixels <= max_megapixels:
        return bgr, 1.0

    scale = (max_megapixels / megapixels) ** 0.5
    new_width = max(64, int(width * scale))
    new_height = max(64, int(height * scale))
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, float(new_width / width)


def unsharp_mask_rgb(rgb: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return rgb

    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.2, sigmaY=1.2)
    sharpened = cv2.addWeighted(rgb, 1.0 + 1.5 * amount, blurred, -1.5 * amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def blend_rgb(before_rgb: np.ndarray, after_rgb: np.ndarray, original_weight: float) -> np.ndarray:
    original_weight = float(np.clip(original_weight, 0.0, 1.0))
    if before_rgb.shape[:2] != after_rgb.shape[:2]:
        before_rgb = cv2.resize(
            before_rgb,
            (after_rgb.shape[1], after_rgb.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    output = cv2.addWeighted(after_rgb, 1.0 - original_weight, before_rgb, original_weight, 0)
    return np.clip(output, 0, 255).astype(np.uint8)


def inspect_image(image_in: Image.Image | None) -> str:
    if image_in is None:
        return "No image loaded."

    image = ImageOps.exif_transpose(image_in)
    width, height = image.size
    megapixels = (width * height) / 1_000_000.0
    payload: dict[str, Any] = {
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 2),
        "max_megapixels": MAX_MEGAPIXELS,
        "will_downscale": bool(megapixels > MAX_MEGAPIXELS),
    }
    return json.dumps(payload, indent=2)


def save_output(out_pil: Image.Image, out_format: str, jpg_quality: int) -> str:
    suffix = ".png" if out_format == "PNG" else ".jpg"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(handle.name)
    handle.close()

    if out_format == "PNG":
        out_pil.save(tmp_path, format="PNG")
    else:
        quality = int(np.clip(jpg_quality, 30, 95))
        out_pil.convert("RGB").save(tmp_path, format="JPEG", quality=quality, optimize=True)

    return str(tmp_path)

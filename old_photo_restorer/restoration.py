"""Core restoration pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import MAX_MEGAPIXELS, MAX_UPLOAD_MB
from .image_ops import (
    blend_rgb,
    inspect_image,
    maybe_downscale,
    pil_to_bgr,
    save_output,
    unsharp_mask_rgb,
)
from .io_utils import file_size_mb, file_to_path, is_allowed_file, safe_remove, safe_rmtree, sanitize_stem
from .model import get_restorer, restorer_lock


def preset_bundle(preset: str) -> dict[str, float | int]:
    """Map UI presets to deterministic restoration controls."""

    if preset == "Fidelity":
        return {"strength": 0.35, "upscale": 1, "blend": 0.65, "detail": 0.10}
    if preset == "Aggressive":
        return {"strength": 0.80, "upscale": 2, "blend": 0.25, "detail": 0.35}
    return {"strength": 0.55, "upscale": 1, "blend": 0.45, "detail": 0.20}


def enhance_core(
    image_in: Image.Image,
    strength: float,
    upscale: int,
    only_center_face: bool,
    paste_back: bool,
    blend: float,
    detail_boost: float,
    progress_callback=None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Run the complete image restoration path for one PIL image."""

    strength = float(np.clip(strength, 0.0, 1.0))
    upscale = int(np.clip(int(upscale), 1, 4))

    before_bgr = pil_to_bgr(image_in)
    before_bgr, scale_ratio = maybe_downscale(before_bgr, MAX_MEGAPIXELS)

    restorer = get_restorer(progress=progress_callback)
    if progress_callback:
        progress_callback(0.92, "Running restoration")

    with restorer_lock():
        restorer.upscale = upscale
        _, _, output_bgr = restorer.enhance(
            before_bgr,
            has_aligned=False,
            only_center_face=bool(only_center_face),
            paste_back=bool(paste_back),
            weight=strength,
        )

    if output_bgr is None:
        raise RuntimeError("GFPGAN did not return a restored full image. Keep paste_back enabled.")

    before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    after_rgb = unsharp_mask_rgb(after_rgb, detail_boost)
    after_rgb = blend_rgb(before_rgb, after_rgb, blend)

    before_aligned = Image.fromarray(before_rgb)
    out_pil = Image.fromarray(after_rgb)
    metadata: dict[str, Any] = {
        "scale_ratio_after_downscale": round(float(scale_ratio), 4),
        "strength": round(float(strength), 2),
        "upscale": int(upscale),
        "only_center_face": bool(only_center_face),
        "paste_back": bool(paste_back),
        "blend_original_weight": round(float(blend), 2),
        "detail_boost": round(float(detail_boost), 2),
        "output_size": [out_pil.size[0], out_pil.size[1]],
    }
    return before_aligned, out_pil, metadata


def enhance_single(
    image_in: Image.Image | None,
    preset: str,
    strength: float,
    upscale: int,
    only_center_face: bool,
    paste_back: bool,
    blend: float,
    detail_boost: float,
    out_format: str,
    jpg_quality: int,
    prev_tmp_path: str | None,
    progress=None,
):
    """Gradio callback for single-image restoration."""

    safe_remove(prev_tmp_path)

    if image_in is None:
        return None, None, "No image loaded.", None, None, "", "No image loaded.", None

    started = time.time()
    try:
        if progress:
            progress(0.02, desc="Loading model")

        before_aligned, out_pil, metadata = enhance_core(
            image_in=image_in,
            strength=strength,
            upscale=upscale,
            only_center_face=only_center_face,
            paste_back=paste_back,
            blend=blend,
            detail_boost=detail_boost,
            progress_callback=(lambda value, desc: progress(value, desc=desc)) if progress else None,
        )

        if progress:
            progress(0.98, desc="Exporting output")
        out_path = save_output(out_pil, out_format, jpg_quality)
        latency = time.time() - started
        info = f"{latency:.2f}s · {out_pil.size[0]}×{out_pil.size[1]}"

        return (
            out_pil,
            out_path,
            info,
            before_aligned,
            out_pil,
            json.dumps(metadata, indent=2),
            inspect_image(image_in),
            out_path,
        )
    except Exception as exc:
        print(traceback.format_exc())
        message = f"Error: {type(exc).__name__}: {exc}"
        return None, None, message, None, None, "See logs for details.", inspect_image(image_in), None


def batch_enhance(
    files: list[Any],
    preset: str,
    strength: float,
    upscale: int,
    only_center_face: bool,
    paste_back: bool,
    blend: float,
    detail_boost: float,
    out_format: str,
    jpg_quality: int,
    prev_batch_workdir: str | None,
    progress=None,
):
    """Gradio callback for batch restoration and ZIP export."""

    safe_rmtree(prev_batch_workdir)

    if not files:
        return [], None, "No files.", None

    workdir = tempfile.mkdtemp(prefix="old_photo_restorer_batch_")
    outdir = Path(workdir) / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []

    try:
        total = len(files)
        for index, file_obj in enumerate(files, start=1):
            path = file_to_path(file_obj)
            filename = os.path.basename(path)

            if not os.path.exists(path):
                skipped.append("missing_file")
                continue
            if file_size_mb(path) > MAX_UPLOAD_MB:
                skipped.append(f"{filename} (too_large)")
                continue
            if not is_allowed_file(path):
                skipped.append(f"{filename} (unsupported_ext)")
                continue

            try:
                image = Image.open(path)
                image.load()
            except Exception:
                skipped.append(f"{filename} (decode_failed)")
                continue

            base_hint = sanitize_stem(Path(path).stem)
            uid = uuid.uuid4().hex[:8]
            output_stem = f"{base_hint}_{uid}_enhanced"

            try:
                if progress:
                    fraction = 0.05 + 0.85 * (index / max(1, total))
                    progress(fraction, desc=f"Processing {index}/{total}")

                _, out_pil, _ = enhance_core(
                    image_in=image,
                    strength=strength,
                    upscale=upscale,
                    only_center_face=only_center_face,
                    paste_back=paste_back,
                    blend=blend,
                    detail_boost=detail_boost,
                    progress_callback=(lambda value, desc: progress(value, desc=desc)) if progress else None,
                )

                if out_format == "PNG":
                    output_path = outdir / f"{output_stem}.png"
                    out_pil.save(output_path, format="PNG")
                else:
                    quality = int(np.clip(jpg_quality, 30, 95))
                    output_path = outdir / f"{output_stem}.jpg"
                    out_pil.convert("RGB").save(output_path, format="JPEG", quality=quality, optimize=True)

                written.append(str(output_path))
            except Exception:
                print(traceback.format_exc())
                skipped.append(f"{filename} (inference_failed)")

        if not written:
            safe_rmtree(workdir)
            return [], None, "0 processed. All files were skipped or failed.", None

        if progress:
            progress(0.95, desc="Creating ZIP")
        zip_path = Path(workdir) / "results.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for output_path in written:
                archive.write(output_path, arcname=os.path.basename(output_path))

        summary = f"{len(written)} processed"
        if skipped:
            summary += f" · {len(skipped)} skipped: " + "; ".join(skipped[:6])
            if len(skipped) > 6:
                summary += f"; +{len(skipped) - 6} more"

        return written, str(zip_path), summary, workdir
    except Exception as exc:
        print(traceback.format_exc())
        safe_rmtree(workdir)
        return [], None, f"Error: {type(exc).__name__}: {exc}", None


def clear_all(prev_tmp_path: str | None, prev_batch_workdir: str | None):
    """Clear UI state and temporary outputs."""

    safe_remove(prev_tmp_path)
    safe_rmtree(prev_batch_workdir)
    return (
        None,
        None,
        "",
        None,
        None,
        "",
        "No image loaded.",
        None,
        None,
        "",
        None,
        None,
    )

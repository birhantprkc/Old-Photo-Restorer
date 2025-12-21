import os
import sys
import time
import json
import zipfile
import tempfile
import threading
import traceback
import shutil
import uuid
import re
from functools import lru_cache
from typing import List, Tuple, Optional, Any, Dict, Callable

import requests
import numpy as np
import cv2
from PIL import Image, ImageOps

try:
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F
except Exception:
    pass

import gradio as gr
from gfpgan import GFPGANer


APP_TITLE = "Old Photo Restorer"

MODEL_NAME = "GFPGANv1.3.pth"
MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

DEFAULT_TIMEOUT_SEC = 120
MAX_MEGAPIXELS = 4.0

MAX_UPLOAD_MB = 30
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

IMG_HEIGHT_FALLBACK = 560
CMP_HEIGHT_FALLBACK = 300

IS_HF_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
_RESTORER_LOCK = threading.Lock()
_WEIGHTS_LOCK = threading.Lock()


def weights_dir() -> str:
    for candidate in ("/data", os.getenv("HF_HOME"), os.getenv("HOME"), "."):
        if not candidate:
            continue
        try:
            base = candidate if candidate != "." else os.getcwd()
            d = os.path.join(base, "weights")
            os.makedirs(d, exist_ok=True)
            test = os.path.join(d, ".write_test")
            with open(test, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test)
            return d
        except Exception:
            continue
    d = os.path.join(os.getcwd(), "weights")
    os.makedirs(d, exist_ok=True)
    return d


CSS = """
:root{
  --bg0:#070a12;
  --bg1:#0a1024;
  --panel: rgba(255,255,255,0.06);
  --panel2: rgba(255,255,255,0.035);
  --stroke: rgba(255,255,255,0.14);
  --stroke2: rgba(255,255,255,0.10);
}

body{
  background:
    radial-gradient(1200px 700px at 12% 10%, rgba(124,58,237,0.20), transparent),
    radial-gradient(900px 600px at 88% 12%, rgba(59,130,246,0.16), transparent),
    radial-gradient(1000px 600px at 50% 100%, rgba(16,185,129,0.10), transparent),
    linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
}

.gradio-container{
  width: 100% !important;
  max-width: 1760px !important;
  margin: 0 auto !important;
  padding: 10px 16px !important;
}

@media (max-width: 900px){
  .gradio-container{
    padding: 8px 10px !important;
  }
}

.block, .wrap{ gap: 8px !important; }
.gr-form{ gap: 6px !important; }

footer, .footer, .gradio-footer, #footer { display:none !important; }

#hero{
  display:flex;
  align-items:center;
  justify-content:center;
  padding: 10px 0 8px;
}

#titlebar{
  width:100%;
  text-align:center;
  font-size: 22px;
  font-weight: 780;
  letter-spacing: 0.4px;
  color: rgba(255,255,255,0.94);
}

.panel{
  border: 1px solid var(--stroke);
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border-radius: 16px;
  padding: 10px 10px;
  box-shadow: 0 16px 40px rgba(0,0,0,0.28);
  backdrop-filter: blur(10px);
}

.image-panel{
  border: 1px solid var(--stroke2) !important;
  border-radius: 14px !important;
  background: rgba(255,255,255,0.03) !important;
}

.image-panel .gr-image, .image-panel img{
  border-radius: 12px !important;
}

button.primary{
  background: linear-gradient(90deg, rgba(124,58,237,0.95), rgba(59,130,246,0.95)) !important;
  border: 0 !important;
}

button.secondary{
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.05) !important;
}

#img_in, #img_out{
  height: 560px !important;
}

@media (max-width: 1100px){
  #img_in, #img_out{
    height: 420px !important;
  }
}

#img_in .gr-image, #img_out .gr-image{
  height: 100% !important;
}

#img_in .image-container, #img_out .image-container{
  height: 100% !important;
}

#img_in .image-container img, #img_out .image-container img{
  height: 100% !important;
  width: 100% !important;
  object-fit: contain !important;
}

#cmp_before, #cmp_after{
  height: 300px !important;
}

@media (max-width: 1100px){
  #cmp_before, #cmp_after{
    height: 240px !important;
  }
}

#cmp_before .gr-image, #cmp_after .gr-image{
  height: 100% !important;
}

#cmp_before .image-container, #cmp_after .image-container{
  height: 100% !important;
}

#cmp_before .image-container img, #cmp_after .image-container img{
  height: 100% !important;
  width: 100% !important;
  object-fit: contain !important;
}
"""


def safe_remove(path: Optional[str]) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def safe_rmtree(path: Optional[str]) -> None:
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def is_allowed_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in ALLOWED_EXTS


def file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def sanitize_stem(stem: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    stem = stem[:24] if stem else "img"
    return stem


def download_file(
    url: str,
    dst_path: str,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    retries: int = 3,
    progress: Optional[Callable[[float, str], None]] = None,
) -> None:
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 10_000_000:
        return

    tmp_path = dst_path + ".tmp"
    last_err: Optional[Exception] = None
    headers = {"User-Agent": "old-photo-restorer/1.0"}

    for i in range(retries):
        try:
            if progress:
                progress(0.02, "Downloading model weights")
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", "0") or "0")
                read = 0
                t_last = time.time()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        read += len(chunk)
                        if progress and total > 0 and (time.time() - t_last) > 0.25:
                            frac = min(0.9, 0.02 + 0.88 * (read / total))
                            progress(frac, "Downloading model weights")
                            t_last = time.time()
            os.replace(tmp_path, dst_path)
            if progress:
                progress(0.95, "Finalizing")
            return
        except Exception as e:
            last_err = e
            safe_remove(tmp_path)
            if progress:
                progress(0.02, "Retrying download")
            time.sleep(1.25 * (i + 1))

    raise RuntimeError(f"Model download failed: {last_err}")


def ensure_weights(progress: Optional[Callable[[float, str], None]] = None) -> str:
    d = weights_dir()
    model_path = os.path.join(d, MODEL_NAME)
    with _WEIGHTS_LOCK:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 10_000_000:
            return model_path
        download_file(MODEL_URL, model_path, progress=progress)
        return model_path


@lru_cache(maxsize=1)
def load_restorer_cached(model_path: str) -> GFPGANer:
    return GFPGANer(
        model_path=model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def get_restorer(progress: Optional[Callable[[float, str], None]] = None) -> GFPGANer:
    model_path = ensure_weights(progress=progress)
    return load_restorer_cached(model_path)


def preset_bundle(preset: str) -> Dict[str, float]:
    if preset == "Fidelity":
        return dict(strength=0.35, upscale=1, blend=0.65, detail=0.10)
    if preset == "Aggressive":
        return dict(strength=0.80, upscale=2, blend=0.25, detail=0.35)
    return dict(strength=0.55, upscale=1, blend=0.45, detail=0.20)


def apply_preset(preset: str):
    b = preset_bundle(preset)
    return (
        gr.update(value=b["strength"]),
        gr.update(value=b["upscale"]),
        gr.update(value=b["blend"]),
        gr.update(value=b["detail"]),
    )


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode not in ("RGB", "RGBA", "L"):
        pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)

    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return bgr.astype(np.uint8)


def maybe_downscale(bgr: np.ndarray, max_megapixels: float = MAX_MEGAPIXELS) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    mp = (h * w) / 1_000_000.0
    if mp <= max_megapixels:
        return bgr, 1.0

    scale = (max_megapixels / mp) ** 0.5
    new_w = max(64, int(w * scale))
    new_h = max(64, int(h * scale))
    out = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, float(new_w / w)


def unsharp_mask_rgb(rgb: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return rgb
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.2, sigmaY=1.2)
    sharp = cv2.addWeighted(rgb, 1.0 + 1.5 * amount, blurred, -1.5 * amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def blend_rgb(before_rgb: np.ndarray, after_rgb: np.ndarray, blend: float) -> np.ndarray:
    blend = float(np.clip(blend, 0.0, 1.0))
    if before_rgb.shape[:2] != after_rgb.shape[:2]:
        before_rgb = cv2.resize(
            before_rgb,
            (after_rgb.shape[1], after_rgb.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    out = cv2.addWeighted(after_rgb, 1.0 - blend, before_rgb, blend, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def inspect_image(image_in: Optional[Image.Image]) -> str:
    if image_in is None:
        return "No image loaded."
    img = ImageOps.exif_transpose(image_in)
    w, h = img.size
    mp = (w * h) / 1_000_000.0
    info = {
        "width": w,
        "height": h,
        "megapixels": round(mp, 2),
        "max_megapixels": MAX_MEGAPIXELS,
        "will_downscale": bool(mp > MAX_MEGAPIXELS),
    }
    return json.dumps(info, indent=2)


def save_output(out_pil: Image.Image, out_format: str, jpg_quality: int) -> str:
    suffix = ".png" if out_format == "PNG" else ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    if out_format == "PNG":
        out_pil.save(tmp_path, format="PNG")
    else:
        q = int(np.clip(jpg_quality, 30, 95))
        out_pil.convert("RGB").save(tmp_path, format="JPEG", quality=q, optimize=True)

    return tmp_path


def file_to_path(f: Any) -> str:
    if hasattr(f, "name"):
        return str(f.name)
    return str(f)


def make_image(label: str, interactive: bool, height_px: int, elem_id: Optional[str] = None):
    try:
        return gr.Image(type="pil", label=label, interactive=interactive, height=height_px, elem_id=elem_id)
    except TypeError:
        return gr.Image(type="pil", label=label, interactive=interactive, elem_id=elem_id)


def enhance_core(
    restorer: GFPGANer,
    image_in: Image.Image,
    strength: float,
    upscale: int,
    only_center_face: bool,
    paste_back: bool,
    blend: float,
    detail_boost: float,
) -> Tuple[Image.Image, Image.Image, dict]:
    strength = float(np.clip(strength, 0.0, 1.0))
    upscale = int(upscale)
    upscale = 1 if upscale < 1 else (4 if upscale > 4 else upscale)

    before_bgr = pil_to_bgr(image_in)
    before_bgr, scale_ratio = maybe_downscale(before_bgr, MAX_MEGAPIXELS)

    with _RESTORER_LOCK:
        restorer.upscale = upscale
        _, _, out_bgr = restorer.enhance(
            before_bgr,
            has_aligned=False,
            only_center_face=bool(only_center_face),
            paste_back=bool(paste_back),
            weight=strength,
        )

    before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    after_rgb = unsharp_mask_rgb(after_rgb, detail_boost)
    after_rgb = blend_rgb(before_rgb, after_rgb, blend)

    before_aligned = Image.fromarray(before_rgb)
    out_pil = Image.fromarray(after_rgb)

    meta = {
        "scale_ratio_after_downscale": round(float(scale_ratio), 4),
        "strength": round(float(strength), 2),
        "upscale": int(upscale),
        "only_center_face": bool(only_center_face),
        "paste_back": bool(paste_back),
        "blend_original_weight": round(float(blend), 2),
        "detail_boost": round(float(detail_boost), 2),
        "output_size": [out_pil.size[0], out_pil.size[1]],
    }
    return before_aligned, out_pil, meta


def enhance_single(
    image_in: Optional[Image.Image],
    preset: str,
    strength: float,
    upscale: int,
    only_center_face: bool,
    paste_back: bool,
    blend: float,
    detail_boost: float,
    out_format: str,
    jpg_quality: int,
    prev_tmp_path: Optional[str],
    progress=gr.Progress(),
):
    safe_remove(prev_tmp_path)

    if image_in is None:
        return None, None, "No image loaded.", None, None, "", "No image loaded.", None

    t0 = time.time()
    try:
        progress(0.02, desc="Loading model")
        restorer = get_restorer(progress=lambda v, d: progress(v, desc=d))
        progress(0.92, desc="Running restoration")
        before_aligned, out_pil, meta = enhance_core(
            restorer=restorer,
            image_in=image_in,
            strength=strength,
            upscale=upscale,
            only_center_face=only_center_face,
            paste_back=paste_back,
            blend=blend,
            detail_boost=detail_boost,
        )
        progress(0.98, desc="Exporting output")
        out_path = save_output(out_pil, out_format, jpg_quality)
        latency = time.time() - t0
        info = f"{latency:.2f}s · {out_pil.size[0]}×{out_pil.size[1]}"
        return (
            out_pil,
            out_path,
            info,
            before_aligned,
            out_pil,
            json.dumps(meta, indent=2),
            inspect_image(image_in),
            out_path,
        )
    except Exception as e:
        print(traceback.format_exc())
        msg = f"Error: {type(e).__name__}: {e}"
        return None, None, msg, None, None, "See logs for details.", inspect_image(image_in), None


def batch_enhance(
    files: List[Any],
    preset: str,
    strength: float,
    upscale: int,
    only_center_face: bool,
    paste_back: bool,
    blend: float,
    detail_boost: float,
    out_format: str,
    jpg_quality: int,
    prev_batch_workdir: Optional[str],
    progress=gr.Progress(),
):
    safe_rmtree(prev_batch_workdir)

    if not files:
        return [], None, "No files.", None

    workdir = tempfile.mkdtemp(prefix="gfpgan_batch_")
    outdir = os.path.join(workdir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    written: List[str] = []
    skipped: List[str] = []

    try:
        progress(0.02, desc="Loading model")
        restorer = get_restorer(progress=lambda v, d: progress(v, desc=d))

        total = len(files)
        for idx, f in enumerate(files, start=1):
            path = file_to_path(f)

            if not os.path.exists(path):
                skipped.append("missing_file")
                continue

            if file_size_mb(path) > MAX_UPLOAD_MB:
                skipped.append(os.path.basename(path) + " (too_large)")
                continue

            if not is_allowed_file(path):
                skipped.append(os.path.basename(path) + " (unsupported_ext)")
                continue

            try:
                img = Image.open(path)
                img.load()
            except Exception:
                skipped.append(os.path.basename(path) + " (decode_failed)")
                continue

            base_raw = os.path.splitext(os.path.basename(path))[0]
            base_hint = sanitize_stem(base_raw)
            uid = uuid.uuid4().hex[:8]
            out_name = f"{base_hint}_{uid}_enhanced"

            try:
                frac = 0.05 + 0.85 * (idx / max(1, total))
                progress(frac, desc=f"Processing {idx}/{total}")
                _, out_pil, _meta = enhance_core(
                    restorer=restorer,
                    image_in=img,
                    strength=strength,
                    upscale=upscale,
                    only_center_face=only_center_face,
                    paste_back=paste_back,
                    blend=blend,
                    detail_boost=detail_boost,
                )

                if out_format == "PNG":
                    out_path = os.path.join(outdir, f"{out_name}.png")
                    out_pil.save(out_path, format="PNG")
                else:
                    q = int(np.clip(jpg_quality, 30, 95))
                    out_path = os.path.join(outdir, f"{out_name}.jpg")
                    out_pil.convert("RGB").save(out_path, format="JPEG", quality=q, optimize=True)

                written.append(out_path)
            except Exception:
                print(traceback.format_exc())
                skipped.append(os.path.basename(path) + " (inference_failed)")

        if not written:
            safe_rmtree(workdir)
            return [], None, "0 processed. All files were skipped or failed.", None

        progress(0.95, desc="Creating ZIP")
        zip_path = os.path.join(workdir, "results.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in written:
                z.write(p, arcname=os.path.basename(p))

        summary = f"{len(written)} processed"
        if skipped:
            summary += f" · {len(skipped)} skipped"

        return written, zip_path, summary, workdir
    except Exception as e:
        print(traceback.format_exc())
        safe_rmtree(workdir)
        return [], None, f"Error: {type(e).__name__}: {e}", None


def clear_all(prev_tmp_path: Optional[str], prev_batch_workdir: Optional[str]):
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


with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
    gr.HTML(f"<div id='hero'><div id='titlebar'>{APP_TITLE}</div></div>")

    st_prev_tmp_path = gr.State(value=None)
    st_prev_batch_workdir = gr.State(value=None)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1.35, min_width=600):
            with gr.Group(elem_classes=["panel", "image-panel"]):
                image_in = make_image("Input", True, IMG_HEIGHT_FALLBACK, elem_id="img_in")

        with gr.Column(scale=1.35, min_width=600):
            with gr.Group(elem_classes=["panel", "image-panel"]):
                result_img = make_image("Result", False, IMG_HEIGHT_FALLBACK, elem_id="img_out")
                with gr.Row():
                    download = gr.File(label="Download")
                    info = gr.Textbox(label="Info", interactive=False, lines=1)

        with gr.Column(scale=0.75, min_width=380):
            with gr.Group(elem_classes=["panel"]):
                preset = gr.Dropdown(["Fidelity", "Balanced", "Aggressive"], value="Balanced", label="Preset")
                strength = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="Strength")
                upscale = gr.Dropdown([1, 2, 4], value=1, label="Upscale")

                with gr.Row():
                    run_btn = gr.Button("Enhance", variant="primary", elem_classes=["primary"])
                    clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["secondary"])

                with gr.Accordion("Options", open=False):
                    only_center_face = gr.Checkbox(value=False, label="Only center face")
                    paste_back = gr.Checkbox(value=True, label="Paste back")
                    blend = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Blend (original weight)")
                    detail_boost = gr.Slider(0.0, 1.0, value=0.20, step=0.05, label="Detail")
                    out_format = gr.Radio(["PNG", "JPG"], value="PNG", label="Format")
                    jpg_quality = gr.Slider(30, 95, value=90, step=1, label="JPG quality")

                with gr.Accordion("Compare (Before / After)", open=False):
                    with gr.Row():
                        before_cmp = make_image("Before", False, CMP_HEIGHT_FALLBACK, elem_id="cmp_before")
                        after_cmp = make_image("After", False, CMP_HEIGHT_FALLBACK, elem_id="cmp_after")

                with gr.Accordion("Batch", open=False):
                    batch_files = gr.Files(label="Upload files", file_count="multiple")
                    batch_run = gr.Button("Run batch", variant="primary", elem_classes=["primary"])
                    batch_outputs = gr.Files(label="Outputs")
                    batch_zip = gr.File(label="ZIP")
                    batch_summary = gr.Textbox(label="Summary", interactive=False)

                with gr.Accordion("Diagnostics", open=False):
                    preflight = gr.Textbox(label="Preflight (JSON)", value="No image loaded.", lines=6)
                    details = gr.Textbox(label="Details", lines=12)

    demo.queue(max_size=32, default_concurrency_limit=1)

    image_in.change(fn=inspect_image, inputs=[image_in], outputs=[preflight])
    preset.change(fn=apply_preset, inputs=[preset], outputs=[strength, upscale, blend, detail_boost])

    run_btn.click(
        fn=enhance_single,
        inputs=[
            image_in,
            preset,
            strength,
            upscale,
            only_center_face,
            paste_back,
            blend,
            detail_boost,
            out_format,
            jpg_quality,
            st_prev_tmp_path,
        ],
        outputs=[result_img, download, info, before_cmp, after_cmp, details, preflight, st_prev_tmp_path],
    )

    batch_run.click(
        fn=batch_enhance,
        inputs=[
            batch_files,
            preset,
            strength,
            upscale,
            only_center_face,
            paste_back,
            blend,
            detail_boost,
            out_format,
            jpg_quality,
            st_prev_batch_workdir,
        ],
        outputs=[batch_outputs, batch_zip, batch_summary, st_prev_batch_workdir],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[st_prev_tmp_path, st_prev_batch_workdir],
        outputs=[
            result_img,
            download,
            info,
            before_cmp,
            after_cmp,
            details,
            preflight,
            batch_outputs,
            batch_zip,
            batch_summary,
            st_prev_tmp_path,
            st_prev_batch_workdir,
        ],
    )


if __name__ == "__main__":
    demo.launch(show_error=not IS_HF_SPACE)

"""Gradio UI composition and callback wiring."""

from __future__ import annotations

import gradio as gr

from .config import APP_TITLE, CMP_HEIGHT_FALLBACK, IMG_HEIGHT_FALLBACK
from .css import CSS
from .image_ops import inspect_image
from .restoration import batch_enhance, clear_all, enhance_single, preset_bundle


def apply_preset(preset: str):
    bundle = preset_bundle(preset)
    return (
        gr.update(value=bundle["strength"]),
        gr.update(value=bundle["upscale"]),
        gr.update(value=bundle["blend"]),
        gr.update(value=bundle["detail"]),
    )


def make_image(label: str, interactive: bool, height_px: int, elem_id: str | None = None):
    try:
        return gr.Image(type="pil", label=label, interactive=interactive, height=height_px, elem_id=elem_id)
    except TypeError:
        return gr.Image(type="pil", label=label, interactive=interactive, elem_id=elem_id)


def build_demo() -> gr.Blocks:
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
                        paste_back = gr.State(value=True)
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

    return demo

# 🖼️ Old Photo Restorer

A clean **Gradio** app for restoring old, blurry, or low-quality portrait photos using **GFPGAN**.  
It focuses on a simple workflow: upload an image, choose a restoration preset, compare before/after, and export the result.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF7A18)
![Model](https://img.shields.io/badge/Model-GFPGAN%20v1.3-6f42c1)
![License](https://img.shields.io/badge/License-Apache--2.0-2ea44f)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/tarekmasryo/Old-Photo-Restorer)

---

## ✨ Preview

![App Preview](assets/Example.png)

---

## 🚀 What this app does

- 🧑‍🦳 Restores faces in old or degraded photos using **GFPGAN v1.3**
- 🎛️ Provides three predictable presets: `Fidelity`, `Balanced`, and `Aggressive`
- 🔍 Shows **before / after** comparison inside the app
- 📦 Supports **single-image** and **batch** processing
- 🗜️ Exports processed batch results as a ZIP file
- 🧪 Includes lightweight diagnostics for image size and downscaling behavior
- 🧹 Keeps downloaded model weights and generated outputs out of Git history

---

## 🧠 How it works

The app runs a practical face-restoration pipeline:

1. Load the uploaded image and normalize orientation.
2. Downscale very large images to reduce memory spikes.
3. Restore faces with GFPGAN.
4. Optionally blend the restored result with the original image.
5. Apply a small detail boost when requested.
6. Export the final image as PNG or JPG.

GFPGAN mainly improves **faces**. Backgrounds, text, clothing, and highly damaged regions may not improve as much.

---

## 🧩 Presets

| Preset | Best for | Behavior |
|---|---|---|
| `Fidelity` | Photos where identity preservation matters most | Lighter restoration, more original texture retained |
| `Balanced` | General old-photo cleanup | Good default balance between restoration and realism |
| `Aggressive` | Very degraded faces | Stronger restoration, higher artifact risk |

---

## ⚙️ Quickstart

```bash
git clone https://github.com/tarekmasryo/Old-Photo-Restorer.git
cd Old-Photo-Restorer

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python app.py
```

Open the local Gradio URL printed in the terminal.

---

## 🗂️ Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions checks
├── app.py                         # Gradio entry point
├── old_photo_restorer/
│   ├── config.py                  # App constants and runtime paths
│   ├── css.py                     # Custom UI styling
│   ├── image_ops.py               # Image conversion, resizing, blending, export
│   ├── io_utils.py                # Validation, cleanup, and downloads
│   ├── model.py                   # GFPGAN weight loading and model cache
│   ├── restoration.py             # Single and batch restoration callbacks
│   └── ui.py                      # Gradio layout and event wiring
├── assets/
│   └── Example.png
├── scripts/
│   └── smoke_test.py
├── CHANGELOG.md
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧪 Local checks

The smoke test checks repository structure and Python syntax without downloading model weights.

```bash
python -m pip install ruff
python -m ruff check .
python -m compileall -q .
python scripts/smoke_test.py
```

---

## 📌 Model weights

Model weights are downloaded on first run and should **not** be committed.

```gitignore
weights/
outputs/
*.pth
```

---

## 🔒 Privacy and usage notes

Uploaded images are processed inside the running app session. The project does not intentionally store user uploads, but generated files may exist temporarily during the session for download/export.

This app uses **GFPGAN** as a third-party pre-trained restoration model. Restoration quality is best-effort and depends on input image quality, face visibility, compression, blur, lighting, and occlusion.

Do not use restored outputs for identity-critical, legal, medical, forensic, or other high-stakes decisions.

---

## ⚠️ Limitations

- This is a **best-effort restoration tool**, not a forensic image-reconstruction system.
- Results depend on blur, compression, scratches, lighting, face size, and occlusion.
- Very small or heavily occluded faces may produce artifacts.
- Not intended for high-stakes or identity-critical use.

---

## ✅ Good use cases

- Restoring family portraits
- Improving old profile photos
- Cleaning low-resolution face images for personal archives
- Demonstrating an image-restoration workflow with a simple Gradio UI

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE).

**Author:** Tarek Masryo

"""Entry point for the Old Photo Restorer Gradio app."""

from old_photo_restorer.config import IS_HF_SPACE
from old_photo_restorer.ui import build_demo

demo = build_demo()


if __name__ == "__main__":
    demo.launch(show_error=not IS_HF_SPACE)

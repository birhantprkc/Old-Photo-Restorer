"""Lightweight repository smoke test.

This intentionally avoids importing heavy runtime dependencies or downloading model
weights. It validates the repo structure and compiles Python source files.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    ROOT / "app.py",
    ROOT / "requirements.txt",
    ROOT / "README.md",
    ROOT / "old_photo_restorer" / "config.py",
    ROOT / "old_photo_restorer" / "ui.py",
    ROOT / "old_photo_restorer" / "restoration.py",
]


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    for path in [ROOT / "app.py", *sorted((ROOT / "old_photo_restorer").glob("*.py"))]:
        py_compile.compile(str(path), doraise=True)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()

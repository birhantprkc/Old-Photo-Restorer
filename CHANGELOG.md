# Changelog

## 1.1.1

- Made paste-back behavior internal to keep full-image restoration outputs stable.
- Added an explicit guard for empty GFPGAN full-image outputs.
- Expanded CI to install dependencies, lint with Ruff, compile source, and run the smoke test.
- Added privacy and usage notes to the README.

## 1.1.0

- Split the single large app file into a small package with clear modules.
- Added safer runtime cleanup and batch-processing summaries.
- Switched OpenCV dependency to the headless package for server-style deployments.
- Added a lightweight smoke test and CI syntax check.
- Rewrote the README with clearer setup, limitations, and deployment notes.

## 1.0.0

- Initial Gradio app for old-photo face restoration with GFPGAN.

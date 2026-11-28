# Open an image in the OA Viewer in Python

`oa.Viewer.preview` is the level-0 surface: one call opens a preconfigured application window with interactive pan, zoom, and per-channel inspection.

## Run

```bash
python3 sdk/py/tutorials/vision/tu_viewer_image/tu_viewer_image.py [image]
```

The path is optional and defaults to `sdk/asset/image/coverMl.jpg`.

## 1. One call, one window

The whole program is argument handling plus a single call:

```python
oa.Viewer.preview(path)
```

`preview` owns the window, the presentation runtime, input handling, and the render loop. An optional second argument takes an `oa.ViewerConfig` for window size, title, and mode.

## 2. Controls

| Input | Action |
|---|---|
| `LMB` / `MMB` / wheel | Pan and zoom |
| `+` / `-` | Zoom in / out |
| `0` / `F` | Fit to window |
| `9` | 100% zoom |
| `1`–`4` | R / G / B / A channel view |
| `5` | RGB composite |
| `B` | Toggle dark / gradient background |
| `G` | Toggle workspace grid |
| `Q` / `Esc` | Quit |

## What you built

An interactive GPU image viewer in one statement. The full source is `sdk/py/tutorials/vision/tu_viewer_image/tu_viewer_image.py`; the Rust form is at [walkthrough-rust](#walkthrough-rust).

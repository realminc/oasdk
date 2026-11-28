# Open an image in the OA Viewer in Rust

`oa::Viewer::preview` is the level-0 surface: one call opens a preconfigured application window with interactive pan, zoom, and per-channel inspection.

## Run

```bash
cargo run -p sdk --release --example tu_viewer_image -- [image.jpg]
```

The path is optional and defaults to `asset/image/coverMl.jpg` through `oa::Path::asset_rel`.

## 1. One call, one window

The whole program is argument handling plus a single viewer call. `preview` owns the window, the presentation runtime, input handling, and the render loop:

```rust
oa::Viewer::preview(path)
```

The OpenCV parallel is exact: `cv::imread` plus `cv::imshow` plus `cv::waitKey` becomes `Viewer::preview(path)`.

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

An interactive GPU image viewer in one statement. The full source is `sdk/rs/tutorials/vision/tu_viewer_image/tu_viewer_image.rs`; the Python form is at [walkthrough-python](#walkthrough-python), and the headless render/sink version is the data augmentation walkthrough.

# Render and save an image headlessly in Rust

Decode an image to a GPU texture, copy and fill it on device, and write results to PNG without opening a window. This is the renderer/sink split: the producer is identical to the interactive viewer, only the terminal node differs.

## Run

```bash
cargo run -p sdk --release --example tu_viewer_image_headless -- [input.jpg] [output.png]
```

Both arguments are optional: input defaults to `asset/image/coverMl.jpg`, output to `/tmp/oa_viewport_batch.png`. Expected output:

```
tu_viewer_image_headless: asset/image/coverMl.jpg → /tmp/oa_viewport_batch.png
OK: 1672×941  asset/image/coverMl.jpg → /tmp/oa_viewport_batch.png
OK: Blit → /tmp/oa_viewport_batch_blit.png
OK: clear(red) → /tmp/oa_viewport_batch_clear.png
```

## 1. Headless means a compute-only runtime

No surface, no `VK_KHR_swapchain`: `Runtime::current()` resolves the thread default compute runtime, and any GPU the driver admits works.

```rust
let _runtime = Runtime::current()?;
```

That is the whole bring-up. In the interactive tutorial the sink is a swapchain present inside `oa::Viewer`; here the sink is a filesystem write.

## 2. The producer is `texture_from_file`

`oa::render::texture_from_file` decodes the image and uploads it as a `Texture`, ready for device work:

```rust
let tex = texture_from_file(&in_path)?;
```

## 3. The sink is `save_texture_file`

`save_texture_file` reads the texture back, encodes it, and writes the file:

```rust
save_texture_file(&tex, &out_path, 90)?;
```

The last argument is the encoder quality. The file in this directory's `asset/` was written by exactly this call:

![Saved headlessly](asset/oa-viewer-headless.png)

## 4. `blit` and `clear` are device-side smoke tests

`oa::render::blit` copies one texture into another on the GPU; saving the result round-trips the original pixels. `oa::render::clear` fills a texture with a solid color:

```rust
let clone = blit(&tex)?;
let red = clear(&clone, oa::Color::new(0.95, 0.10, 0.10, 1.0))?;
```

![A texture cleared to red](asset/oa-viewer-headless-clear.png)

## What you built

One binary that runs on any admitted GPU with or without a display server: decode, GPU copy, GPU fill, and PNG write, all on a compute-only runtime. The full source is `sdk/rs/tutorials/vision/tu_viewer_image_headless/tu_viewer_image_headless.rs`; the Python image operations are at [walkthrough-python](#walkthrough-python).

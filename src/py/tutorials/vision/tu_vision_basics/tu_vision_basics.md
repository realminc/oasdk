# Decode and transform an image in Python

Decode a JPEG to an `oa.Image`, resize and adjust it on the GPU, and read the result back through `oa.Matrix`. Image operations resolve the thread default runtime.

## Run

```bash
python3 sdk/py/tutorials/vision/tu_vision_basics/tu_vision_basics.py
```

The input path resolves under `OA_ASSET` (default `asset`), using `image/coverMl.jpg` from the SDK asset root. Expected output:

```
[1, 3, 90, 160] min=0.0516 max=1.1500
```

## 1. Decode

`oa.image.decode_file` takes the path, a pixel format token, and a runtime. `None` resolves `oa.Runtime.current()`. The format names the decoded channel layout (`"gray"`, `"rgb"`, `"rgba"`, `"bgr"`, `"bgra"`, `"gray_alpha"`), not the file codec:

```python
image = oa.image.decode_file(path, "rgb", None)
```

## 2. Transform on device

`oa.image.resize` and `oa.image.brightness_contrast` run on the GPU and return new `oa.Image` values:

```python
small = oa.image.resize(image, 160, 90)
adjusted = oa.image.brightness_contrast(small, 0.05, 1.1)
```

## 3. Read back through `oa.Matrix`

`image.as_matrix()` views the pixels as an NCHW `oa.Matrix`; `matrix.read_f32()` copies the values to the host:

```python
matrix = adjusted.as_matrix()
values = matrix.read_f32()

assert matrix.shape == [1, 3, 90, 160]
assert len(values) == 3 * 90 * 160
```

The contrast factor pushes `max` past `1.0`, which the assertion output shows directly: `min=0.0516 max=1.1500`.

## What you built

Decode, resize, tone adjust, and verified readback on the default runtime in ten lines. The full source is `sdk/py/tutorials/vision/tu_vision_basics/tu_vision_basics.py`; the Rust headless render path is at [walkthrough-rust](#walkthrough-rust).

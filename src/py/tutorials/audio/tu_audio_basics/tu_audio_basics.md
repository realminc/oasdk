# Decode, normalize, and save audio in Python

The smallest complete OA audio pipeline: decode a FLAC file, normalize its
peak level, inspect the result, and write it back as WAV. This walkthrough
follows the same experiment as the C++ `tuCoreEngine` tutorial. The output
below is from the Python run with OA 0.18.0 on Intel Iris Xe with
Mesa 26.3.1.

## Run in Python

From the OASDK root, with the sibling `oa-dev` checkout present, install its
native package into an activated virtual environment with `maturin`.

```sh
(cd ../oa-dev && maturin develop --release)

python src/py/tutorials/audio/tu_audio_basics/tu_audio_basics.py
```

## 1. Resolve the input and output paths

`OA_ASSET` points at the SDK asset tree and `OA_VAR` at the output tree, so
generated media stays out of the source checkout. The tutorial creates the
output directory before writing.

```python
asset = os.path.join(os.environ.get("OA_ASSET", "asset"), "audio/oaNarration.flac")
output = os.path.join(
	os.environ.get("OA_VAR", "var"),
	"tutorial/audio/oaNarrationNormalized.wav",
)
os.makedirs(os.path.dirname(output), exist_ok=True)
```

## 2. Decode the file

`oa.audio.decode_file` probes the container, decodes every stream, and
uploads planar FP32 samples. The `None` runtime argument resolves the
calling thread's default runtime. The bundled narration clip is mono at
24 kHz.

```python
audio = oa.audio.decode_file(asset, None)
```

## 3. Normalize the peak level

`oa.audio.normalize` rescales the signal so its global peak sits at the
decibel target, `−3.0` here.

```python
clean = oa.audio.normalize(audio, -3.0)
```

## 4. Save and inspect

`oa.audio.save_wav_f32` encodes the FP32 buffers as interleaved WAV and
writes the file. `channels`, `sample_rate`, and `duration_seconds` report
the decoded shape of the normalized result.

```python
oa.audio.save_wav_f32(output, clean)
print(f"Channels: {clean.channels}  Sample-rate: {clean.sample_rate}  "
	  f"Duration: {clean.duration_seconds:.2f} s")
print(f"Saved: {os.path.abspath(output)}")
```

```text
Channels: 1  Sample-rate: 24000  Duration: 12.52 s
Saved: var/tutorial/audio/oaNarrationNormalized.wav
```

The Rust program decodes and normalizes the same clip and prints the same
summary.

## Source

[Python example](tu_audio_basics.py) ·
[Full Rust walkthrough](../../../../rs/tutorials/audio/tu_audio_basics/tu_audio_basics.md)

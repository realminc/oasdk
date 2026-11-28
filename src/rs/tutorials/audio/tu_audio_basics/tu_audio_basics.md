# Decode, normalize, and save audio in Rust

The smallest complete OA audio pipeline: decode a FLAC file, normalize its
peak level, inspect the result, and write it back as WAV. This walkthrough
follows the same experiment as the C++ `tuCoreEngine` tutorial. The output
below is from the Rust run with OA 0.18.0 on Intel Iris Xe with Mesa 26.3.1.

## Run in Rust

From the OASDK root, with `oa-dev` as a sibling checkout:

```sh
cargo run --release --example tu_audio_basics
```

## 1. Resolve the input and output paths

`Path::asset_rel` resolves a file inside the SDK asset tree, honoring
`OA_ASSET` when set. `Path::var_rel` resolves under the `var` output tree so
generated media stays out of the source checkout. The tutorial creates the
output directory before writing.

```rust
let asset = Path::asset_rel("audio/oaNarration.flac");
let output = Path::var_rel("tutorial/audio/oaNarrationNormalized.wav");
if let Some(parent) = output.parent() {
	std::fs::create_dir_all(&parent)?;
}
```

## 2. Decode the file

`decode_file` probes the container, decodes every stream, and uploads planar
FP32 samples on this thread's default runtime. The bundled narration clip is
mono at 24 kHz.

```rust
let audio = decode_file(&asset)?;
```

## 3. Normalize the peak level

`Audio::normalize` rescales the signal so its global peak sits at
`target_db`. The default `Peak` mode matches the Python call.

```rust
let clean = audio.normalize(NormalizeAudioConfig {
	target_db: -3.0,
	..NormalizeAudioConfig::default()
})?;
```

## 4. Save and inspect

`save_wav_f32` encodes the FP32 buffers as interleaved WAV and writes the
file. `channels`, `sample_rate`, and `duration_seconds` report the decoded
shape of the normalized result.

```rust
save_wav_f32(&output, &clean)?;
println!(
	"Channels: {}  Sample-rate: {}  Duration: {:.2} s",
	clean.channels(),
	clean.sample_rate(),
	clean.duration_seconds()
);
println!("Saved: {}", output.display());
```

```text
Channels: 1  Sample-rate: 24000  Duration: 12.52 s
Saved: var/tutorial/audio/oaNarrationNormalized.wav
```

The Python program decodes and normalizes the same clip and prints the same
summary.

## Source

[Rust example](tu_audio_basics.rs) ·
[Full Python walkthrough](../../../../py/tutorials/audio/tu_audio_basics/tu_audio_basics.md)

# Open audio in the OA Viewer in Rust

`ViewerMode::Audio` turns the level-0 viewer into a player: waveform, spectrum, or Mel-spectrogram visualization plus playback through the default output device.

## Run

```bash
cargo run -p sdk --release --example tu_viewer_audio -- [file.flac] [waveform|spectrum|mel]
```

Both arguments are optional: the file defaults to `asset/audio/oaNarration.flac` and the view to `waveform`.

## 1. `ViewerConfig` selects the mode

The config record carries the mode, the media path, the audio visualization, and the window geometry. `Viewer::new(config).run_standalone()` owns everything else:

```rust
let config = ViewerConfig {
	mode: ViewerMode::Audio,
	path: path.to_owned(),
	audio_view,
	title: "oa::Viewer · Audio".to_owned(),
	width: 960,
	height: 360,
	..ViewerConfig::default()
};

Viewer::new(config).run_standalone()
```

`audio_view` accepts `ViewerAudioView::Waveform`, `Spectrum`, or `Mel`; the second CLI argument maps onto the same variants.

## 2. Controls

| Input | Action |
|---|---|
| `Space` | Play / pause |
| `L` | Toggle loop |
| `W` / `S` / `M` | Waveform / spectrum / Mel view |
| `←` / `→` | Seek ∓ / ± 5 s |
| `↑` / `↓` | Seek ± / ∓ 25 s |
| `Q` / `Esc` | Quit |

## What you built

A GPU-rendered audio player with three analysis views from one config struct. The full source is `sdk/rs/tutorials/vision/tu_viewer_audio/tu_viewer_audio.rs`; the Python form is at [walkthrough-python](#walkthrough-python).

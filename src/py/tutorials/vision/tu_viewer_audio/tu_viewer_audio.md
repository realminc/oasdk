# Open audio in the OA Viewer in Python

`oa.ViewerMode.Audio` turns the level-0 viewer into a player: waveform, spectrum, or Mel-spectrogram visualization plus playback through the default output device.

## Run

```bash
python3 sdk/py/tutorials/vision/tu_viewer_audio/tu_viewer_audio.py [audio] [waveform|spectrum|mel]
```

Both arguments are optional: the file defaults to `sdk/asset/audio/oaNarration.flac` and the view to `waveform`.

## 1. Fill an `oa.ViewerConfig`

The Python binding exposes the config as a mutable record. Set the mode, the audio visualization, and the window title, then call `preview` with both the path and the config:

```python
config = oa.ViewerConfig()
config.mode = oa.ViewerMode.Audio
config.audio_view = {
	"spectrum": oa.ViewerAudioView.Spectrum,
	"mel": oa.ViewerAudioView.Mel,
}.get(view, oa.ViewerAudioView.Waveform)
config.title = "OA Viewer · Audio"
oa.Viewer.preview(path, config)
```

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

A GPU-rendered audio player with three analysis views from one config record. The full source is `sdk/py/tutorials/vision/tu_viewer_audio/tu_viewer_audio.py`; the Rust form is at [walkthrough-rust](#walkthrough-rust).

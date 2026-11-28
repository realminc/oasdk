//! Decode, normalize, and save one audio file with the OA audio API.
//!
//! Mirrors sdk/cpp/tutorials/core/tuCoreEngine.cpp, the basic Audio DSP
//! tutorial.

use anyhow::{Context, Result};
use oa::Path;
use oa::audio::{NormalizeAudioConfig, decode_file, save_wav_f32};

fn main() -> Result<()> {
	let asset = Path::asset_rel("audio/oaNarration.flac");
	let output = Path::var_rel("tutorial/audio/oaNarrationNormalized.wav");
	if let Some(parent) = output.parent() {
		std::fs::create_dir_all(&parent)
			.with_context(|| format!("failed to create {}", parent.display()))?;
	}

	let audio = decode_file(&asset)?;
	let clean = audio.normalize(NormalizeAudioConfig {
		target_db: -3.0,
		..NormalizeAudioConfig::default()
	})?;
	save_wav_f32(&output, &clean)?;

	println!(
		"Channels: {}  Sample-rate: {}  Duration: {:.2} s",
		clean.channels(),
		clean.sample_rate(),
		clean.duration_seconds()
	);
	println!("Saved: {}", output.display());
	Ok(())
}

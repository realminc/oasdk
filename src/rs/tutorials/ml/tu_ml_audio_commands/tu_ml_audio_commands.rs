//! Train and inspect the TensorFlow mini Speech Commands workload with OA.

use std::{
	path::{Path, PathBuf},
	rc::Rc,
};

use anyhow::{Context, Result, ensure};
use oa::{
	Audio, AudioChannelLayout, Image, Matrix, Runtime,
	audio::{ResampleConfig, StftConfig},
	image::{ImageFormat, ImageLayout, InterpolationMode},
	ml::{
		self, AdamW, GradientTape, ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar,
		TrainingSummary, nn,
	},
};
use oads::{MiniSpeechCommands, MiniSpeechCommandsConfig};
use serde_json::json;

const COMMANDS: [&str; 8] = ["down", "go", "left", "no", "right", "stop", "up", "yes"];
const SAMPLES: usize = 16_000;
const FRAMES: usize = 124;
const BINS: usize = 129;
const SIDE: usize = 32;
const SEED: u64 = 42;
const CHUNK: usize = 64;

fn arg(flag: &str) -> Option<String> {
	std::env::args().skip_while(|value| value != flag).nth(1)
}

fn shuffled(count: usize, seed: u64) -> Result<Vec<usize>> {
	let mut order = (0..count as i64).collect::<Vec<_>>();
	oa::data::shuffle(&mut order, seed)?;
	Ok(order.into_iter().map(|index| index as usize).collect())
}

fn waveforms(dataset: &MiniSpeechCommands, rows: &[usize]) -> Result<Vec<f32>> {
	let indices = rows.iter().map(|&index| index as u32).collect::<Vec<_>>();
	let indices = Matrix::new_from_slice([indices.len()], &indices)?;
	Ok(
		dataset
			.samples()
			.reshape([8000, SAMPLES])?
			.gather(&indices)?
			.read_f32()?,
	)
}

fn spectrogram(rows: &[f32], runtime: &Runtime) -> Result<Matrix> {
	let batch = rows.len() / SAMPLES;
	let audio = Audio::new_from_planar_f32(
		runtime,
		rows,
		batch,
		SAMPLES as u32,
		AudioChannelLayout::Unknown,
	)?;
	let magnitude = audio.stft(StftConfig {
		fft_size: 256,
		hop_size: 128,
		win_size: 255,
		center: false,
		..Default::default()
	})?;
	ensure!(
		magnitude.shape() == [batch, FRAMES, BINS],
		"unexpected STFT geometry"
	);
	Ok(magnitude.reshape([batch, 1, FRAMES, BINS])?)
}

fn resize(magnitude: Matrix) -> Result<Matrix> {
	let image = Image::new(magnitude, ImageLayout::Nchw, ImageFormat::Gray)?;
	Ok(
		image
			.resize(SIDE as u32, SIDE as u32, InterpolationMode::Bilinear)?
			.into_matrix(),
	)
}

fn extract(
	dataset: &MiniSpeechCommands,
	indices: &[usize],
	runtime: &Runtime,
	fit: bool,
) -> Result<(Vec<f32>, f32, f32)> {
	let mut features = Vec::with_capacity(indices.len() * SIDE * SIDE);
	let (mut sum, mut squared) = (0.0, 0.0);
	for rows in indices.chunks(CHUNK) {
		let magnitude = spectrogram(&waveforms(dataset, rows)?, runtime)?;
		if fit {
			for value in magnitude.read_f32()? {
				let value = f64::from(value);
				sum += value;
				squared += value * value;
			}
		}
		features.extend(resize(magnitude)?.read_f32()?);
	}
	let count = (indices.len() * FRAMES * BINS) as f64;
	let mean = sum / count;
	let inv_std = (squared / count - mean * mean)
		.max(0.0)
		.sqrt()
		.max(1e-7)
		.recip();
	Ok((features, mean as f32, inv_std as f32))
}

fn classifier(runtime: &Runtime) -> Result<nn::Sequential> {
	let mut model = nn::Sequential::new();
	model.add_named(
		"conv1",
		Rc::new(nn::Conv2d::new_with_seed_on(
			runtime, 1, 32, 3, 1, 0, 1, SEED,
		)?),
	)?;
	model.add_named("relu1", Rc::new(nn::Relu::new()))?;
	model.add_named(
		"conv2",
		Rc::new(nn::Conv2d::new_with_seed_on(
			runtime,
			32,
			64,
			3,
			1,
			0,
			1,
			SEED + 1,
		)?),
	)?;
	model.add_named("relu2", Rc::new(nn::Relu::new()))?;
	model.add_named("pool", Rc::new(nn::MaxPool2d::new(2)?))?;
	model.add_named(
		"drop1",
		Rc::new(nn::Dropout::new_with_seed(0.25, SEED + 4)?),
	)?;
	model.add_named("flatten", Rc::new(nn::Flatten::new(1, -1)))?;
	model.add_named(
		"dense1",
		Rc::new(nn::Linear::new_with_seed_on(
			runtime,
			64 * 14 * 14,
			128,
			SEED + 2,
		)?),
	)?;
	model.add_named("relu3", Rc::new(nn::Relu::new()))?;
	model.add_named("drop2", Rc::new(nn::Dropout::new_with_seed(0.5, SEED + 5)?))?;
	model.add_named(
		"dense2",
		Rc::new(nn::Linear::new_with_seed_on(
			runtime,
			128,
			COMMANDS.len(),
			SEED + 3,
		)?),
	)?;
	Ok(model)
}

fn argmax(scores: &[f32]) -> usize {
	scores
		.iter()
		.enumerate()
		.rev()
		.max_by(|a, b| a.1.total_cmp(b.1))
		.expect("nonempty scores")
		.0
}

type Confusion = [[u32; COMMANDS.len()]; COMMANDS.len()];

fn evaluate(
	model: &impl Module,
	features: &[f32],
	labels: &[u32],
	batch: usize,
	confusion: bool,
) -> Result<(f64, f64, Confusion)> {
	let mut correct = 0_u64;
	let mut loss_sum = 0.0;
	let mut table = [[0; COMMANDS.len()]; COMMANDS.len()];
	for (block, targets) in labels.chunks(batch).enumerate() {
		let begin = block * batch * SIDE * SIDE;
		let end = begin + targets.len() * SIDE * SIDE;
		let input = Matrix::new_from_slice([targets.len(), 1, SIDE, SIDE], &features[begin..end])?;
		let target = Matrix::new_from_slice([targets.len()], targets)?;
		let logits = model.forward(&input)?;
		correct += u64::from(logits.categorical_accuracy_count(&target)?.read::<u32>()?[0]);
		loss_sum +=
			f64::from(ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0]) * targets.len() as f64;
		if confusion {
			let scores = logits.read_f32()?;
			let (rows, remainder) = scores.as_chunks::<{ COMMANDS.len() }>();
			ensure!(
				rows.len() == targets.len() && remainder.is_empty(),
				"invalid logits shape"
			);
			for (&label, row) in targets.iter().zip(rows) {
				table[label as usize][argmax(row)] += 1;
			}
		}
	}
	if confusion {
		ensure!(
			(0..COMMANDS.len())
				.map(|i| u64::from(table[i][i]))
				.sum::<u64>()
				== correct,
			"confusion matrix disagrees with OA accuracy"
		);
	}
	Ok((
		correct as f64 / labels.len() as f64,
		loss_sum / labels.len() as f64,
		table,
	))
}

fn predict(
	model: &impl Module,
	waveform: &[f32],
	runtime: &Runtime,
	mean: f32,
	inv_std: f32,
) -> Result<Vec<f32>> {
	let mut features = resize(spectrogram(waveform, runtime)?)?.read_f32()?;
	oasdk::ml::audio::standardize(&mut features, mean, inv_std);
	let input = Matrix::new_from_slice([1, 1, SIDE, SIDE], &features)?;
	Ok(model.forward(&input)?.softmax(-1)?.read_f32()?)
}

fn read_waveform(path: &Path) -> Result<Vec<f32>> {
	let mut mono = oa::audio::decode_file(path)?.to_mono()?;
	if mono.sample_rate() != SAMPLES as u32 {
		mono = mono.resample(ResampleConfig {
			out_rate: SAMPLES as u32,
			..Default::default()
		})?;
	}
	let mut waveform = mono.as_matrix().read_f32()?;
	waveform.resize(SAMPLES, 0.0);
	Ok(waveform)
}

fn new_optimizer(model: &impl Module) -> Result<AdamW> {
	Ok(AdamW::new_with_hyperparameters(
		model.all_parameters()?,
		0.001,
		0.9,
		0.999,
		1e-7,
		0.0,
	)?)
}

fn load_classifier(
	directory: &Path,
	runtime: &Runtime,
) -> Result<(nn::Sequential, AdamW, [f32; 2])> {
	let model = classifier(runtime)?;
	model.eval();
	let mut optimizer = new_optimizer(&model)?;
	ml::load_checkpoint(directory.join("commands.oam"), &model, &mut optimizer)?;
	let stats: serde_json::Value =
		serde_json::from_slice(&std::fs::read(directory.join("preprocessing.json"))?)?;
	let mean = stats["mean"].as_f64().context("missing mean")? as f32;
	let inv_std = stats["inv_std"]
		.as_f64()
		.context("missing inverse standard deviation")? as f32;
	Ok((model, optimizer, [mean, inv_std]))
}

fn report_prediction(scores: &[f32]) {
	let choice = argmax(scores);
	println!(
		"WAV prediction: {} ({:.2}%)",
		COMMANDS[choice],
		100.0 * scores[choice]
	);
}

fn main() -> Result<()> {
	let epochs = arg("--epochs").map_or(Ok(10), |value| value.parse::<usize>())?;
	let batch = arg("--batch").map_or(Ok(64), |value| value.parse::<usize>())?;
	let step_limit = arg("--steps")
		.map(|value| value.parse::<usize>())
		.transpose()?;
	ensure!(
		epochs >= 1 && (1..=128).contains(&batch) && step_limit.is_none_or(|steps| steps >= 3),
		"require epochs >= 1, batch in 1..128, and at least three replay smoke steps"
	);
	let mut builder = Runtime::builder();
	if let Some(index) = arg("--device") {
		builder = builder.devices(oa::DeviceSelection::Index(index.parse()?));
	}
	let runtime = builder.build()?;
	Runtime::set_default(runtime.clone());
	let wav = arg("--wav").map(PathBuf::from);
	if let Some(directory) = arg("--load") {
		let path = wav.as_deref().context("--load requires --wav")?;
		let (model, _, [mean, inv_std]) = load_classifier(Path::new(&directory), &runtime)?;
		let scores = predict(&model, &read_waveform(path)?, &runtime, mean, inv_std)?;
		report_prediction(&scores);
		return Ok(());
	}
	let output = arg("--output").map_or_else(
		|| PathBuf::from("var/tutorials/audio_commands/rust"),
		PathBuf::from,
	);
	std::fs::create_dir_all(&output)?;
	let dataset = MiniSpeechCommands::load_on(&runtime, &MiniSpeechCommandsConfig::default())?;
	ensure!(
		dataset.samples().shape() == [8000, 1, SAMPLES],
		"unexpected OADS pack dimensions"
	);
	let labels = dataset
		.labels()
		.read_f32()?
		.into_iter()
		.map(|value| value as u32)
		.collect::<Vec<_>>();
	ensure!(
		(0..COMMANDS.len() as u32)
			.all(|label| labels.iter().filter(|&&value| value == label).count() == 1000),
		"unexpected command distribution in the OADS pack"
	);
	let order = shuffled(8000, SEED)?;
	let (train_idx, held_out) = order.split_at(6400);
	let (val_idx, test_idx) = held_out.split_at(800);
	println!("Split: 6400 train · 800 validation · 800 test");
	let clips = waveforms(&dataset, &train_idx[..9])?;
	let preview = json!({
		"commands": COMMANDS, "labels": train_idx[..9].iter().map(|&i| labels[i]).collect::<Vec<_>>(),
		"spectrograms": spectrogram(&clips, &runtime)?.read_f32()?, "waveforms": clips,
	});
	std::fs::write(output.join("preview.json"), serde_json::to_vec(&preview)?)?;
	if std::env::args().any(|value| value == "--preview-only") {
		return Ok(());
	}
	let (mut train_features_host, mean, inv_std) = extract(&dataset, train_idx, &runtime, true)?;
	let (mut val_features, _, _) = extract(&dataset, val_idx, &runtime, false)?;
	let (mut test_features, _, _) = extract(&dataset, test_idx, &runtime, false)?;
	for split in [
		&mut train_features_host,
		&mut val_features,
		&mut test_features,
	] {
		oasdk::ml::audio::standardize(split, mean, inv_std);
	}
	let labels_for = |indices: &[usize]| {
		indices
			.iter()
			.map(|&index| labels[index])
			.collect::<Vec<_>>()
	};
	let (train_labels_host, val_labels, test_labels) = (
		labels_for(train_idx),
		labels_for(val_idx),
		labels_for(test_idx),
	);
	let no_index = test_idx
		.iter()
		.copied()
		.find(|&index| COMMANDS[labels[index] as usize] == "no")
		.context("missing test clip for no")?;
	let waveform = waveforms(&dataset, &[no_index])?;
	drop(dataset);
	let source_label = wav.is_none().then_some("no");
	let path = match wav {
		Some(path) => path,
		None => {
			let path = output.join("sample.wav");
			let clip = Audio::new_from_planar_f32(
				&runtime,
				&waveform,
				1,
				SAMPLES as u32,
				AudioChannelLayout::Mono,
			)?;
			oa::audio::save_wav_f32(&path, &clip)?;
			path
		}
	};
	let waveform = read_waveform(&path)?;
	let model = classifier(&runtime)?;
	ensure!(
		model.num_parameters()? == 1_625_608,
		"unexpected CNN parameter count"
	);
	model.train(true);
	runtime.checkpoint()?.wait()?;
	let mut optimizer = new_optimizer(&model)?;
	let windows = train_idx.len() / batch;
	let steps = step_limit.unwrap_or(epochs * windows);
	let clip_feature_len = SIDE * SIDE;
	let mut loss_metric = LossMetric::default();
	let mut progress = ProgressBar::default();
	let mut summary_callback = TrainingSummary::default();
	let mut training = ItTraining::new(
		&runtime,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: steps as u64,
			steps_per_epoch: windows as u64,
			batch_size: batch as u64,
			source_unit: "clip".into(),
			timer_name: "commands_training_step".into(),
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary_callback);

	let mut history = Vec::new();
	let mut best_loss = f64::INFINITY;
	let mut stale_epochs = 0;
	let mut order = Vec::new();
	// The per-step accuracy count is recorded as an extra program output and
	// read back after each step — the same readback the loss metric uses.
	let step_correct = std::cell::RefCell::new(Option::<oa::Matrix>::None);
	let mut epoch_correct = 0_u64;
	let mut epoch_clips = 0_u64;
	for step in 0..steps {
		training.record_source_units(batch as u64);
		if step % windows == 0 {
			order = shuffled(train_idx.len(), SEED + (step / windows) as u64)?;
		}
		assert!(training.step(
			|| {
				let begin = (step % windows) * batch;
				let rows = &order[begin..begin + batch];
				let mut values = Vec::with_capacity(batch * clip_feature_len);
				let mut labels = Vec::with_capacity(batch);
				for &row in rows {
					values.extend_from_slice(
						&train_features_host[row * clip_feature_len..(row + 1) * clip_feature_len],
					);
					labels.push(train_labels_host[row]);
				}
				let input = Matrix::new_from_slice([batch, 1, SIDE, SIDE], &values)?;
				let target = Matrix::new_from_slice([batch], &labels)?;
				Ok((input, target))
			},
			|(input, target)| {
				let tape = GradientTape::new();
				let logits = model.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
				tape.backward(&loss)?;
				*step_correct.borrow_mut() =
					Some(oa::matrix::categorical_accuracy_count(&logits, &target)?);
				Ok(loss)
			},
		)?);
		if let Some(count) = &*step_correct.borrow() {
			epoch_correct += count.read::<u32>()?[0] as u64;
		}
		epoch_clips += batch as u64;
		let snapshot = training.snapshot();
		if snapshot.is_epoch_boundary() {
			model.eval();
			let (val_accuracy, val_loss, _) = evaluate(&model, &val_features, &val_labels, batch, false)?;
			model.train(true);
			let seconds = snapshot.epoch_elapsed().as_secs_f64();
			println!(
				"  {}/{} — {seconds:.1}s {:.0} ms/step · accuracy: {:.4} · loss: {:.4} · val_accuracy: {val_accuracy:.4} · val_loss: {val_loss:.4}",
				snapshot.step_in_epoch(),
				snapshot.steps_in_epoch(),
				seconds * 1e3 / snapshot.steps_in_epoch().max(1) as f64,
				epoch_correct as f64 / epoch_clips.max(1) as f64,
				snapshot.epoch_mean_loss(),
			);
			history.push(json!({
				"loss": snapshot.epoch_mean_loss(), "accuracy": epoch_correct as f64 / epoch_clips as f64,
				"val_loss": val_loss, "val_accuracy": val_accuracy,
			}));
			if val_loss < best_loss {
				best_loss = val_loss;
				stale_epochs = 0;
			} else {
				stale_epochs += 1;
				if stale_epochs >= 2 {
					println!("Early stopping after two epochs without validation loss improvement");
					break;
				}
			}
			epoch_correct = 0;
			epoch_clips = 0;
		}
	}
	let diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let fallbacks = training.program_capture_fallback_count();
	let result = training.finish()?;
	let diagnostics = diagnostics.context("training capture did not produce a compiled step")?;
	ensure!(
		diagnostics.command_cache_hit_count() > 0 && fallbacks == 0,
		"expected cached training replay, observed {fallbacks} fallbacks"
	);
	ensure!(
		u64::from(optimizer.step_count()) == result.step_count(),
		"optimizer update count mismatch"
	);
	println!(
		"Compiled step: {} nodes · {} cache hits",
		diagnostics.node_count(),
		diagnostics.command_cache_hit_count()
	);
	model.eval();
	let (accuracy, loss, confusion) = evaluate(&model, &test_features, &test_labels, batch, true)?;
	let scores = predict(&model, &waveform, &runtime, mean, inv_std)?;
	ensure!(
		(scores.iter().sum::<f32>() - 1.0).abs() < 2e-5,
		"invalid softmax"
	);
	println!(
		"Test accuracy: {:.2}% · test loss: {loss:.6}",
		accuracy * 100.0
	);
	report_prediction(&scores);

	let checkpoint = output.join("commands.oam");
	ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let stats = output.join("preprocessing.json");
	std::fs::write(
		&stats,
		serde_json::to_vec(&json!({"mean": mean, "inv_std": inv_std}))?,
	)?;
	let (reloaded, reloaded_optimizer, [saved_mean, saved_inv_std]) =
		load_classifier(&output, &runtime)?;
	ensure!(
		predict(&reloaded, &waveform, &runtime, saved_mean, saved_inv_std)? == scores,
		"reload changed inference"
	);
	ensure!(
		reloaded_optimizer.step_count() == optimizer.step_count(),
		"reload changed optimizer step count"
	);
	let results = json!({
		"history": history, "test_accuracy": accuracy, "test_loss": loss, "confusion": confusion,
		"scores": scores, "waveform": waveform, "source_label": source_label,
		"steps": result.step_count(), "cache_hits": diagnostics.command_cache_hit_count(),
	});
	std::fs::write(output.join("results.json"), serde_json::to_vec(&results)?)?;
	Ok(())
}

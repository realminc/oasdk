#!/usr/bin/env python3
"""OA Tutorial — speech commands · mel features + CNN (autograd).

Mirrors the Rust tu_audio_commands tutorial and its canonical
`oasdk::ml::audio` recipe. Decodes one-second clips, extracts mel features
for a whole batch in one mel_spectrogram dispatch, and trains a small CNN
classifier with ItTraining capture/replay. The dataset loader, model
composition, and constants live here as plain host-side Python over the
shipped `oa.audio`/`oa.ml.nn` surface.

The tutorial runs against the TensorFlow mini_speech_commands corpus
(~8,000 one-second clips): fetch the pack through `oads`/`oadspython` or
the upstream archive and extract it under the shared OA data root
(`OA_DATA_DIR`, else `~/.oads`) so the tree sits at
mini_speech_commands/1.0.0/mini_speech_commands, or pass --dataset to any
<label>/*.wav root.
"""

import argparse
import math
import os
import pathlib
import sys

import oa
import oa.ml
import oa.ml.nn as nn
import oa.audio

# ── Canonical workload (mirrors src/rs/lib/ml/audio.rs) ───────────────────────

COMMAND_LABELS = ("down", "go", "left", "no", "right", "stop", "up", "yes")
CLIP_SAMPLE_RATE = 16_000
SAMPLES_PER_CLIP = 16_000
MEL_FFT_SIZE = 1_024
MEL_HOP_SIZE = 256
MEL_BANDS = 40
RNG_SEED = 20_260_818
TRAINING_EPOCHS = 10
BATCH_SIZE = 8
MEL_CHUNK_CLIPS = 512


def load_clips(dataset_dir: pathlib.Path, runtime: oa.Runtime) -> list:
	"""Decode every clip under `<dir>/<label>/` in canonical label order.

	Clips are sorted per label for deterministic ordering, decoded to mono,
	resampled to 16 kHz when needed, and zero-padded or trimmed to one
	second. Returns host `(samples, label)` pairs; mel features are produced
	per batch through `mel_features`.
	"""
	clips = []
	for label, name in enumerate(COMMAND_LABELS):
		label_dir = dataset_dir / name
		if not label_dir.is_dir():
			raise ValueError(
				f"speech-commands label directory missing: {label_dir}"
			)
		for path in sorted(label_dir.glob("*.wav")):
			mono = oa.audio.decode_file(str(path), runtime).to_mono()
			if mono.sample_rate != CLIP_SAMPLE_RATE:
				mono = mono.resample(CLIP_SAMPLE_RATE)
			samples = mono.as_matrix().read_f32()
			samples = (samples + [0.0] * SAMPLES_PER_CLIP)[:SAMPLES_PER_CLIP]
			clips.append((samples, label))
	if not clips:
		raise ValueError(
			f"speech-commands dataset at {dataset_dir} contains no wav clips"
		)
	return clips


def mel_features(clips: list, indices: list, runtime: oa.Runtime) -> oa.Matrix:
	"""Compute one mel feature per selected clip in a single dispatch.

	Clips pack as planar channels, so one `mel_spectrogram` call returns
	`[batch, mels, frames]`; the result is reshaped to `[batch, 1, mels,
	frames]` NCHW for convolution.
	"""
	if not indices:
		raise ValueError("mel_features requires at least one clip index")
	planar = []
	for index in indices:
		planar.extend(clips[index][0])
	batch = oa.Audio.new_from_planar_f32(
		planar, len(indices), CLIP_SAMPLE_RATE, "unknown", runtime
	)
	features = batch.mel_spectrogram(
		fft_size=MEL_FFT_SIZE,
		hop_size=MEL_HOP_SIZE,
		num_mels=MEL_BANDS,
		log_scale=True,
	)
	batch_count, mels, frames = features.shape
	return features.reshape([batch_count, 1, mels, frames])


def labels(clips: list, indices: list, runtime: oa.Runtime) -> oa.Matrix:
	"""U32 class targets for the selected clips."""
	return oa.Matrix.new_from_u32(
		[len(indices)], [clips[index][1] for index in indices], runtime
	)


def _parameters_of(module) -> list:
	getter = getattr(module, "all_parameters", None) or module.parameters
	return list(getter())


class CommandsCnn:
	"""TF-tutorial-mirrored classifier composed from `oa.ml.nn` primitives.

	`Conv2d(1→8, 3×3) → ReLU → MaxPool → Conv2d(8→16, 3×3) → ReLU → MaxPool
	→ Flatten → Linear(→64) → ReLU → Dropout(0.25) → Linear(→classes)`.
	Two stride-2 pools quarter each spatial axis before the flatten.
	"""

	def __init__(
		self,
		runtime: oa.Runtime,
		mel_bands: int,
		mel_frames: int,
		num_classes: int,
	):
		flat = 16 * (mel_bands // 4) * (mel_frames // 4)
		if flat == 0:
			raise ValueError("commands CNN requires mel extents of at least 4×4")
		self.conv1 = nn.Conv2d(1, 8, 3, padding=1, seed=RNG_SEED, runtime=runtime)
		self.relu = nn.Relu()
		self.pool = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(
			8, 16, 3, padding=1, seed=RNG_SEED + 1, runtime=runtime
		)
		self.flatten = nn.Flatten(1, -1)
		self.dense1 = nn.Linear(flat, 64, RNG_SEED + 2, runtime=runtime)
		self.dropout = nn.Dropout(0.25, RNG_SEED + 3)
		self.dense2 = nn.Linear(
			64, num_classes, RNG_SEED + 4, runtime=runtime
		)

	def forward(self, input: oa.Matrix) -> oa.Matrix:
		out = self.conv1.forward(input)
		out = self.pool.forward(self.relu.forward(out))
		out = self.conv2.forward(out)
		out = self.pool.forward(self.relu.forward(out))
		out = self.flatten.forward(out)
		out = self.relu.forward(self.dense1.forward(out))
		out = self.dropout.forward(out)
		return self.dense2.forward(out)

	def train(self) -> None:
		self.dropout.train()

	def eval(self) -> None:
		self.dropout.eval()

	def all_parameters(self) -> list:
		return (
			_parameters_of(self.conv1)
			+ _parameters_of(self.conv2)
			+ _parameters_of(self.dense1)
			+ _parameters_of(self.dense2)
		)


def _num_parameters(model) -> int:
	return sum(p.data().num_elements for p in model.all_parameters())


def _evaluate(runtime, model, features, labels, mel_bands, mel_frames, chunk):
	"""Chunked held-out pass: mean accuracy and clip-mean loss, run eagerly
	between training steps."""
	clip_feature_len = mel_bands * mel_frames
	correct = 0
	loss_sum = 0.0
	for begin in range(0, len(labels), chunk):
		end = min(begin + chunk, len(labels))
		input_matrix = oa.Matrix.new_from_f32(
			[end - begin, 1, mel_bands, mel_frames],
			features[begin * clip_feature_len : end * clip_feature_len], runtime=runtime)
		target = oa.Matrix.new_from_u32([end - begin], labels[begin:end], runtime)
		logits = model.forward(input_matrix)
		correct += oa.matrix.categorical_accuracy_count(logits, target).read_u32()[0]
		loss_sum += oa.ml.loss.cross_entropy(logits, target).read_f32()[0] * (end - begin)
	total = max(1, len(labels))
	return correct / total, loss_sum / total


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="tu_audio_commands",
		description="OA Tutorial — speech commands · mel features + CNN",
	)
	parser.add_argument(
		"--dataset",
		default=None,
		help="speech-commands root containing <label>/*.wav directories",
	)
	parser.add_argument("--epochs", type=int, default=0, help="data passes")
	parser.add_argument(
		"--steps", type=int, default=0, help="training steps (overrides --epochs)"
	)
	parser.add_argument("--batch", type=int, default=0, help="clips per step")
	args = parser.parse_args()
	epochs = args.epochs if args.epochs > 0 else TRAINING_EPOCHS
	batch = args.batch if args.batch > 0 else BATCH_SIZE

	runtime = oa.Runtime()

	oa.Runtime.set_default(runtime=runtime)
	# Default mini_speech_commands tree inside the shared OA data root —
	# OA_DATA_DIR, else ~/.oads — beside the pack's cached zip.
	root = pathlib.Path(os.environ.get("OA_DATA_DIR", pathlib.Path.home() / ".oads"))
	dataset_dir = pathlib.Path(
		args.dataset or root / "mini_speech_commands" / "1.0.0" / "mini_speech_commands"
	)
	if not dataset_dir.is_dir():
		parser.error(
			f"speech-commands dataset not found at {dataset_dir}; extract the "
			"mini_speech_commands pack there (oads or the upstream archive) or "
			"pass --dataset"
		)
	clips = load_clips(dataset_dir, runtime)
	clip_count = len(clips)

	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print("║  OA Tutorial — speech commands · mel features + CNN (autograd)   ║")
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()
	print(
		f"dataset: {dataset_dir} · {clip_count} clips · "
		f"{len(COMMAND_LABELS)} commands "
		f"({SAMPLES_PER_CLIP // CLIP_SAMPLE_RATE} s"
		f"{CLIP_SAMPLE_RATE // 1000}kHz samples each)"
	)
	print(
		f"features: {MEL_BANDS} mel bands · fft {MEL_FFT_SIZE} · "
		f"hop {MEL_HOP_SIZE} — one dispatch per batch"
	)

	# Deterministic split: every 4th clip per sorted label goes to test.
	train_idx = [index for index in range(clip_count) if index % 4 != 3]
	test_idx = [index for index in range(clip_count) if index % 4 == 3]
	# The loader emits clips in label order — unshuffled windows would be
	# single-label blocks and the optimizer collapses at label boundaries.
	train_idx = oa.data.shuffle(train_idx, RNG_SEED)
	print(f"split: {len(train_idx)} train · {len(test_idx)} test\n")

	# Chunked extraction keeps each mel dispatch bounded — one call over a
	# multi-thousand-clip planar upload can exceed device memory or driver
	# watchdog limits. ItTraining.step requires an empty execution session,
	# and the prepare closure may only upload fresh inputs, so features
	# materialize to host once and windows slice host-side per step.
	mel_bands = mel_frames = 0
	train_features_host = []
	chunk = MEL_CHUNK_CLIPS
	for begin in range(0, len(train_idx), chunk):
		features = mel_features(clips, train_idx[begin : begin + chunk], runtime)
		if mel_bands == 0:
			_, _, mel_bands, mel_frames = features.shape
		train_features_host.extend(features.read_f32())
	clip_feature_len = mel_bands * mel_frames
	train_labels_host = labels(clips, train_idx, runtime).read_u32()
	# Standardize with training-set statistics — the host-side equivalent of
	# the TF tutorial's Normalization layer.
	count = len(train_features_host)
	mean = sum(train_features_host) / count
	variance = sum((v - mean) ** 2 for v in train_features_host) / count
	inv_std = 1.0 / (variance**0.5 + 1e-6)
	train_features_host = [(v - mean) * inv_std for v in train_features_host]
	# Held-out split materializes upfront — the per-epoch validation pass
	# runs eagerly between training steps.
	test_features_host = []
	for begin in range(0, len(test_idx), chunk):
		features = mel_features(clips, test_idx[begin : begin + chunk], runtime)
		test_features_host.extend(features.read_f32())
	test_features_host = [(v - mean) * inv_std for v in test_features_host]
	test_labels_host = labels(clips, test_idx, runtime).read_u32()
	num_classes = len(COMMAND_LABELS)
	model = CommandsCnn(runtime, mel_bands, mel_frames, num_classes)
	model.train()
	# Complete asynchronous parameter initialization before isolated capture.
	runtime.checkpoint().wait()

	# TF's Adam default — 1e-2 diverges on the full dataset's early batches.
	learning_rate = 0.001
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=learning_rate)

	print(
		"Model: Conv2d(1→8,3×3)→ReLU→MaxPool→Conv2d(8→16,3×3)→ReLU→MaxPool→"
		"Linear(→64)→Dropout→Linear(→8)"
	)
	print(f"params: {_num_parameters(model)}    Optimizer: AdamW(lr={learning_rate})\n")

	# Full-width windows only — a fixed batch shape keeps the captured
	# training program stable; the tail remainder is dropped.
	windows = max(1, len(train_idx) // batch)
	# The canonical recipe counts data passes like TensorFlow: epochs ×
	# windows-per-pass gives the total step count. --steps overrides it.
	steps = args.steps if args.steps > 0 else epochs * windows
	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			steps_per_epoch=windows,
			batch_size=batch,
			source_unit="clip",
			timer_name="commands_training_step",
			enable_gpu_timing=True,
		), runtime=runtime)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	initial_loss = 0.0
	# The per-step accuracy count is recorded as an extra program output and
	# read back after each step — the same readback the loss metric uses.
	step_correct = []
	epoch_correct = 0
	epoch_clips = 0
	for step in range(steps):
		training.record_source_units(batch)
		# Reshuffle the window order each data pass, as TF's
		# dataset.shuffle does per epoch.
		window_order = oa.data.shuffle(
			list(range(windows)), RNG_SEED + step // windows
		)

		def prepare():
			begin = min(
				window_order[step % windows] * batch,
				max(0, len(train_idx) - batch),
			)
			end = begin + min(batch, len(train_idx))
			return (
				oa.Matrix.new_from_f32(
					[end - begin, 1, mel_bands, mel_frames],
					train_features_host[begin * clip_feature_len : end * clip_feature_len],
					runtime=runtime,
				),
				oa.Matrix.new_from_u32([end - begin], train_labels_host[begin:end], runtime),
			)

		def record(prepared):
			input_matrix, target = prepared
			with oa.ml.GradientTape() as tape:
				logits = model.forward(input_matrix)
				loss = oa.ml.loss.cross_entropy(logits, target)
				tape.backward(loss)
			step_correct.clear()
			step_correct.append(oa.matrix.categorical_accuracy_count(logits, target))
			return loss

		assert training.step(prepare, record)
		if step_correct:
			epoch_correct += step_correct[0].read_u32()[0]
		epoch_clips += min(batch, len(train_idx))
		snap = training.snapshot()
		if snap.step_count == 1:
			initial_loss = snap.last_loss or 0.0
		if snap.is_epoch_boundary:
			model.eval()
			val_accuracy, val_loss = _evaluate(
				runtime,
				model,
				test_features_host,
				test_labels_host,
				mel_bands,
				mel_frames,
				chunk,
			)
			model.train()
			elapsed = snap.epoch_elapsed_secs
			print(
				f"  {snap.step_in_epoch}/{snap.steps_in_epoch} — {elapsed:.1f}s "
				f"{elapsed * 1e3 / max(1, snap.steps_in_epoch):.0f} ms/step · "
				f"accuracy: {epoch_correct / max(1, epoch_clips):.4f} · "
				f"loss: {snap.epoch_mean_loss:.4f} · "
				f"val_accuracy: {val_accuracy:.4f} · val_loss: {val_loss:.4f}"
			)
			epoch_correct = 0
			epoch_clips = 0

	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_training_loss = result.last_loss or 0.0

	# Final held-out evaluation through the same chunked pass the per-epoch
	# validation used.
	model.eval()
	accuracy, evaluation_loss = _evaluate(
		runtime,
		model,
		test_features_host,
		test_labels_host,
		mel_bands,
		mel_frames,
		chunk,
	)

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({num_classes}) = {math.log(num_classes):.4f}"
	)
	print(f"  test accuracy: {accuracy * 100.0:.1f}%")
	print(f"  post-update evaluation loss: {evaluation_loss:.6f}")
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	assert initial_loss > 1.0
	assert final_training_loss < initial_loss
	checkpoint = oa.Path.temp().join("oars_commands").with_extension("oam")
	oa.ml.save_checkpoint(str(checkpoint), model.all_parameters(), optimizer)
	reloaded = CommandsCnn(runtime, mel_bands, mel_frames, num_classes)
	reloaded.eval()
	reloaded_optimizer = oa.ml.AdamW(
		reloaded.all_parameters(), learning_rate=learning_rate
	)
	oa.ml.load_checkpoint( str(checkpoint), reloaded.all_parameters(), reloaded_optimizer, runtime=runtime)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert (
		_evaluate(
			runtime,
			reloaded,
			test_features_host,
			test_labels_host,
			mel_bands,
			mel_frames,
			chunk,
		)[0]
		== accuracy
	)
	oa.Filesystem.remove_file(checkpoint)


if __name__ == "__main__":
	main()

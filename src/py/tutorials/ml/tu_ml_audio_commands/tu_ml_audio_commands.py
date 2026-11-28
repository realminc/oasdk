#!/usr/bin/env python3
"""Train and inspect the TensorFlow mini Speech Commands workload with OA."""

import argparse
import json
import math
from pathlib import Path

import oa
import oa.audio
import oa.ml
import oa.ml.nn as nn
from oads.audio import MiniSpeechCommands

COMMANDS = MiniSpeechCommands.COMMANDS
SAMPLES = 16_000
FRAMES, BINS = 124, 129
SIDE = 32
SEED = 42
CHUNK = 64


def spectrogram(rows, runtime):
	batch = len(rows) // SAMPLES
	audio = oa.Audio.new_from_planar_f32(rows, batch, SAMPLES, "unknown", runtime)
	magnitude = audio.stft(fft_size=256, hop_size=128, win_size=255, center=False)
	assert list(magnitude.shape) == [batch, FRAMES, BINS]
	return magnitude.reshape([batch, 1, FRAMES, BINS])


def resize(magnitude):
	image = oa.Image.new_from_matrix(magnitude, "nchw", "gray")
	return image.resize(SIDE, SIDE, "bilinear").as_matrix()


def extract(dataset, indices, runtime, fit=False):
	features = []
	total = square = 0.0
	for start in range(0, len(indices), CHUNK):
		rows = indices[start:start + CHUNK]
		index = oa.Matrix.new_from_u32([len(rows)], rows, runtime)
		waveforms = dataset.samples.reshape([8000, SAMPLES]).gather(index).read_f32()
		magnitude = spectrogram(waveforms, runtime)
		if fit:
			values = magnitude.read_f32()
			total += math.fsum(values)
			square += math.fsum(value * value for value in values)
		features.extend(resize(magnitude).read_f32())
	if not fit:
		return features
	count = len(indices) * FRAMES * BINS
	mean = total / count
	return features, mean, 1.0 / max(math.sqrt(max(0, square / count - mean * mean)), 1e-7)


class CommandsCnn:
	def __init__(self, runtime):
		self.learned = (
			nn.Conv2d(1, 32, 3, padding=0, seed=SEED, runtime=runtime),
			nn.Conv2d(32, 64, 3, padding=0, seed=SEED + 1, runtime=runtime),
			nn.Linear(64 * 14 * 14, 128, SEED + 2, runtime=runtime),
			nn.Linear(128, len(COMMANDS), SEED + 3, runtime=runtime),
		)
		conv1, conv2, dense1, dense2 = self.learned
		relu = nn.Relu()
		self.dropouts = nn.Dropout(0.25, SEED + 4), nn.Dropout(0.5, SEED + 5)
		self.layers = (
			conv1, relu, conv2, relu, nn.MaxPool2d(2), self.dropouts[0],
			nn.Flatten(1, -1), dense1, relu, self.dropouts[1], dense2,
		)

	def forward(self, value):
		for layer in self.layers:
			value = layer.forward(value)
		return value

	def train(self):
		for layer in self.dropouts:
			layer.train()

	def eval(self):
		for layer in self.dropouts:
			layer.eval()

	def parameters(self):
		return [parameter for layer in self.learned for parameter in (getattr(layer, "all_parameters", None) or layer.parameters)()]


def argmax(row):
	return max(range(len(row)), key=row.__getitem__)


def evaluate(runtime, model, features, labels, batch, confusion=False):
	correct, loss_sum = 0, 0.0
	table = [[0] * len(COMMANDS) for _ in COMMANDS]
	for begin in range(0, len(labels), batch):
		end = min(begin + batch, len(labels))
		inputs = oa.Matrix.new_from_f32([end - begin, 1, SIDE, SIDE],
			features[begin * SIDE**2:end * SIDE**2], runtime)
		targets = oa.Matrix.new_from_u32([end - begin], labels[begin:end], runtime)
		logits = model.forward(inputs)
		hits = logits.categorical_accuracy_count(targets).read_u32()[0]
		correct += hits
		loss_sum += oa.ml.loss.cross_entropy(logits, targets).read_f32()[0] * (end - begin)
		if confusion:
			values = logits.read_f32()
			width = len(COMMANDS)
			for row, label in enumerate(labels[begin:end]):
				table[label][argmax(values[row * width:(row + 1) * width])] += 1
	if confusion:
		assert sum(table[i][i] for i in range(len(COMMANDS))) == correct
	return correct / len(labels), loss_sum / len(labels), table


def predict(model, waveform, runtime, mean, inv_std):
	values = resize(spectrogram(waveform, runtime)).read_f32()
	inputs = oa.Matrix.new_from_f32([1, 1, SIDE, SIDE], [(v - mean) * inv_std for v in values], runtime)
	return model.forward(inputs).softmax(-1).read_f32()


def read_waveform(path, runtime):
	mono = oa.audio.decode_file(str(path), runtime).to_mono()
	if mono.sample_rate != SAMPLES:
		mono = mono.resample(SAMPLES)
	return (mono.as_matrix().read_f32() + [0.0] * SAMPLES)[:SAMPLES]


def new_optimizer(model):
	return oa.ml.AdamW(model.parameters(), learning_rate=0.001, epsilon=1e-7, weight_decay=0.0)


def load_classifier(directory, runtime):
	model = CommandsCnn(runtime)
	model.eval()
	optimizer = new_optimizer(model)
	oa.ml.load_checkpoint(str(directory / "commands.oam"), model.parameters(), optimizer, runtime=runtime)
	return model, optimizer, json.loads((directory / "preprocessing.json").read_text())


def report_prediction(scores):
	print(f"WAV prediction: {COMMANDS[argmax(scores)]} ({max(scores):.2%})")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--device", type=int)
	parser.add_argument("--batch", type=int, default=64)
	parser.add_argument("--steps", type=int, help="limit training updates for a replay smoke run")
	parser.add_argument("--output", type=Path, default=Path("var/tutorials/audio_commands/python"))
	parser.add_argument("--preview-only", action="store_true")
	parser.add_argument("--wav", type=Path, help="WAV to classify after training or with --load")
	parser.add_argument("--load", type=Path, help="load exported model and preprocessing from this directory")
	args = parser.parse_args()
	if args.epochs < 1 or not 1 <= args.batch <= 128 or (args.steps is not None and args.steps < 3):
		parser.error("require epochs >= 1, batch in 1..128, and at least three replay smoke steps")
	if args.load and not args.wav:
		parser.error("--load requires --wav")
	runtime = oa.Runtime(device=args.device)
	oa.Runtime.set_default(runtime=runtime)
	if args.load:
		model, _, stats = load_classifier(args.load, runtime)
		report_prediction(predict(model, read_waveform(args.wav, runtime), runtime, **stats))
		return
	dataset = MiniSpeechCommands.load(runtime, progress=False)
	assert list(dataset.samples.shape) == [8000, 1, SAMPLES]
	raw_labels = dataset.labels.read_f32()
	assert all(value.is_integer() and 0 <= value < len(COMMANDS) for value in raw_labels)
	labels = [int(value) for value in raw_labels]
	assert all(labels.count(label) == 1000 for label in range(len(COMMANDS)))
	order = oa.data.shuffle(list(range(8000)), SEED)
	train_idx, val_idx, test_idx = order[:6400], order[6400:7200], order[7200:]
	print("Split: 6400 train · 800 validation · 800 test", flush=True)
	args.output.mkdir(parents=True, exist_ok=True)
	preview_idx = oa.Matrix.new_from_u32([9], train_idx[:9], runtime)
	waveforms = dataset.samples.reshape([8000, SAMPLES]).gather(preview_idx).read_f32()
	preview = {
		"commands": COMMANDS, "labels": [labels[i] for i in train_idx[:9]],
		"waveforms": waveforms, "spectrograms": spectrogram(waveforms, runtime).read_f32(),
	}
	(args.output / "preview.json").write_text(json.dumps(preview))
	if args.preview_only:
		return

	train_features_host, mean, inv_std = extract(dataset, train_idx, runtime, fit=True)
	val_features = extract(dataset, val_idx, runtime)
	test_features = extract(dataset, test_idx, runtime)
	train_features_host, val_features, test_features = (
		[(v - mean) * inv_std for v in features]
		for features in (train_features_host, val_features, test_features)
	)
	train_labels_host, val_labels, test_labels = (
		[labels[i] for i in split] for split in (train_idx, val_idx, test_idx)
	)
	no_index = next(i for i in test_idx if labels[i] == COMMANDS.index("no"))
	waveform = dataset.samples.slice(0, no_index, no_index + 1).read_f32()
	del dataset
	if args.wav:
		waveform = read_waveform(args.wav, runtime)
	else:
		path = args.output / "sample.wav"
		clip = oa.Audio.new_from_planar_f32(waveform, 1, SAMPLES, "mono", runtime)
		oa.audio.save_wav_f32(str(path), clip)
		waveform = read_waveform(path, runtime)
	model = CommandsCnn(runtime)
	model.train()
	parameters = model.parameters()
	assert sum(p.data().num_elements for p in parameters) == 1_625_608
	runtime.checkpoint().wait()
	optimizer = new_optimizer(model)
	batch = args.batch
	windows = len(train_idx) // batch
	steps = args.steps or args.epochs * windows
	clip_feature_len = SIDE**2
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

	history = []
	best_loss, stale_epochs = math.inf, 0
	# The per-step accuracy count is recorded as an extra program output and
	# read back after each step — the same readback the loss metric uses.
	step_correct = []
	epoch_correct = 0
	epoch_clips = 0
	for step in range(steps):
		if step % windows == 0:
			row_order = oa.data.shuffle(list(range(len(train_idx))), SEED + step // windows)
		training.record_source_units(batch)

		def prepare():
			begin = (step % windows) * batch
			rows = row_order[begin:begin + batch]
			values = [value for row in rows
				for value in train_features_host[row * clip_feature_len:(row + 1) * clip_feature_len]]
			return (
				oa.Matrix.new_from_f32(
					[batch, 1, SIDE, SIDE], values,
					runtime=runtime,
				),
				oa.Matrix.new_from_u32([batch], [train_labels_host[row] for row in rows], runtime),
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

		if not training.step(prepare, record):
			raise RuntimeError("The training iterator stopped before the scheduled update")
		epoch_correct += step_correct[0].read_u32()[0]
		epoch_clips += batch
		snap = training.snapshot()
		if snap.is_epoch_boundary:
			model.eval()
			val_accuracy, val_loss, _ = evaluate(
				runtime,
				model,
				val_features,
				val_labels,
				batch,
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
			history.append({
				"loss": snap.epoch_mean_loss, "accuracy": epoch_correct / epoch_clips,
				"val_loss": val_loss, "val_accuracy": val_accuracy,
			})
			if val_loss < best_loss:
				best_loss, stale_epochs = val_loss, 0
			else:
				stale_epochs += 1
				if stale_epochs >= 2:
					print("Early stopping after two epochs without validation loss improvement")
					break
			epoch_correct = 0
			epoch_clips = 0

	diagnostics = training.program_diagnostics()
	fallbacks = training.program_capture_fallback_count()
	result = training.finish()
	if diagnostics is None or diagnostics.command_cache_hit_count == 0 or fallbacks:
		raise RuntimeError(f"Expected cached training replay, observed {fallbacks} fallbacks")
	assert optimizer.step_count == result.step_count
	print(f"Compiled step: {diagnostics.node_count} nodes · {diagnostics.command_cache_hit_count:,} cache hits")
	model.eval()
	accuracy, loss, confusion = evaluate(runtime, model, test_features, test_labels, batch, confusion=True)
	scores = predict(model, waveform, runtime, mean, inv_std)
	assert abs(sum(scores) - 1.0) < 2e-5
	print(f"Test accuracy: {accuracy:.2%} · test loss: {loss:.6f}")
	report_prediction(scores)

	checkpoint = args.output / "commands.oam"
	oa.ml.save_checkpoint(str(checkpoint), parameters, optimizer)
	stats = args.output / "preprocessing.json"
	stats.write_text(json.dumps({"mean": mean, "inv_std": inv_std}))
	reloaded, reloaded_optimizer, saved = load_classifier(args.output, runtime)
	assert predict(reloaded, waveform, runtime, **saved) == scores
	assert reloaded_optimizer.step_count == optimizer.step_count
	(args.output / "results.json").write_text(json.dumps({
		"history": history, "test_accuracy": accuracy, "test_loss": loss,
		"confusion": confusion, "scores": scores, "waveform": waveform,
		"source_label": None if args.wav else "no", "steps": result.step_count,
		"cache_hits": diagnostics.command_cache_hit_count,
	}))


if __name__ == "__main__":
	main()

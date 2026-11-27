#!/usr/bin/env python3
"""Train and inspect the Fashion-MNIST dense classifier with OA and OADS."""

import argparse
import json
import math
import sys
from pathlib import Path

import oa
from oads.image import FashionMnist


SIDE = 28
PIXELS = SIDE * SIDE
CLASSES = (
	"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
BATCH_SIZE = 32
SEED = 42
BLUE = (0.35, 0.62, 1.0, 1.0)
RED = (1.0, 0.40, 0.43, 1.0)
GRAY = (0.47, 0.47, 0.47, 1.0)


def figure(title, rows, cols, width, height):
	return oa.plot.Figure(oa.plot.FigureConfig(
		title=title, rows=rows, cols=cols, width=width, height=height,
		theme=oa.plot.PlotTheme.Dark,
	))


def image_plot(axes, pixels, *, raw=False):
	# The first preview uses raw intensities; later previews use dark ink on white.
	values = pixels if raw else [1.0 - pixel for pixel in pixels]
	axes.heatmap(values, SIDE, SIDE, oa.plot.HeatmapStyle(
		v_max=255.0 if raw else 1.0, colormap=1 if raw else 3,
		auto_scale=False, show_grid=False, colorbar=raw,
	))
	axes.grid(False)


def argmax(probabilities):
	return max(range(len(CLASSES)), key=probabilities.__getitem__)


def probability_plot(axes, probabilities, label, names=False):
	predicted = argmax(probabilities)
	colors = [
		BLUE if index == label else RED if index == predicted else GRAY
		for index in range(len(CLASSES))
	]
	for color in (GRAY, RED, BLUE):
		axes.bar(
			[value if colors[index] == color else float("nan")
			 for index, value in enumerate(probabilities)],
			oa.plot.BarStyle(color=color),
		)
	axes.categories(list(CLASSES) if names else list(map(str, range(len(CLASSES)))))
	axes.limits(-0.5, 9.5, 0.0, 1.0)
	axes.yticks([0.0, 0.5, 1.0])
	axes.grid(False)
	axes.legend(False)


def prediction_grid(images, labels, probabilities, indices, columns, title):
	rows = len(indices) // columns
	plot = figure(title, rows, 2 * columns, 600 * columns, 312 * rows + 54)
	for position, index in enumerate(indices):
		row, col = divmod(position, columns)
		scores, label = probabilities[index], labels[index]
		predicted = argmax(scores)
		axes = plot.ax(row, 2 * col)
		image_plot(axes, images[index * PIXELS:(index + 1) * PIXELS])
		axes.caption_color(
			f"{CLASSES[predicted]} {scores[predicted]:.0%} ({CLASSES[label]})",
			BLUE if predicted == label else RED,
		)
		probability_plot(plot.ax(row, 2 * col + 1), scores, label)
	return plot


def load_split(runtime, split, count):
	data = FashionMnist.load(runtime, split=split, normalize=False, progress=False)
	raw_labels = data.labels.read_f32()
	if list(data.images.shape) != [count, 1, SIDE, SIDE] or list(data.labels.shape) != [count]:
		raise ValueError(f"Unexpected {split} shapes: {data.images.shape}, {data.labels.shape}")
	if any(value not in range(len(CLASSES)) for value in raw_labels):
		raise ValueError(f"Invalid class index in {split}")
	labels = list(map(int, raw_labels))
	if any(labels.count(index) != count // len(CLASSES) for index in range(len(CLASSES))):
		raise ValueError(f"Unexpected {split} class frequencies")
	print(f"{split}: {data.images.shape}, {len(labels)} labels, {labels[:10]}", flush=True)
	return data.images.reshape([count, PIXELS]), data.labels.reshape([count, 1]), labels


class Classifier:
	def __init__(self, runtime):
		self.flatten = oa.ml.nn.Flatten()
		self.hidden = oa.ml.nn.Linear(PIXELS, 128, seed=SEED, runtime=runtime)
		self.relu = oa.ml.nn.Relu()
		self.output = oa.ml.nn.Linear(128, len(CLASSES), seed=SEED + 1, runtime=runtime)

	def parameters(self):
		return self.hidden.parameters() + self.output.parameters()

	def forward(self, images):
		features = self.flatten.forward(images)
		return self.output.forward(self.relu.forward(self.hidden.forward(features)))


def fit(runtime, model, train_images, train_labels, optimizer, epochs):
	count = train_images.shape[0]
	steps_per_epoch = count // BATCH_SIZE
	training = oa.ml.ItTraining(
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=epochs * steps_per_epoch,
			steps_per_epoch=steps_per_epoch,
			batch_size=BATCH_SIZE,
		),
		runtime=runtime,
	)
	training.add_callback(oa.ml.ProgressBar())
	training.add_callback(oa.ml.TrainingSummary())

	hits = None

	def record(indices):
		nonlocal hits
		images = train_images.gather(indices).div_scalar(255.0).reshape([BATCH_SIZE, SIDE, SIDE])
		labels = train_labels.gather(indices).reshape([BATCH_SIZE]).cast_f32_i32()
		with oa.ml.GradientTape() as tape:
			logits = model.forward(images)
			loss = oa.ml.loss.cross_entropy(logits, labels)
			tape.backward(loss)
		# Retain the captured counter so each replay updates this same handle.
		hits = logits.categorical_accuracy_count(labels)
		return loss

	history = []
	for epoch in range(1, epochs + 1):
		order = oa.data.shuffle(list(range(count)), SEED + epoch - 1)
		correct = 0
		for start in range(0, count, BATCH_SIZE):
			rows = order[start:start + BATCH_SIZE]
			if not training.step(lambda: oa.Matrix.new_from_u32([BATCH_SIZE], rows, runtime), record):
				raise ValueError("Training stopped before the epoch finished")
			correct += hits.read_u32()[0]
		state = training.snapshot()
		metrics = {"epoch": epoch, "loss": state.epoch_mean_loss, "accuracy": correct / count}
		if not math.isfinite(metrics["loss"]):
			raise ValueError(f"Nonfinite training loss at epoch {epoch}")
		history.append(metrics)
		print(f"  mean loss: {metrics['loss']:.5f} · accuracy: {metrics['accuracy']:.2%}", flush=True)

	diagnostics = training.program_diagnostics()
	fallbacks = training.program_capture_fallback_count()
	result = training.finish()
	if diagnostics is None or diagnostics.command_cache_hit_count == 0 or fallbacks:
		raise ValueError(f"Expected cached training replay, observed {fallbacks} capture fallbacks")
	if result.step_count != epochs * steps_per_epoch or optimizer.step_count != result.step_count:
		raise ValueError("AdamW update count disagrees with the training schedule")
	print(f"Compiled step: {diagnostics.node_count} nodes · {diagnostics.command_cache_hit_count:,} cache hits")
	return history


def evaluate(runtime, model, images, labels):
	count = len(labels)
	probabilities = []
	loss_sum, correct = 0.0, 0
	for start in range(0, count, BATCH_SIZE):
		end = min(start + BATCH_SIZE, count)
		targets = oa.Matrix.new_from_u32([end - start], labels[start:end], runtime)
		logits = model.forward(images.slice(0, start, end).div_scalar(255.0).reshape([end - start, SIDE, SIDE]))
		loss = oa.ml.loss.cross_entropy(logits, targets)
		hits = logits.categorical_accuracy_count(targets)
		scores = logits.softmax(-1).read_f32()
		probabilities.extend(scores[offset:offset + len(CLASSES)] for offset in range(0, len(scores), len(CLASSES)))
		loss_sum += (end - start) * oa.ml.metric.scalar_loss(loss)
		correct += hits.read_u32()[0]
	if len(probabilities) != count or any(
		len(row) != len(CLASSES) or not all(0.0 <= value <= 1.0 for value in row)
		or abs(sum(row) - 1.0) > 2e-5
		for row in probabilities
	):
		raise ValueError("Softmax did not return one normalized distribution per test image")
	host_correct = sum(argmax(row) == label for row, label in zip(probabilities, labels))
	host_loss = -sum(math.log(row[label]) for row, label in zip(probabilities, labels)) / count
	if host_correct != correct or abs(host_loss - loss_sum / count) > 2e-5:
		raise ValueError("Device metrics disagree with host metrics from the probability rows")
	return loss_sum / count, correct / count, probabilities


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--epochs", type=int, default=10, help="training epochs (default: 10)")
	parser.add_argument("--output", type=Path, default=Path("var/tutorials/fashion_mnist/python"))
	args = parser.parse_args()
	if args.epochs < 1:
		parser.error("--epochs must be positive")
	args.output.mkdir(parents=True, exist_ok=True)

	# Rust callbacks and Python prints share the same terminal.
	sys.stdout.reconfigure(line_buffering=True)
	runtime = oa.Runtime()

	def save(plot, name):
		plot.save_to(str(args.output / f"{name}.png"), runtime=runtime)

	# 1. Inspect both official OADS splits before scaling.
	train, train_labels, train_classes = load_split(runtime, "train", 60_000)
	test, _, test_classes = load_split(runtime, "test", 10_000)
	if train_classes[:3] != [9, 0, 0] or test_classes[:2] != [9, 2]:
		raise ValueError("Unexpected sample order in the official splits")
	first = train.slice(0, 0, 1).read_f32()
	if min(first) != 0 or max(first) != 255:
		raise ValueError("Expected the first raw image to span 0 to 255")
	raw_plot = figure("First training image · raw pixel values", 1, 1, 407, 400)
	image_plot(raw_plot.ax(0, 0), first, raw=True)
	save(raw_plot, "01_raw_image")

	# 2. Use the same OA scaling for each batch, then preview the first 25 training images.
	preview = train.slice(0, 0, 25).div_scalar(255.0).read_f32()
	if any(abs(raw / 255.0 - scaled) > 1e-7 for raw, scaled in zip(first, preview[:PIXELS])):
		raise ValueError("Raw and normalized pixels disagree")
	grid = figure("First 25 training images", 5, 5, 990, 1135)
	for index in range(25):
		axes = grid.ax(*divmod(index, 5))
		image_plot(axes, preview[index * PIXELS:(index + 1) * PIXELS])
		axes.caption(CLASSES[train_classes[index]])
	save(grid, "02_training_images")

	# 3. Build Flatten → Dense(128) → ReLU → Dense(10) with trainable biases.
	model = Classifier(runtime)
	parameters = model.parameters()
	parameter_count = sum(parameter.data().num_elements for parameter in parameters)
	if len(parameters) != 4 or parameter_count != 101_770:
		raise ValueError(f"Unexpected classifier parameter count: {parameter_count}")
	optimizer = oa.ml.AdamW(parameters, learning_rate=0.001, epsilon=1e-7, weight_decay=0.0)
	print(f"Flatten (784) → Linear (128) → ReLU → Linear (10), {parameter_count:,} parameters", flush=True)

	# 4. Fit on training logits, then evaluate the complete held-out split.
	history = fit(runtime, model, train, train_labels, optimizer, args.epochs)
	plot = figure("Training history", 1, 2, 840, 300)
	for column, name in enumerate(("loss", "accuracy")):
		axes = plot.ax(0, column)
		axes.title(name.capitalize())
		axes.x_label("Epoch")
		axes.plot_xy([row["epoch"] for row in history], [row[name] for row in history])
	save(plot, "03_training_history")
	test_loss, test_accuracy, probabilities = evaluate(runtime, model, test, test_classes)
	print(f"test: loss={test_loss:.5f} accuracy={test_accuracy:.2%}", flush=True)

	# 5. Inspect test probabilities. Captions show predicted class, confidence, and truth.
	print(f"test[0]: {probabilities[0]} → {CLASSES[argmax(probabilities[0])]} ({CLASSES[test_classes[0]]})")
	test_preview = test.slice(0, 0, 15).div_scalar(255.0).read_f32()
	for index in (0, 12):
		save(prediction_grid(test_preview, test_classes, probabilities, [index], 1, f"Test image {index}"),
		     f"04_prediction_{index:02}")
	save(prediction_grid(test_preview, test_classes, probabilities, range(15), 3, "First 15 test predictions"),
	     "05_prediction_grid")

	# A fixed preview can contain only correct predictions. Show the most confident error too.
	errors = (index for index, (row, label) in enumerate(zip(probabilities, test_classes))
	          if argmax(row) != label)
	error = max(errors, key=lambda index: max(probabilities[index]), default=None)
	if error is not None:
		pixels = test.slice(0, error, error + 1).div_scalar(255.0).read_f32()
		save(
			prediction_grid(pixels, [test_classes[error]], [probabilities[error]], [0], 1,
			                f"Most confident error · test image {error}"),
			"05_confident_error",
		)

	# 6. Insert an explicit batch axis and predict test image 1 on its own.
	image = test.slice(0, 1, 2).div_scalar(255.0).reshape([SIDE, SIDE])
	single = image.reshape([1, SIDE, SIDE])
	single_scores = model.forward(single).softmax(-1).read_f32()
	if any(abs(a - b) > 2e-5 for a, b in zip(single_scores, probabilities[1], strict=True)):
		raise ValueError("Singleton prediction disagrees with batched test row 1")
	print(f"test[1]: {image.shape} → {single.shape} → {CLASSES[argmax(single_scores)]}")
	plot = figure("Single image · test image 1", 1, 1, 1100, 330)
	probability_plot(plot.ax(0, 0), single_scores, test_classes[1], names=True)
	save(plot, "06_single_image")

	results = {
		"class_names": CLASSES, "parameter_count": parameter_count, "batch_size": BATCH_SIZE,
		"optimizer_steps": optimizer.step_count, "history": history,
		"test_loss": test_loss, "test_accuracy": test_accuracy, "test_labels": test_classes,
		"probabilities": probabilities, "single_probabilities": single_scores,
	}
	(args.output / "results.json").write_text(json.dumps(results, allow_nan=False) + "\n", encoding="utf-8")
	print(f"Wrote plots and results to {args.output}")
	if args.epochs == 10 and (
		history[-1]["loss"] >= history[0]["loss"] or test_accuracy < 0.85 or test_loss >= 0.6
	):
		raise ValueError("The 10-epoch run missed the Fashion-MNIST convergence gate")


if __name__ == "__main__":
	main()

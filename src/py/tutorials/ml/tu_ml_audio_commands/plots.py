#!/usr/bin/env python3
"""Render the recorded Python or Rust Speech Commands run with OA Plot."""

import argparse
import json
import math
import struct
import wave
from pathlib import Path

import oa
import oa.plot as plot

SAMPLES = 16_000
FRAMES, BINS = 124, 129
BLUE = (0.35, 0.62, 1.0, 1.0)
ORANGE = (1.0, 0.63, 0.35, 1.0)


def figure(title, rows, cols, width, height):
	return plot.Figure(plot.FigureConfig(
		title=title, rows=rows, cols=cols, width=width, height=height,
		theme=plot.PlotTheme.Dark,
	))


def waveform(axes, values):
	# Keep one maximum and minimum per pixel group so brief peaks stay visible.
	times, peaks = [], []
	for start in range(0, len(values), 32):
		samples = values[start:start + 32]
		low, high = min(samples), max(samples)
		for index in sorted((samples.index(low), samples.index(high))):
			times.append((start + index) / SAMPLES)
			peaks.append(samples[index])
	axes.plot_xy(times, peaks, plot.LineStyle(color=BLUE))
	axes.limits(0.0, 1.0, -1.05, 1.05)
	axes.x_label("Time (s)")


def spectrum(axes, values, colorbar=False):
	# Rows run high to low frequency. Compress dynamic range only for display.
	log_values = [math.log10(max(1e-5, values[frame * BINS + freq]))
		for freq in reversed(range(BINS)) for frame in range(FRAMES)]
	axes.heatmap(log_values, BINS, FRAMES, plot.HeatmapStyle(
		colormap=1, show_grid=False, colorbar=colorbar,
	))
	axes.limits(0.0, 124.0, 0.0, 129.0)
	axes.xticks([0, 31, 62, 93, 123], labels=["0", ".25", ".50", ".75", ".98"])
	axes.yticks([0, 32, 64, 96, 128], labels=["0", "2", "4", "6", "8"])
	axes.x_label("Time (s)")
	axes.y_label("Frequency (kHz)")


def audio_file(path, values):
	samples = [round(max(-1, min(1, sample)) * 32767) for sample in values]
	with wave.open(str(path), "wb") as handle:
		handle.setparams((1, 2, SAMPLES, 0, "NONE", "not compressed"))
		handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def render(source, destination, runtime):
	preview = json.loads((source / "preview.json").read_text())
	commands = preview["commands"]
	clips = [preview["waveforms"][i * SAMPLES:(i + 1) * SAMPLES] for i in range(9)]
	spectra = [preview["spectrograms"][i * FRAMES * BINS:(i + 1) * FRAMES * BINS] for i in range(9)]
	labels = [commands[i] for i in preview["labels"]]
	destination.mkdir(parents=True, exist_ok=True)

	def save(chart, name):
		chart.save_to(str(destination / f"{name}.jpg"), runtime=runtime)

	waves = figure("Nine training waveforms", 3, 3, 1120, 720)
	spectrograms = figure("Nine training spectrograms", 3, 3, 1120, 810)
	for i, (clip, spec, label) in enumerate(zip(clips, spectra, labels)):
		row, column = divmod(i, 3)
		axes = waves.ax(row, column)
		axes.title(label)
		waveform(axes, clip)
		axes = spectrograms.ax(row, column)
		axes.title(label)
		spectrum(axes, spec)
		if i < 3:
			audio_file(destination / f"clip_{i + 1}.wav", clip)
	save(waves, "01_waveforms")
	pair = figure(f"From waveform to spectrogram · {labels[0]}", 2, 1, 900, 620)
	waveform(pair.ax(0, 0), clips[0])
	spectrum(pair.ax(1, 0), spectra[0], colorbar=True)
	save(pair, "02_stft")
	save(spectrograms, "03_spectrograms")

	if not (source / "results.json").is_file():
		return
	results = json.loads((source / "results.json").read_text())
	history = results["history"]
	if history:
		title = "Replay check · training and validation" if len(history) == 1 else "Training history"
		chart = figure(title, 1, 2, 960, 360)
		epochs = list(range(1, len(history) + 1))
		for column, name in enumerate(("loss", "accuracy")):
			axes = chart.ax(0, column)
			axes.title(name.capitalize())
			axes.x_label("Epoch")
			if len(history) == 1:
				axes.x_label("")
				values = [history[0][name], history[0][f"val_{name}"]]
				for i, color in enumerate((BLUE, ORANGE)):
					axes.bar([value if i == j else math.nan for j, value in enumerate(values)],
						plot.BarStyle(color=color))
				axes.categories(["Training", "Validation"])
				axes.limits(-0.5, 1.5, 0, 1 if name == "accuracy" else max(values) * 1.2)
				for i, value in enumerate(values):
					axes.annotate(f"{value:.2%}" if name == "accuracy" else f"{value:.4f}",
						i, value + 0.02 * max(values))
			else:
				for key, color, label in ((name, BLUE, "Training"), (f"val_{name}", ORANGE, "Validation")):
					values = [row[key] for row in history]
					axes.plot_xy(epochs, values, plot.LineStyle(color=color, label=label))
					axes.scatter(epochs, values, plot.ScatterStyle(color=color))
				axes.xticks(epochs, decimals=0)
		save(chart, "04_training_history")

	table = results["confusion"]
	count = len(commands)
	chart = figure("Test-set confusion matrix", 1, 1, 660, 600)
	axes = chart.ax(0, 0)
	axes.heatmap([value for row in table for value in row], count, count,
		plot.HeatmapStyle(colormap=1, v_max=max(map(max, table)), auto_scale=False, show_grid=False))
	centers = [index + 0.5 for index in range(count)]
	axes.limits(0.0, count, 0.0, count)
	axes.xticks(centers, labels=commands)
	axes.yticks(centers, labels=commands[::-1])
	axes.grid(False)
	axes.x_label("Prediction")
	axes.y_label("True label")
	for row, values in enumerate(table):
		for column, value in enumerate(values):
			axes.annotate(str(value), column + 0.5, count - row - 0.5)
	save(chart, "05_confusion")

	scores = results["scores"]
	predicted = max(range(count), key=scores.__getitem__)
	source_label = results["source_label"]
	chart = figure(f"WAV inference · predicted {commands[predicted]}", 1, 1, 850, 360)
	axes = chart.ax(0, 0)
	if source_label:
		axes.title(f"True label: {source_label}")
	for color in (BLUE, ORANGE):
		axes.bar([value if (i == predicted) == (color == ORANGE) else math.nan
			for i, value in enumerate(scores)], plot.BarStyle(color=color))
	axes.annotate(f"{scores[predicted]:.2%}", predicted, scores[predicted] + 0.055)
	axes.categories(commands)
	axes.limits(-0.5, count - 0.5, 0, 1)
	axes.y_label("Probability")
	save(chart, "06_wav_prediction")
	audio_file(destination / "inference.wav", results["waveform"])


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("run", type=Path)
	parser.add_argument("destination", type=Path)
	parser.add_argument("--device", type=int)
	args = parser.parse_args()
	render(args.run, args.destination, oa.Runtime(device=args.device))

#!/usr/bin/env python3
"""Fit a one-layer regression model with OA autograd and the ItTraining iterator.

Mirrors sdk/cpp/tutorials/ml/tuMnistClassifierAg.cpp, the basic ML training loop.
"""

import sys

import oa

SAMPLES = 5
EPOCHS = 5
STEPS_PER_EPOCH = 16
STEPS = EPOCHS * STEPS_PER_EPOCH


def main():
	# Rust callbacks and Python prints share the same terminal.
	sys.stdout.reconfigure(line_buffering=True)
	x = oa.Matrix.new_from_f32([SAMPLES, 1], [-2.0, -1.0, 0.0, 1.0, 2.0], None)
	target = oa.Matrix.new_from_f32([SAMPLES, 1], [-3.0, -1.0, 1.0, 3.0, 5.0], None)

	model = oa.ml.nn.Linear(1, 1, seed=0)
	parameters = model.parameters()
	if len(parameters) != 2:
		raise ValueError(f"Expected one weight and one bias, got {len(parameters)}")
	optimizer = oa.ml.AdamW(parameters, learning_rate=0.05, weight_decay=0.0)

	initial_loss = oa.ml.metric.scalar_loss(oa.ml.loss.mse(model.forward(x), target))

	training = oa.ml.ItTraining(
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=STEPS,
			steps_per_epoch=STEPS_PER_EPOCH,
			batch_size=SAMPLES,
		),
	)
	training.add_callback(oa.ml.ProgressBar())
	training.add_callback(oa.ml.TrainingSummary())

	rows = list(range(SAMPLES))

	def record(indices):
		batch = x.gather(indices)
		targets = target.gather(indices)
		with oa.ml.GradientTape() as tape:
			prediction = model.forward(batch)
			loss = oa.ml.loss.mse(prediction, targets)
			tape.backward(loss)
		return loss

	for epoch in range(1, EPOCHS + 1):
		for _ in range(STEPS_PER_EPOCH):
			if not training.step(lambda: oa.Matrix.new_from_u32([SAMPLES], rows, None), record):
				raise ValueError("Training stopped before the schedule finished")
		print(f"  epoch {epoch}: mean loss {training.snapshot().epoch_mean_loss:.5f}", flush=True)

	diagnostics = training.program_diagnostics()
	fallbacks = training.program_capture_fallback_count()
	result = training.finish()
	if diagnostics is None or diagnostics.command_cache_hit_count == 0 or fallbacks:
		raise ValueError(f"Expected cached training replay, observed {fallbacks} capture fallbacks")
	if result.step_count != STEPS or optimizer.step_count != STEPS:
		raise ValueError("AdamW update count disagrees with the training schedule")
	print(f"Compiled step: {diagnostics.node_count} nodes · {diagnostics.command_cache_hit_count:,} cache hits")

	final_loss = oa.ml.metric.scalar_loss(oa.ml.loss.mse(model.forward(x), target))
	values = model.forward(x).read_f32()

	assert final_loss < initial_loss * 0.01
	assert max(abs(a - b) for a, b in zip(values, [-3, -1, 1, 3, 5])) < 0.1

	print(f"loss: {initial_loss:.6f} -> {final_loss:.6f}")
	print(values)


if __name__ == "__main__":
	main()

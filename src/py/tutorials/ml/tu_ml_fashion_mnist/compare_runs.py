#!/usr/bin/env python3
"""Compare matching Rust and Python Fashion-MNIST runs."""

import argparse
import json
import math
from pathlib import Path


def compare(rust, python):
	for name in ("class_names", "parameter_count", "batch_size", "optimizer_steps", "test_labels"):
		if rust[name] != python[name]:
			raise ValueError(f"Runs disagree on {name}")

	for name in ("test_loss", "test_accuracy"):
		if not math.isclose(rust[name], python[name], abs_tol=2e-5, rel_tol=0):
			raise ValueError(f"Runs disagree on {name}: {rust[name]} vs {python[name]}")

	for left, right in zip(rust["history"], python["history"], strict=True):
		if left["epoch"] != right["epoch"] or any(
			not math.isclose(left[name], right[name], abs_tol=2e-5, rel_tol=0)
			for name in ("loss", "accuracy")
		):
			raise ValueError(f"Runs disagree on epoch metrics: {left} vs {right}")

	max_error = 0.0
	for index, (left, right) in enumerate(zip(rust["probabilities"], python["probabilities"], strict=True)):
		error = max(abs(a - b) for a, b in zip(left, right, strict=True))
		if not error <= 2e-5:
			raise ValueError(f"Runs disagree on test row {index}: maximum probability error {error:.3g}")
		max_error = max(max_error, error)
		if left.index(max(left)) != right.index(max(right)):
			raise ValueError(f"Runs disagree on predicted class for test row {index}")

	for left, right in zip(rust["single_probabilities"], python["single_probabilities"], strict=True):
		if not math.isclose(left, right, abs_tol=2e-5, rel_tol=0):
			raise ValueError("Runs disagree on single-image probabilities")

	print(
		f"Rust and Python agree on {len(rust['history'])} epochs, "
		f"{rust['optimizer_steps']:,} optimizer steps and {len(rust['probabilities']):,} test predictions. "
		f"Maximum probability difference: {max_error:.3g}."
	)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("rust", type=Path, nargs="?", default=Path("var/tutorials/fashion_mnist/rust/results.json"))
	parser.add_argument("python", type=Path, nargs="?", default=Path("var/tutorials/fashion_mnist/python/results.json"))
	args = parser.parse_args()
	compare(json.loads(args.rust.read_text(encoding="utf-8")), json.loads(args.python.read_text(encoding="utf-8")))


if __name__ == "__main__":
	main()

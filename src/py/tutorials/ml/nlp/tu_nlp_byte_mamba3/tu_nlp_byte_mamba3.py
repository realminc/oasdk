#!/usr/bin/env python3
"""OA Tutorial — Byte Mamba-3 · all-position LM (autograd)

Mirrors the Rust tu_nlp_byte_mamba3 tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import oa
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical byte Embedding → Mamba-3 + residual → Linear language model composed from `oa.ml.nn` primitives."""

	def __init__(self, runtime):
		self.embedding = nn.Embedding(
			oa.ml.byte.VOCAB_SIZE, _nlp_common.MODEL_WIDTH, _nlp_common.RNG_SEED, runtime,
		)
		self.mamba = _nlp_common.canonical_mamba3(runtime, _nlp_common.RNG_SEED + 1)
		self.head = nn.Linear(
			_nlp_common.MODEL_WIDTH, oa.ml.byte.VOCAB_SIZE, _nlp_common.RNG_SEED + 2,
			runtime=runtime,
		)

	def forward(self, tokens):
		batch, sequence = tokens.shape
		embedded = self.embedding.forward(tokens)
		output = self.mamba.forward(embedded)
		residual = output.reshape(
			[batch * sequence, _nlp_common.MODEL_WIDTH]
		) + embedded.reshape([batch * sequence, _nlp_common.MODEL_WIDTH])
		return self.head.forward(residual)

	def all_parameters(self):
		return (
			_nlp_common.parameters_of(self.embedding)
			+ _nlp_common.parameters_of(self.mamba)
			+ _nlp_common.parameters_of(self.head)
		)



def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Byte Mamba-3 · all-position LM (autograd)",
			description="Byte Embedding → Mamba-3(32,state=32,expand=2) + residual → Linear(32→256)",
			timer_name="byte_mamba3_training_step",
			checkpoint_stem="oars_byte_mamba3",
			learning_rate=0.003,
			expected_loss=0.207_903,
			loss_tolerance=0.001,
			expected_accuracy=0.929_688,
			expected_generation=b"to be that is the question whether tis nobler in the mind to suffer the slings and ar",
		),
		Model,
	)


if __name__ == "__main__":
	main()

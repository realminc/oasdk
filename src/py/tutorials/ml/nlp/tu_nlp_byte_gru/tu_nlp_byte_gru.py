#!/usr/bin/env python3
"""OA Tutorial — Byte GRU · all-position LM (autograd)

Mirrors the Rust tu_nlp_byte_gru tutorial. The Rust model rebuilds its donor
initialization through `new_from_matrices`; that constructor is not part of the
shipped Python surface, so this tutorial uses the seeded `nn.Gru` recipe and
carries its own measured expected values.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import oa
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical byte ByteEmbedding → GRU → Linear language model composed from `oa.ml.nn` primitives."""

	def __init__(self, runtime):
		self.embedding = nn.ByteEmbedding(
			_nlp_common.MODEL_WIDTH, _nlp_common.RNG_SEED, runtime,
		)
		self.recurrent = nn.Gru(
			_nlp_common.MODEL_WIDTH, _nlp_common.HIDDEN_WIDTH,
			seed=_nlp_common.RNG_SEED + 1, runtime=runtime,
		)
		self.head = nn.Linear(
			_nlp_common.HIDDEN_WIDTH, oa.ml.byte.VOCAB_SIZE, _nlp_common.RNG_SEED + 2,
			runtime=runtime,
		)

	def forward(self, tokens):
		batch, sequence = tokens.shape
		output = self.recurrent.forward(self.embedding.forward(tokens))
		return self.head.forward(
			output.reshape([batch * sequence, _nlp_common.HIDDEN_WIDTH])
		)

	def all_parameters(self):
		return (
			_nlp_common.parameters_of(self.embedding)
			+ _nlp_common.parameters_of(self.recurrent)
			+ _nlp_common.parameters_of(self.head)
		)



def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Byte GRU · all-position LM (autograd)",
			description="ByteEmbedding(32) → GRU(64) → Linear(64→256)",
			timer_name="byte_gru_training_step",
			checkpoint_stem="oars_byte_gru",
			learning_rate=0.01,
			expected_loss=0.515_180,
			loss_tolerance=0.001,
			expected_accuracy=0.851_562_5,
			expected_generation=b"to be ther the mind to be ther the mind to be ther the mind to be ther the mind to be",
		),
		Model,
	)


if __name__ == "__main__":
	main()

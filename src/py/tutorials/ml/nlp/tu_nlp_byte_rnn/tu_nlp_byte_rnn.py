#!/usr/bin/env python3
"""OA Tutorial — Byte RNN · all-position LM (autograd)

Mirrors the Rust tu_nlp_byte_rnn tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import oa
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical byte ByteEmbedding → Elman RNN → ByteHead language model composed from `oa.ml.nn` primitives."""

	def __init__(self, runtime):
		self.embedding = nn.ByteEmbedding(
			_nlp_common.MODEL_WIDTH, _nlp_common.RNG_SEED, runtime,
		)
		self.recurrent = nn.Rnn(
			_nlp_common.MODEL_WIDTH, _nlp_common.HIDDEN_WIDTH,
			seed=_nlp_common.RNG_SEED + 1, runtime=runtime,
		)
		self.head = nn.ByteHead(
			_nlp_common.HIDDEN_WIDTH, _nlp_common.RNG_SEED + 2, runtime,
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
			title="OA Tutorial — Byte RNN · all-position LM (autograd)",
			description="ByteEmbedding(32) → RNN(64) → ByteHead(256)",
			timer_name="byte_rnn_training_step",
			checkpoint_stem="oars_byte_rnn",
			learning_rate=0.01,
			expected_loss=0.186_080,
			loss_tolerance=0.001,
			expected_accuracy=0.921_875,
			expected_generation=b"to be that is the question whether tis nobler in the mind to suffer the slings and ar",
		),
		Model,
	)


if __name__ == "__main__":
	main()

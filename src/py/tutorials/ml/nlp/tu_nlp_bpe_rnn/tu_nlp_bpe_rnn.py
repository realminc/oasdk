#!/usr/bin/env python3
"""OA Tutorial — BPE RNN · all-position LM (autograd)

Mirrors the Rust tu_nlp_bpe_rnn tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical BPE Embedding → Elman RNN → Linear language model composed from `oa.ml.nn` primitives."""

	def __init__(self, runtime):
		self.embedding = nn.Embedding(
			_nlp_common.BPE_VOCAB_SIZE, _nlp_common.MODEL_WIDTH, _nlp_common.RNG_SEED, runtime,
		)
		self.recurrent = nn.Rnn(
			_nlp_common.MODEL_WIDTH, _nlp_common.HIDDEN_WIDTH,
			seed=_nlp_common.RNG_SEED + 1, runtime=runtime,
		)
		self.head = nn.Linear(
			_nlp_common.HIDDEN_WIDTH, _nlp_common.BPE_VOCAB_SIZE, _nlp_common.RNG_SEED + 2,
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
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE RNN · all-position LM (autograd)",
			description="BPE Embedding → RNN(32→64) → Linear(64→320)",
			timer_name="bpe_rnn_training_step",
			checkpoint_stem="oars_bpe_rnn",
			learning_rate=0.01,
			expected_loss=0.020_861,
			expected_accuracy=0.988_281,
			expected_generation=b"to be not to be that is the question whether tis nobler in the mind to suffer the sli",
		),
		Model,
	)


if __name__ == "__main__":
	main()

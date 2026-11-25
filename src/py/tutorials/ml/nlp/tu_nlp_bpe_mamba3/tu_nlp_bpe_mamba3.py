#!/usr/bin/env python3
"""OA Tutorial — BPE Mamba-3 · all-position LM (autograd)

Mirrors the Rust tu_nlp_bpe_mamba3 tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical BPE Embedding → Mamba-3 + residual → Linear language model composed from `oa.ml.nn` primitives."""

	def __init__(self, runtime):
		self.embedding = nn.Embedding(
			_nlp_common.BPE_VOCAB_SIZE, _nlp_common.MODEL_WIDTH, _nlp_common.RNG_SEED, runtime,
		)
		self.mamba = _nlp_common.canonical_mamba3(runtime, _nlp_common.RNG_SEED + 1)
		self.head = nn.Linear(
			_nlp_common.MODEL_WIDTH, _nlp_common.BPE_VOCAB_SIZE, _nlp_common.RNG_SEED + 2,
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
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE Mamba-3 · all-position LM (autograd)",
			description="BPE Embedding → Mamba-3(32,state=32,expand=2) + residual → Linear",
			timer_name="bpe_mamba3_training_step",
			checkpoint_stem="oars_bpe_mamba3",
			learning_rate=0.003,
			expected_loss=0.019_415,
			expected_accuracy=0.989_258,
			expected_generation=b"to beion wher tis nobler in the mind to suffer the slings and arrows of outrageous fo",
		),
		Model,
	)


if __name__ == "__main__":
	main()

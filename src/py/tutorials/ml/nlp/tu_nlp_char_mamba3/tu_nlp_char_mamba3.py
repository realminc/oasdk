#!/usr/bin/env python3
"""OA Tutorial — Char Mamba-3 · all-position LM (autograd)

Mirrors the Rust tu_nlp_char_mamba3 tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical character Embedding → Mamba-3 + residual → Linear language model composed from `oa.ml.nn` primitives."""

	def __init__(self, runtime):
		self.embedding = nn.Embedding(
			_nlp_common.CHAR_VOCAB_SIZE, _nlp_common.MODEL_WIDTH, _nlp_common.RNG_SEED, runtime,
		)
		self.mamba = _nlp_common.canonical_mamba3(runtime, _nlp_common.RNG_SEED + 1)
		self.head = nn.Linear(
			_nlp_common.MODEL_WIDTH, _nlp_common.CHAR_VOCAB_SIZE, _nlp_common.RNG_SEED + 2,
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
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_mamba3",
			cli_title="OA Tutorial — Char Mamba-3",
			header_line="║  OA Tutorial — Char Mamba-3 · all-position LM (autograd)         ║",
			model_line="Model: Embedding(32) → Mamba-3 SISO → residual → Linear(27)",
			sequence_unit="token",
			timer_name="char_mamba3_training_step",
			log_dir_leaf="log/nlp/char_mamba3",
			checkpoint_name="oars_char_mamba3.oam",
			default_lr=0.003,
			show_training_line=True,
			eval_loss_line=True,
			accuracy_note=True,
			loss_tolerance=0.001,
			accuracy_tolerance=0.000001,
			expected_loss=0.190_578,
			expected_accuracy=0.929_688,
			expected_generation="to be that is the question whether tis nobler in the mind to suffer the slings and ar",
		),
		Model,
	)


if __name__ == "__main__":
	main()

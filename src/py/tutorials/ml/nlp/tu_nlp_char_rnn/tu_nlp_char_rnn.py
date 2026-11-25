#!/usr/bin/env python3
"""OA Tutorial — Char RNN · all-position LM (autograd)

Mirrors the Rust tu_nlp_char_rnn tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical character Embedding → Elman RNN → Linear language model composed from `oa.ml.nn` primitives."""

	def __init__(self, runtime):
		self.embedding = nn.Embedding(
			_nlp_common.CHAR_VOCAB_SIZE, _nlp_common.MODEL_WIDTH, _nlp_common.RNG_SEED, runtime,
		)
		self.recurrent = nn.Rnn(
			_nlp_common.MODEL_WIDTH, _nlp_common.HIDDEN_WIDTH,
			seed=_nlp_common.RNG_SEED + 1, runtime=runtime,
		)
		self.head = nn.Linear(
			_nlp_common.HIDDEN_WIDTH, _nlp_common.CHAR_VOCAB_SIZE, _nlp_common.RNG_SEED + 2,
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
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_rnn",
			cli_title="OA Tutorial — Char RNN",
			header_line="║  OA Tutorial — Char RNN · all-position LM (autograd)             ║",
			model_line="Model: Embedding(27,32) → RNN(64) → Linear(64,27)",
			sequence_unit="token",
			timer_name="char_rnn_training_step",
			log_dir_leaf="log/nlp/char_rnn",
			checkpoint_name="oars_char_rnn.oam",
			default_lr=0.01,
			show_training_line=True,
			eval_loss_line=False,
			accuracy_note=True,
			final_loss_max=0.25,
			accuracy_min=0.9,
			expected_generation="to be that is the question whether tis nobler in the mind to suffer the slings and ar",
		),
		Model,
	)


if __name__ == "__main__":
	main()

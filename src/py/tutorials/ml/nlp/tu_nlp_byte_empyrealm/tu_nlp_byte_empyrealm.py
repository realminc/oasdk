#!/usr/bin/env python3
"""OA Tutorial — Empyrealm Core · autograd fidelity

Mirrors the Rust tu_nlp_byte_empyrealm tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import oa
import _nlp_common
import oa.ml.nn as nn


class Model:
	"""Canonical byte Empyrealm hybrid stack composed from `oa.ml.nn`."""

	def __init__(self, runtime):
		self.core = nn.Empyrealm(
			oa.ml.byte.VOCAB_SIZE,
			_nlp_common.canonical_empyrealm_config(),
			_nlp_common.RNG_SEED,
			runtime,
		)
		self.head = nn.Linear(
			_nlp_common.MODEL_WIDTH, oa.ml.byte.VOCAB_SIZE,
			_nlp_common.RNG_SEED + 2, runtime=runtime,
		)

	def forward(self, tokens):
		mixed = self.core.forward(tokens)
		return self.head.forward(
			mixed.reshape([tokens.num_elements, _nlp_common.MODEL_WIDTH])
		)

	def all_parameters(self):
		return (
			_nlp_common.parameters_of(self.core)
			+ _nlp_common.parameters_of(self.head)
		)



def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Empyrealm Core · autograd fidelity",
			description="Empyrealm(Byte Embedding + Mamba-3/attention hybrid stack) → Linear(32→256)",
			timer_name="byte_empyrealm_training_step",
			checkpoint_stem="oars_byte_empyrealm",
			learning_rate=0.003,
			expected_loss=0.179_619,
			loss_tolerance=0.001,
			expected_accuracy=0.929_688,
			expected_generation=b"to be that is the question whether tis nobler in the mind to suffer the slings and ar",
		),
		Model,
	)


if __name__ == "__main__":
	main()

#!/usr/bin/env python3
"""OA Tutorial — Byte Transformer · all-position LM (autograd)

Mirrors the Rust tu_nlp_byte_transformer tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import oa
import _nlp_common
import oa.ml.nn as nn


def construct(runtime):
	"""Construct the canonical one-block, one-head byte Transformer."""
	return nn.Transformer(
			oa.ml.byte.VOCAB_SIZE,
			_nlp_common.CONTEXT_LENGTH,
			_nlp_common.MODEL_WIDTH,
			_nlp_common.HIDDEN_WIDTH,
			1,
			1,
			epsilon=1e-5,
			seed=_nlp_common.RNG_SEED,
			runtime=runtime,
		)



def main() -> None:
	_nlp_common.run_byte(
		_nlp_common.Tutorial(
			title="OA Tutorial — Byte Transformer · all-position LM (autograd)",
			description="Byte + position Embedding → Transformer(32,64) → LN → Linear(32→256)",
			timer_name="byte_transformer_training_step",
			checkpoint_stem="oars_byte_transformer",
			learning_rate=0.01,
			expected_loss=0.190_565,
			loss_tolerance=0.001,
			expected_accuracy=0.928_711,
			expected_generation=b"to be that is the question whether tis nobler in the mind to suffer the slings and ar",
		),
		construct,
	)


if __name__ == "__main__":
	main()

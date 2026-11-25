#!/usr/bin/env python3
"""OA Tutorial — BPE Transformer · all-position LM (autograd)

Mirrors the Rust tu_nlp_bpe_transformer tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


def construct(runtime):
	"""Construct the canonical one-block, one-head BPE Transformer."""
	return nn.Transformer(
			_nlp_common.BPE_VOCAB_SIZE,
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
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE Transformer · all-position LM (autograd)",
			description="BPE + position Embedding → Transformer(32,64) → LN → Linear(32→320)",
			timer_name="bpe_transformer_training_step",
			checkpoint_stem="oars_bpe_transformer",
			learning_rate=0.01,
			expected_loss=0.020_028,
			expected_accuracy=0.989_258,
			expected_generation=b"to be end them to be or not to be that is the question whether tis nobler in the mind",
		),
		construct,
	)


if __name__ == "__main__":
	main()

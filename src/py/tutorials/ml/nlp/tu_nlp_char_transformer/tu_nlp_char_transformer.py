#!/usr/bin/env python3
"""OA Tutorial — Char Transformer · all-position LM (autograd)

Mirrors the Rust tu_nlp_char_transformer tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


def construct(runtime):
	"""Construct the canonical one-block, one-head character Transformer."""
	return nn.Transformer(
			_nlp_common.CHAR_VOCAB_SIZE,
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
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_transformer",
			cli_title="OA Tutorial — Char Transformer",
			header_line="║  OA Tutorial — Char Transformer · all-position LM (autograd)     ║",
			model_line="Model: token+position embedding → TransformerBlock → LayerNorm → Linear",
			sequence_unit="token",
			timer_name="char_transformer_training_step",
			log_dir_leaf="log/nlp/char_transformer",
			checkpoint_name="oars_char_transformer.oam",
			default_lr=0.01,
			show_training_line=True,
			eval_loss_line=True,
			accuracy_note=True,
			loss_tolerance=0.02,
			accuracy_tolerance=0.01,
			expected_loss=0.190_238,
			expected_accuracy=0.927,
			expected_generation="to be or not to be that is the question whether tis nobler in the mind to suffer the ",
		),
		construct,
	)


if __name__ == "__main__":
	main()

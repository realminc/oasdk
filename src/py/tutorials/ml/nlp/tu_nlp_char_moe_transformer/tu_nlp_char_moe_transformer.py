#!/usr/bin/env python3
"""OA Tutorial — Char MoE Transformer · all-position LM

Mirrors the Rust tu_nlp_char_moe_transformer tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


def construct(runtime):
	"""Construct the canonical sparse-MoE character Transformer."""
	return nn.Transformer.new_with_moe(
			_nlp_common.CHAR_VOCAB_SIZE,
			_nlp_common.CONTEXT_LENGTH,
			_nlp_common.MODEL_WIDTH,
			_nlp_common.MOE_EXPERT_HIDDEN_WIDTH,
			1,
			1,
			_nlp_common.MOE_NUM_EXPERTS,
			_nlp_common.MOE_EXPERTS_PER_TOKEN,
			epsilon=1e-5,
			seed=_nlp_common.RNG_SEED,
			runtime=runtime,
		)



def main() -> None:
	_nlp_common.run_char(
		_nlp_common.CharTutorial(
			prog="tu_nlp_char_moe_transformer",
			cli_title="OA Tutorial — Char MoE Transformer",
			header_line="║  OA Tutorial — Char MoE Transformer · all-position LM           ║",
			model_line=(
				"Model: token+position embedding → attention → "
				"sparse MoE(E=4, K=2, DFF=16) → LayerNorm → Linear"
			),
			sequence_unit="token",
			timer_name="char_moe_transformer_training_step",
			log_dir_leaf="log/nlp/char_moe_transformer",
			checkpoint_name="oars_char_moe_transformer.oam",
			default_lr=0.01,
			show_training_line=True,
			eval_loss_line=True,
			accuracy_note=False,
			loss_tolerance=0.02,
			accuracy_tolerance=0.01,
			expected_loss=0.190_7,
			expected_accuracy=0.932,
			expected_generation="to be that is the question whether tis nobler in the mind to suffer the slings and ar",
		),
		construct,
	)


if __name__ == "__main__":
	main()

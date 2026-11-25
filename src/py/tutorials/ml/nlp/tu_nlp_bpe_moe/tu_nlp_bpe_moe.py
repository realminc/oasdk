#!/usr/bin/env python3
"""OA Tutorial — BPE MoE Transformer · all-position LM (autograd)

Mirrors the Rust tu_nlp_bpe_moe tutorial and its canonical `oasdk::ml::nlp` recipe;
the model is composed here from the shipped `oa.ml.nn` primitive layers.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import _nlp_common
import oa.ml.nn as nn


def construct(runtime):
	"""Construct the canonical sparse-MoE BPE Transformer."""
	return nn.Transformer.new_with_moe(
			_nlp_common.BPE_VOCAB_SIZE,
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
	_nlp_common.run_bpe(
		_nlp_common.Tutorial(
			title="OA Tutorial — BPE MoE Transformer · all-position LM (autograd)",
			description="BPE + position Embedding → Attention + MoE(E=4,K=2,DFF=16) → LN → Linear",
			timer_name="bpe_moe_training_step",
			checkpoint_stem="oars_bpe_moe",
			learning_rate=0.01,
			expected_loss=0.019_962,
			expected_accuracy=0.988_281,
			expected_generation=b"to bee or not to be that is the question whether tis nobler in the mind to suffer the",
		),
		construct,
	)


if __name__ == "__main__":
	main()

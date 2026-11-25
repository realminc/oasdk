"""Shared training loops for the OA Rust-parity NLP suite.

Mirrors src/rs/lib/ml/rs and the Rust tutorial loops: every loop is driven
by `oa.ml.ItTraining` so the same lifecycle, progress bar, training summary,
metrics log, whole-program capture, and program diagnostics apply in both
languages. The canonical workload constants, corpus samplers, and greedy
generation helpers live here as plain host-side Python over the shipped
`oa.ml`/`oa.Matrix` surface.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import oa

# Match the native tutorials' unbuffered interleaving with ProgressBar output.
sys.stdout.reconfigure(line_buffering=True)

# ── Canonical workload (mirrors src/rs/lib/ml/rs) ─────────────────────────

CONTEXT_LENGTH = 16
MODEL_WIDTH = 32
HIDDEN_WIDTH = 64
MOE_EXPERT_HIDDEN_WIDTH = 16
MOE_NUM_EXPERTS = 4
MOE_EXPERTS_PER_TOKEN = 2
TRAINING_STEPS = 300
BATCH_SIZE = 64
RNG_SEED = 20_260_714
GENERATION_PROMPT = "to be"
GENERATION_LENGTH = 80
CHAR_VOCAB_SIZE = 27
BPE_VOCAB_SIZE = 320
BPE_MERGES = BPE_VOCAB_SIZE - oa.ml.byte.VOCAB_SIZE

CORPUS = (
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them "
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them "
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them "
)


def encode_char(character: str) -> int:
	"""Encode one character using the canonical `a-z` plus space vocabulary."""
	if "a" <= character <= "z":
		return ord(character) - ord("a")
	if "A" <= character <= "Z":
		return ord(character) - ord("A")
	return 26


def decode_char(token: int) -> str:
	"""Decode one canonical character token. Invalid values decode to space."""
	return chr(ord("a") + token) if token < 26 else " "


def encode(text: str) -> list:
	"""Encode text into canonical character tokens."""
	return [encode_char(character) for character in text]


def decode(tokens) -> str:
	"""Decode canonical character tokens."""
	return "".join(decode_char(token) for token in tokens)


def canonical_mamba3(runtime: oa.Runtime, seed: int) -> oa.ml.nn.Mamba3:
	"""Construct the canonical `32/state=32/expand=2/head=16` Mamba-3 layer."""
	config = oa.ml.nn.Mamba3Config(MODEL_WIDTH)
	config.state_size = 32
	config.expand = 2
	config.head_dim = 16
	config.num_groups = 1
	config.mimo_rank = 1
	config.rope_fraction = 0.5
	config.dt_min = 0.001
	config.dt_max = 0.1
	config.dt_init_floor = 0.0001
	config.a_floor = 0.0001
	config.output_norm = True
	return oa.ml.nn.Mamba3(config, seed, runtime)


def canonical_empyrealm_config() -> oa.ml.nn.EmpyrealmConfig:
	"""Construct the canonical hybrid fidelity Empyrealm configuration."""
	config = oa.ml.nn.EmpyrealmConfig(
		model_width=MODEL_WIDTH,
		sequence_length=CONTEXT_LENGTH,
		layers=4,
		attention_every=4,
		attention_heads=4,
		ffn_hidden=MODEL_WIDTH * 2,
		moe_experts=0,
		moe_experts_per_token=0,
		epsilon=1e-5,
	)
	mamba = oa.ml.nn.Mamba3Config(MODEL_WIDTH)
	mamba.state_size = 32
	mamba.expand = 2
	mamba.head_dim = 16
	mamba.num_groups = 1
	mamba.mimo_rank = 1
	mamba.rope_fraction = 0.5
	mamba.dt_min = 0.001
	mamba.dt_max = 0.1
	mamba.dt_init_floor = 0.0001
	mamba.a_floor = 0.0001
	mamba.output_norm = True
	config.mamba = mamba
	return config


# ── Corpus samplers ───────────────────────────────────────────────────────────


class CharSampler:
	"""Deterministic dense all-position next-character sampler."""

	def __init__(self, batch_size: int):
		if batch_size == 0:
			raise ValueError("NLP sampler batch size must be nonzero")
		self.tokens = encode(CORPUS)
		self.batch_size = batch_size
		self.cursor = 0
		self._cache = {}

	def next_values(self):
		"""Produce the next host-side input and shifted-target token arrays."""
		limit = len(self.tokens) - CONTEXT_LENGTH - 1
		cached = self._cache.get(self.cursor)
		if cached is None:
			input = [0] * (self.batch_size * CONTEXT_LENGTH)
			target = [0] * (self.batch_size * CONTEXT_LENGTH)
			for batch in range(self.batch_size):
				start = (self.cursor + batch * 7) % limit
				for position in range(CONTEXT_LENGTH):
					output = batch * CONTEXT_LENGTH + position
					input[output] = self.tokens[start + position]
					target[output] = self.tokens[start + position + 1]
			cached = (input, target)
			self._cache[self.cursor] = cached
		self.cursor = (self.cursor + self.batch_size) % limit
		return cached


class ByteSampler:
	"""Deterministic dense all-position next-byte sampler over the corpus."""

	def __init__(self, batch_size: int):
		if batch_size == 0:
			raise ValueError("NLP byte sampler batch size must be nonzero")
		self.tokens = CORPUS.encode("utf-8")
		self.batch_size = batch_size
		self.cursor = 0
		self._cache = {}

	def next_values(self):
		"""Produce the next packed-U8 input and U32 shifted-target arrays."""
		limit = len(self.tokens) - CONTEXT_LENGTH - 1
		cached = self._cache.get(self.cursor)
		if cached is None:
			input = [0] * (self.batch_size * CONTEXT_LENGTH)
			target = [0] * (self.batch_size * CONTEXT_LENGTH)
			for batch in range(self.batch_size):
				start = (self.cursor + batch * 7) % limit
				for position in range(CONTEXT_LENGTH):
					output = batch * CONTEXT_LENGTH + position
					input[output] = self.tokens[start + position]
					target[output] = self.tokens[start + position + 1]
			cached = (input, target)
			self._cache[self.cursor] = cached
		self.cursor = (self.cursor + self.batch_size) % limit
		return cached


class BpeSampler:
	"""Deterministic dense all-position sampler over canonical BPE tokens."""

	def __init__(self, batch_size: int, tokenizer: oa.ml.BpeTokenizer):
		if batch_size == 0:
			raise ValueError("NLP BPE sampler batch size must be nonzero")
		self.tokenizer = tokenizer
		self.tokens = tokenizer.encode_text(CORPUS)
		if len(self.tokens) <= CONTEXT_LENGTH + 1:
			raise ValueError("NLP BPE corpus must contain more than one shifted context")
		self.batch_size = batch_size
		self.cursor = 0
		self._last_batch_bytes = 0
		self._cache = {}

	def next_values(self):
		"""Produce the next U32 BPE input and shifted-target arrays."""
		limit = len(self.tokens) - CONTEXT_LENGTH - 1
		cached = self._cache.get(self.cursor)
		if cached is None:
			input = [0] * (self.batch_size * CONTEXT_LENGTH)
			target = [0] * (self.batch_size * CONTEXT_LENGTH)
			batch_bytes = 0
			for batch in range(self.batch_size):
				start = (self.cursor + batch * 7) % limit
				for position in range(CONTEXT_LENGTH):
					output = batch * CONTEXT_LENGTH + position
					input[output] = self.tokens[start + position]
					target[output] = self.tokens[start + position + 1]
					batch_bytes += len(
						self.tokenizer.decode([target[output]])
					)
			cached = (input, target, batch_bytes)
			self._cache[self.cursor] = cached
		input, target, self._last_batch_bytes = cached
		self.cursor = (self.cursor + self.batch_size) % limit
		return input, target

	def last_batch_bytes(self) -> int:
		"""Exact source bytes represented by the most recent shifted targets."""
		return self._last_batch_bytes

	def last_batch_bytes_per_token(self) -> float:
		"""Mean source bytes per target token in the last batch."""
		return self._last_batch_bytes / (self.batch_size * CONTEXT_LENGTH)

	def next_batch_bytes(self) -> int:
		"""Exact source bytes of the next target batch, without advancing."""
		limit = len(self.tokens) - CONTEXT_LENGTH - 1
		count = 0
		for batch in range(self.batch_size):
			start = (self.cursor + batch * 7) % limit
			for position in range(CONTEXT_LENGTH):
				count += len(self.tokenizer.decode([self.tokens[start + position + 1]]))
		return count


# ── Evaluation and greedy generation ─────────────────────────────────────────


def accuracy(logits: oa.Matrix, targets: oa.Matrix) -> float:
	"""Compute all-position argmax accuracy as a fraction in `[0, 1]`."""
	return oa.ml.metric.accuracy(logits, targets)


def parameters_of(module) -> list:
	"""Collect a layer's parameter list across `parameters`/`all_parameters`."""
	getter = getattr(module, "all_parameters", None) or module.parameters
	return list(getter())


def _argmax(values) -> int:
	best_index = 0
	for index, value in enumerate(values):
		if value > values[best_index]:
			best_index = index
	return best_index


def generate_greedy(model, prompt: str, count: int) -> str:
	"""Greedily generate characters with the canonical left-filled window."""
	prompt_tokens = encode(prompt)
	if not prompt_tokens:
		raise ValueError("NLP generation prompt must not be empty")
	context = [26] * CONTEXT_LENGTH
	copied = min(len(prompt_tokens), CONTEXT_LENGTH)
	context[:copied] = prompt_tokens[:copied]
	filled = copied
	logit_row = filled - 1
	output = prompt
	for _ in range(count):
		input_matrix = oa.Matrix.new_from_u32([1, CONTEXT_LENGTH], context, None)
		logits = model.forward(input_matrix)
		if logits.shape != [CONTEXT_LENGTH, CHAR_VOCAB_SIZE]:
			raise ValueError("character model generation produced an invalid logit shape")
		values = logits.read_f32()
		begin = logit_row * CHAR_VOCAB_SIZE
		next_token = _argmax(values[begin : begin + CHAR_VOCAB_SIZE])
		output += decode_char(next_token)
		if filled < CONTEXT_LENGTH:
			context[filled] = next_token
			filled += 1
			logit_row = filled - 1
		else:
			context = context[1:] + [next_token]
			logit_row = CONTEXT_LENGTH - 1
	return output


def generate_bytes_greedy(model, prompt: bytes, count: int) -> bytes:
	"""Greedily generate exact bytes with the canonical sliding window."""
	if not prompt:
		raise ValueError("NLP byte generation prompt must not be empty")
	context = [0] * CONTEXT_LENGTH
	copied = min(len(prompt), CONTEXT_LENGTH)
	context[:copied] = prompt[:copied]
	filled = copied
	logit_row = filled - 1
	output = bytearray(prompt)
	for _ in range(count):
		input_matrix = oa.Matrix.new_from_u8([1, CONTEXT_LENGTH], context, None)
		logits = model.forward(input_matrix)
		if logits.shape != [CONTEXT_LENGTH, oa.ml.byte.VOCAB_SIZE]:
			raise ValueError("byte model generation produced an invalid logit shape")
		row = logits.slice(0, logit_row, logit_row + 1).reshape(
			[oa.ml.byte.VOCAB_SIZE]
		)
		next_byte = oa.ml.byte.decode(row)[0]
		output.append(next_byte)
		if filled < CONTEXT_LENGTH:
			context[filled] = next_byte
			filled += 1
			logit_row = filled - 1
		else:
			context = context[1:] + [next_byte]
			logit_row = CONTEXT_LENGTH - 1
	return bytes(output)


def generate_bpe_greedy(
	model, tokenizer: oa.ml.BpeTokenizer, prompt: bytes, byte_count: int
) -> bytes:
	"""Greedily generate BPE tokens until `byte_count` decoded source bytes.

	The decoded result is truncated to ``len(prompt) + byte_count``, matching
	the donor when the last learned token crosses the byte boundary.
	"""
	if not prompt:
		raise ValueError("NLP BPE generation prompt must not be empty")
	prompt_tokens = tokenizer.encode(prompt)
	copied = min(len(prompt_tokens), CONTEXT_LENGTH)
	context = [0] * CONTEXT_LENGTH
	context[:copied] = prompt_tokens[:copied]
	filled = max(copied, 1)
	logit_row = filled - 1
	generated = list(prompt_tokens)
	generated_bytes = 0
	for _ in range(byte_count):
		if generated_bytes >= byte_count:
			break
		input_matrix = oa.Matrix.new_from_u32([1, CONTEXT_LENGTH], context, None)
		logits = model.forward(input_matrix)
		if logits.shape != [CONTEXT_LENGTH, tokenizer.vocab_size()]:
			raise ValueError("BPE model generation produced an invalid logit shape")
		row = logits.slice(0, logit_row, logit_row + 1).reshape(
			[tokenizer.vocab_size()]
		)
		next_token = row.sample_logits(0.0, 0, 1.0, 0).read_i32()[0]
		if next_token < 0 or next_token >= tokenizer.vocab_size():
			raise ValueError("BPE model generated an invalid token ID")
		generated.append(next_token)
		generated_bytes += len(tokenizer.decode([next_token]))
		if filled < CONTEXT_LENGTH:
			context[filled] = next_token
			filled += 1
			logit_row = filled - 1
		else:
			context = context[1:] + [next_token]
			logit_row = CONTEXT_LENGTH - 1
	output = tokenizer.decode(generated)
	return output[: len(prompt) + byte_count]


# ── Workload overrides (OA_TUTORIAL_* env + --steps/--batch/--lr) ─────────────


def _configured_positive(name: str, default: int) -> int:
	"""Read a positive workload override; reject invalid recipes before training."""
	value = int(os.environ.get(name, default))
	if value <= 0:
		raise ValueError(f"{name} must be a positive integer")
	return value


def _steps() -> int:
	return _configured_positive("OA_TUTORIAL_STEPS", TRAINING_STEPS)


def _batch() -> int:
	return _configured_positive("OA_TUTORIAL_BATCH", BATCH_SIZE)


def _generation_units() -> int:
	return _configured_positive("OA_TUTORIAL_GENERATION_UNITS", GENERATION_LENGTH)


def _default_recipe(steps: int, batch: int, lr: float, default_lr: float) -> bool:
	return (
		steps == TRAINING_STEPS
		and batch == BATCH_SIZE
		and lr == default_lr
		and _generation_units() == GENERATION_LENGTH
	)


def _num_parameters(model) -> int:
	return sum(p.data().num_elements for p in model.all_parameters())


def _print_header(title: str) -> None:
	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print(f"║  {title:<62}║")
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()


def _train(
	training: oa.ml.ItTraining,
	model,
	sampler,
	runtime: oa.Runtime,
	batch: int,
	steps: int,
	input_dtype: str,
	metrics: oa.LogMetrics | None = None,
	source_units=None,
	losses_out: list | None = None,
) -> tuple[float, list, list]:
	"""Drive the ItTraining step loop; return (initial_loss, last_input, last_target)."""
	new_input = oa.Matrix.new_from_u8 if input_dtype == "u8" else oa.Matrix.new_from_u32
	initial_loss = 0.0
	last_input: list = []
	last_target: list = []
	for _ in range(steps):
		if source_units is not None:
			training.record_source_units(source_units())

		def prepare():
			nonlocal last_input, last_target
			last_input, last_target = sampler.next_values()
			return (
				new_input([batch, CONTEXT_LENGTH], last_input, runtime),
				oa.Matrix.new_from_u32([batch * CONTEXT_LENGTH], last_target, runtime),
			)

		def record(prepared):
			input_matrix, target = prepared
			with oa.ml.GradientTape() as tape:
				logits = model.forward(input_matrix)
				loss = oa.ml.loss.cross_entropy(logits, target)
				tape.backward(loss)
			return loss

		assert training.step(prepare, record)
		snap = training.snapshot()
		if snap.step_count == 1:
			initial_loss = snap.last_loss or 0.0
		if snap.last_loss is not None:
			if losses_out is not None:
				losses_out.append(snap.last_loss)
			if metrics is not None:
				metrics.log_scalar("loss", snap.step_count, snap.last_loss)
	return initial_loss, last_input, last_target


def _open_metrics(log_dir_leaf: str) -> oa.LogMetrics:
	metrics = oa.LogMetrics()
	try:
		metrics.open(oa.Path.var_rel(log_dir_leaf))
	except Exception as e:
		print(f"[OA] metrics unavailable: {e}", file=sys.stderr)
	return metrics


def _eval_input(runtime: oa.Runtime, dtype: str, batch: int, values: list) -> oa.Matrix:
	new_input = oa.Matrix.new_from_u8 if dtype == "u8" else oa.Matrix.new_from_u32
	return new_input([batch, CONTEXT_LENGTH], values, runtime)


# ── Byte workload (mirrors _byte_common.rs) ───────────────────────────────────


@dataclass
class Tutorial:
	"""Mirrors the shared `Tutorial` struct of _byte_common.rs/_bpe_common.rs."""

	title: str
	description: str
	timer_name: str
	checkpoint_stem: str
	learning_rate: float
	expected_loss: float
	expected_accuracy: float
	expected_generation: bytes | str
	loss_tolerance: float = 0.001


def run_byte(tutorial: Tutorial, construct) -> None:
	parser = argparse.ArgumentParser(prog=None, description=tutorial.title)
	parser.add_argument("--steps", type=int, default=0, help="training steps")
	parser.add_argument(
		"--plot",
		type=str,
		default="",
		help="render the loss curve to an image file",
	)
	args = parser.parse_args()
	steps = args.steps if args.steps > 0 else _steps()
	batch = _batch()
	runtime = oa.Runtime()
	oa.Runtime.set_default(runtime=runtime)
	model = construct(runtime)
	# Complete asynchronous parameter initialization before isolated capture.
	runtime.checkpoint().wait()
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=tutorial.learning_rate)
	sampler = ByteSampler(batch)

	_print_header(tutorial.title)
	print(f"tokenizer: raw bytes · vocab={oa.ml.byte.VOCAB_SIZE}")
	print("Task: dense next-byte prediction at every position\n")
	print(f"Model: {tutorial.description}")
	print(
		f"params: {_num_parameters(model)}    "
		f"Optimizer: AdamW(lr={tutorial.learning_rate})\n"
	)

	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			batch_size=batch,
			sequence_length=CONTEXT_LENGTH,
			sequence_unit="byte token",
			source_unit="byte",
			source_units_per_sample=CONTEXT_LENGTH,
			timer_name=tutorial.timer_name,
			enable_gpu_timing=True,
		), runtime=runtime)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	metrics = _open_metrics(f"log/nlp/{tutorial.timer_name}")
	loss_curve: list = []
	initial_loss, last_input, last_target = _train(
		training, model, sampler, runtime, batch, steps, "u8", metrics,
		losses_out=loss_curve,
	)
	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_training_loss = result.last_loss or 0.0

	input_matrix = _eval_input(runtime, "u8", batch, last_input)
	target = oa.Matrix.new_from_u32([batch * CONTEXT_LENGTH], last_target, runtime)
	logits = model.forward(input_matrix)
	eval_accuracy = accuracy(logits, target)
	evaluation_loss = oa.ml.loss.cross_entropy(logits, target).read_f32()[0]
	generated = generate_bytes_greedy(
		model, GENERATION_PROMPT.encode(), _generation_units()
	).decode("utf-8", errors="replace")

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({oa.ml.byte.VOCAB_SIZE}) = "
		f"{math.log(oa.ml.byte.VOCAB_SIZE):.4f}"
	)
	print(f"  byte-token accuracy: {eval_accuracy * 100.0:.1f}%")
	print(f"  post-update evaluation loss: {evaluation_loss:.6f}")
	print(
		f'\nGeneration:\n  prompt: "{GENERATION_PROMPT}"\n'
		f'  generated: "{generated}"'
	)
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	checkpoint = oa.Path.temp().join(f"{tutorial.checkpoint_stem}-{os.getpid()}.oam")
	params = model.all_parameters()
	oa.ml.save_checkpoint(str(checkpoint), params, optimizer)
	reloaded = construct(runtime)
	reloaded_optimizer = oa.ml.AdamW(
		reloaded.all_parameters(), learning_rate=tutorial.learning_rate
	)
	oa.ml.load_checkpoint( str(checkpoint), reloaded.all_parameters(), reloaded_optimizer, runtime=runtime)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert accuracy(reloaded.forward(input_matrix), target) == eval_accuracy
	assert (
		generate_bytes_greedy(
			reloaded, GENERATION_PROMPT.encode(), _generation_units()
		).decode("utf-8", errors="replace")
		== generated
	)
	oa.Filesystem.remove_file(checkpoint)
	print("checkpoint: optimizer step, accuracy, and exact generation restored")

	if args.plot:
		_write_loss_curve(
			runtime,
			loss_curve,
			args.plot,
			tutorial.title.removeprefix("OA Tutorial — "),
			oa.ml.byte.VOCAB_SIZE,
		)
		print(f"plot: {args.plot}")

	if _default_recipe(steps, batch, tutorial.learning_rate, tutorial.learning_rate):
		assert initial_loss > 5.0
		assert final_training_loss < initial_loss
		assert abs(evaluation_loss - tutorial.expected_loss) < tutorial.loss_tolerance
		assert abs(eval_accuracy - tutorial.expected_accuracy) < 0.001
		expected = tutorial.expected_generation
		assert generated == (
			expected.decode("utf-8") if isinstance(expected, bytes) else expected
		)


# ── BPE workload (mirrors _bpe_common.rs) ─────────────────────────────────────


def run_bpe(tutorial: Tutorial, construct) -> None:
	parser = argparse.ArgumentParser(prog=None, description=tutorial.title)
	parser.add_argument("--steps", type=int, default=0, help="training steps")
	parser.add_argument(
		"--plot",
		type=str,
		default="",
		help="render the loss curve to an image file",
	)
	args = parser.parse_args()
	steps = args.steps if args.steps > 0 else _steps()
	batch = _batch()
	runtime = oa.Runtime()
	oa.Runtime.set_default(runtime=runtime)
	tokenizer = oa.ml.BpeTokenizer(BPE_VOCAB_SIZE)
	tokenizer.train_text(CORPUS, BPE_MERGES)
	corpus_tokens = tokenizer.encode_text(CORPUS)
	assert tokenizer.decode_text(corpus_tokens) == CORPUS
	bytes_per_token = len(CORPUS) / len(corpus_tokens)
	model = construct(runtime)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=tutorial.learning_rate)
	sampler = BpeSampler(batch, tokenizer)

	_print_header(tutorial.title)
	print(
		f"tokenizer: byte BPE · vocab={tokenizer.vocab_size()} "
		f"(256 bytes + {tokenizer.num_merges()} merges)"
	)
	print(
		f"compression: {len(CORPUS)} bytes → {len(corpus_tokens)} tokens "
		f"· {bytes_per_token:.3f} byte/token "
		f"· {100.0 * (1.0 - len(corpus_tokens) / len(CORPUS)):.1f}% fewer positions"
	)
	print(
		f"context coverage: {CONTEXT_LENGTH} BPE tokens ≈ "
		f"{CONTEXT_LENGTH * bytes_per_token:.1f} source bytes at corpus average"
	)
	print("Task: dense next-token prediction at every position\n")
	print(f"Model: {tutorial.description}")
	print(
		f"params: {_num_parameters(model)}    "
		f"Optimizer: AdamW(lr={tutorial.learning_rate})\n"
	)

	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			batch_size=batch,
			sequence_length=CONTEXT_LENGTH,
			sequence_unit="BPE token",
			source_unit="byte",
			timer_name=tutorial.timer_name,
			enable_gpu_timing=True,
		), runtime=runtime)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	loss_curve: list = []
	initial_loss, last_input, last_target = _train(
		training,
		model,
		sampler,
		runtime,
		batch,
		steps,
		"u32",
		source_units=lambda: sampler.next_batch_bytes(),
		losses_out=loss_curve,
	)
	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_training_loss = result.last_loss or 0.0

	input_matrix = _eval_input(runtime, "u32", batch, last_input)
	target = oa.Matrix.new_from_u32([batch * CONTEXT_LENGTH], last_target, runtime)
	logits = model.forward(input_matrix)
	eval_accuracy = accuracy(logits, target)
	evaluation_loss = oa.ml.loss.cross_entropy(logits, target).read_f32()[0]
	bytes_per_token_last = sampler.last_batch_bytes_per_token()
	generated = generate_bpe_greedy(
		model, tokenizer, GENERATION_PROMPT.encode(), _generation_units()
	).decode("utf-8", errors="replace")

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({tokenizer.vocab_size()}) = "
		f"{math.log(tokenizer.vocab_size()):.4f}"
	)
	print(
		f"  final batch: {bytes_per_token_last:.3f} byte/token · "
		f"{final_training_loss / math.log(2) / bytes_per_token_last:.4f} bits/byte"
	)
	print(f"  BPE-token accuracy: {eval_accuracy * 100.0:.1f}%")
	print(f"  post-update evaluation loss: {evaluation_loss:.6f}")
	print(
		f'\nGeneration:\n  prompt: "{GENERATION_PROMPT}"\n'
		f'  generated: "{generated}"'
	)
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	if _default_recipe(steps, batch, tutorial.learning_rate, tutorial.learning_rate):
		assert initial_loss > 5.0
		assert final_training_loss < initial_loss
		assert abs(evaluation_loss - tutorial.expected_loss) < 0.001
		assert abs(eval_accuracy - tutorial.expected_accuracy) < 0.001
		expected = tutorial.expected_generation
		assert generated == (
			expected.decode("utf-8") if isinstance(expected, bytes) else expected
		)

	base = oa.Path.temp().join(f"{tutorial.checkpoint_stem}-{os.getpid()}")
	checkpoint = base.with_extension("oam")
	vocabulary = base.with_extension("bpe")
	params = model.all_parameters()
	oa.ml.save_checkpoint(str(checkpoint), params, optimizer)
	tokenizer.save(str(vocabulary))
	reloaded_tokenizer = oa.ml.BpeTokenizer(256)
	reloaded_tokenizer.load(str(vocabulary))
	reloaded = construct(runtime)
	reloaded_optimizer = oa.ml.AdamW(
		reloaded.all_parameters(), learning_rate=tutorial.learning_rate
	)
	oa.ml.load_checkpoint( str(checkpoint), reloaded.all_parameters(), reloaded_optimizer, runtime=runtime)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert accuracy(reloaded.forward(input_matrix), target) == eval_accuracy
	assert (
		generate_bpe_greedy(
						reloaded,
			reloaded_tokenizer,
			GENERATION_PROMPT.encode(),
			_generation_units(),
		).decode("utf-8", errors="replace")
		== generated
	)
	oa.Filesystem.remove_file(checkpoint)
	oa.Filesystem.remove_file(vocabulary)

	if args.plot:
		_write_loss_curve(
			runtime,
			loss_curve,
			args.plot,
			tutorial.title.removeprefix("OA Tutorial — "),
			tokenizer.vocab_size()
		)
		print(f"plot: {args.plot}")


# ── Char workload (mirrors the standalone tu_nlp_char_* tutorials) ────────────


@dataclass
class CharTutorial:
	"""Per-file literals of the standalone Rust Char tutorials."""

	prog: str
	cli_title: str
	header_line: str
	model_line: str
	sequence_unit: str
	timer_name: str
	log_dir_leaf: str
	checkpoint_name: str
	default_lr: float
	show_training_line: bool
	eval_loss_line: bool
	accuracy_note: bool
	expected_generation: str
	loss_tolerance: float | None = None
	accuracy_tolerance: float | None = None
	expected_loss: float = 0.0
	expected_accuracy: float = 0.0
	final_loss_max: float | None = None
	accuracy_min: float | None = None


def run_char(tutorial: CharTutorial, construct) -> None:
	parser = argparse.ArgumentParser(prog=tutorial.prog, description=tutorial.cli_title)
	parser.add_argument("--steps", type=int, default=0, help="training steps")
	parser.add_argument("--batch", type=int, default=0, help="batch size")
	parser.add_argument("--lr", type=float, default=0.0, help="learning rate")
	parser.add_argument(
		"--plot",
		type=str,
		default="",
		help="render the loss curve to an image file",
	)
	args = parser.parse_args()
	steps = args.steps if args.steps > 0 else _steps()
	batch = args.batch if args.batch > 0 else _batch()
	lr = args.lr if args.lr > 0.0 else tutorial.default_lr

	runtime = oa.Runtime()

	oa.Runtime.set_default(runtime=runtime)
	model = construct(runtime)
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=lr)
	sampler = CharSampler(batch)

	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print(tutorial.header_line)
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()
	print(f"tokenizer: character · vocab={CHAR_VOCAB_SIZE} (a-z + space)")
	print("Task: dense next-character prediction at every position\n")
	print(tutorial.model_line)
	print(f"params: {_num_parameters(model)}    Optimizer: AdamW(lr={lr})\n")
	if tutorial.show_training_line:
		print(
			f"training: {steps} steps · batch={batch} · "
			f"sequence={CONTEXT_LENGTH} character tokens"
		)

	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			batch_size=batch,
			sequence_length=CONTEXT_LENGTH,
			sequence_unit=tutorial.sequence_unit,
			timer_name=tutorial.timer_name,
			enable_gpu_timing=True,
		), runtime=runtime)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	metrics = _open_metrics(tutorial.log_dir_leaf)
	loss_curve: list = []
	initial_loss, last_input, last_target = _train(
		training,
		model,
		sampler,
		runtime,
		batch,
		steps,
		"u32",
		metrics,
		losses_out=loss_curve,
	)
	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_loss = result.last_loss or 0.0

	input_matrix = _eval_input(runtime, "u32", batch, last_input)
	target = oa.Matrix.new_from_u32([batch * CONTEXT_LENGTH], last_target, runtime)
	logits = model.forward(input_matrix)
	eval_accuracy = accuracy(logits, target)
	evaluation_loss = (
		oa.ml.loss.cross_entropy(logits, target).read_f32()[0]
		if tutorial.eval_loss_line
		else 0.0
	)
	generated = generate_greedy(
		model, GENERATION_PROMPT, _generation_units()
	)

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({CHAR_VOCAB_SIZE}) = "
		f"{math.log(CHAR_VOCAB_SIZE):.4f}"
	)
	if tutorial.eval_loss_line:
		print(
			f"  loss: evaluation {evaluation_loss:.6f} · "
			f"last training {final_loss:.6f}"
		)
	note = " (compare within Char only)" if tutorial.accuracy_note else ""
	print(f"  character-token accuracy: {eval_accuracy * 100.0:.1f}%{note}")
	print(
		f'\nGeneration:\n  prompt: "{GENERATION_PROMPT}"\n'
		f'  generated: "{generated}"'
	)
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	if args.plot:
		_write_loss_curve(
			runtime,
			loss_curve,
			args.plot,
			tutorial.cli_title.removeprefix("OA Tutorial — "),
			CHAR_VOCAB_SIZE,
		)
		print(f"plot: {args.plot}")

	if _default_recipe(steps, batch, lr, tutorial.default_lr):
		assert initial_loss > 3.0
		if tutorial.loss_tolerance is not None:
			assert abs(evaluation_loss - tutorial.expected_loss) < tutorial.loss_tolerance
		if tutorial.accuracy_tolerance is not None:
			assert abs(eval_accuracy - tutorial.expected_accuracy) < tutorial.accuracy_tolerance
		if tutorial.final_loss_max is not None:
			assert final_loss < tutorial.final_loss_max
		if tutorial.accuracy_min is not None:
			assert eval_accuracy > tutorial.accuracy_min
		assert generated == tutorial.expected_generation

	checkpoint = oa.Path.temp().join(tutorial.checkpoint_name)
	params = model.all_parameters()
	oa.ml.save_checkpoint(str(checkpoint), params, optimizer)
	reloaded = construct(runtime)
	reloaded_optimizer = oa.ml.AdamW(reloaded.all_parameters(), learning_rate=lr)
	oa.ml.load_checkpoint( str(checkpoint), reloaded.all_parameters(), reloaded_optimizer, runtime=runtime)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert accuracy(reloaded.forward(input_matrix), target) == eval_accuracy
	assert (
		generate_greedy(reloaded, GENERATION_PROMPT, _generation_units())
		== generated
	)
	oa.Filesystem.remove_file(checkpoint)


def _write_loss_curve(
	runtime: oa.Runtime, loss_curve: list, path: str, title: str, vocab_size: int
) -> None:
	"""Render the recorded loss curve through oa.plot headless GPU composition."""
	fig = oa.plot.Figure(
		oa.plot.FigureConfig(
			title=f"{title} — dense next-token LM",
			rows=1,
			cols=1,
			width=960,
			height=540,
			theme=oa.plot.PlotTheme.Dark,
		)
	)
	ax = fig.ax(0, 0)
	ax.title("training loss")
	ax.x_label("optimizer step")
	ax.y_label("cross entropy")
	ax.grid(True)
	ax.legend(True)
	steps_x = [float(i) for i in range(1, len(loss_curve) + 1)]
	baseline = math.log(vocab_size)
	ax.limits(0.0, len(loss_curve) + 1.0, 0.0, max(baseline, *loss_curve) + 0.3)
	ax.plot_xy(steps_x, loss_curve, oa.plot.LineStyle(label="train", width=1.6))
	ax.plot_xy(
		[steps_x[0], steps_x[-1]],
		[baseline, baseline],
		oa.plot.LineStyle(color=(0.565, 0.565, 0.565, 1.0), label=f"ln({vocab_size}) baseline"),
	)
	fig.save_to(path)

# Fit a linear regression model in Python

The smallest complete OA training loop: five scalar samples, one `Linear`
layer, AdamW, and the `ItTraining` step lifecycle that captures the recorded
step once and replays it for the rest of the schedule. This walkthrough
follows the same experiment as the C++ `tuMnistClassifierAg` tutorial. The
output below is from the Python run with OA 0.18.0 on Intel Iris Xe with
Mesa 26.3.1.

## Run in Python

From the OASDK root, with the sibling `oa-dev` checkout present, install its
native package into an activated virtual environment with `maturin`.

```sh
(cd ../oa-dev && maturin develop --release)

python src/py/tutorials/ml/tu_ml_basics/tu_ml_basics.py
```

## 1. Create the training pair

The model learns the line `y = 2x − 1` from five samples. A `None` runtime
argument resolves the calling thread's default runtime, and constructors
that accept no runtime use it as well.

```python
x = oa.Matrix.new_from_f32([SAMPLES, 1], [-2.0, -1.0, 0.0, 1.0, 2.0], None)
target = oa.Matrix.new_from_f32([SAMPLES, 1], [-3.0, -1.0, 1.0, 3.0, 5.0], None)
```

## 2. Build the model and measure the initial loss

`Linear` initializes weight and bias with a seeded Glorot uniform draw and a
zero bias, giving two trainable parameters. AdamW receives weight decay `0.0`
so the update matches plain Adam. An eager forward pass and `mse` give the
loss before training.

```python
model = oa.ml.nn.Linear(1, 1, seed=0)
optimizer = oa.ml.AdamW(model.parameters(), learning_rate=0.05, weight_decay=0.0)
initial_loss = oa.ml.metric.scalar_loss(oa.ml.loss.mse(model.forward(x), target))
```

## 3. Drive the schedule through ItTraining

`ItTraining` resolves the default runtime and requires an AdamW optimizer:
whole program capture is implemented for AdamW. The iterator owns step
counting, epoch accounting, callback dispatch, and the capture and replay
lifecycle. `ProgressBar` renders the progress bar for each epoch and
`TrainingSummary` prints the final totals.

```python
training = oa.ml.ItTraining(
	optimizer,
	oa.ml.ItTrainingConfig(
		total_steps=STEPS,
		steps_per_epoch=STEPS_PER_EPOCH,
		batch_size=SAMPLES,
	),
)
training.add_callback(oa.ml.ProgressBar())
training.add_callback(oa.ml.TrainingSummary())
```

## 4. Record one step, replay seventy nine

Each `step` uploads the batch's row indices through `prepare`, then `record`
gathers the batch, runs forward, `mse`, and `backward` under a `GradientTape`
and returns the loss. The first step runs eagerly; the second is captured
into a compiled command that replays for every subsequent step. `snapshot()`
reports the mean loss for the completed epoch.

```python
def record(indices):
	batch = x.gather(indices)
	targets = target.gather(indices)
	with oa.ml.GradientTape() as tape:
		prediction = model.forward(batch)
		loss = oa.ml.loss.mse(prediction, targets)
		tape.backward(loss)
	return loss

for epoch in range(1, EPOCHS + 1):
	for _ in range(STEPS_PER_EPOCH):
		if not training.step(lambda: oa.Matrix.new_from_u32([SAMPLES], rows, None), record):
			raise ValueError("Training stopped before the schedule finished")
```

```text
epoch 1/5
  16/16 |██████████| 0.01s · 0.44 ms/step · 11.33K sample/s · loss: 0.0867
  epoch 1: mean loss 0.74184

…

epoch 5/5
  16/16 |██████████| 0.00s · 0.28 ms/step · 18.06K sample/s · loss: 0.0000
  epoch 5: mean loss 0.00077
```

## 5. Check the replay and the fit

`finish` completes the lifecycle and returns the step totals; the diagnostics
confirm the recorded step was compiled once and replayed 79 times without a
capture fallback. The final assertions require the loss below 1% of its
initial value and every prediction within 0.1 of the target.

```text
Summary:
  loss: initial 1.903632 · final 0.000050 · mean 0.160425
  Wall: 0.47 ms/step · 10.72K sample/s
  run: 0.04s · 80 steps · batch 5
Compiled step: 10 nodes · 79 cache hits
loss: 1.903632 -> 0.000018
[-3.0058205, -1.0049047, 0.9960111, 2.9969268, 4.9978428]
```

The Rust program runs the identical schedule and prints the same loss
trajectory. Seeded initialization and the donor update kernels match across
languages.

## Source

[Python example](tu_ml_basics.py) ·
[Full Rust walkthrough](../../../../rs/tutorials/ml/tu_ml_basics/tu_ml_basics.md)

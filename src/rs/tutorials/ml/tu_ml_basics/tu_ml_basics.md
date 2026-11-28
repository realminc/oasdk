# Fit a linear regression model in Rust

The smallest complete OA training loop: five scalar samples, one `Linear`
layer, AdamW, and the `ItTraining` step lifecycle that captures the recorded
step once and replays it for the rest of the schedule. This walkthrough
follows the same experiment as the C++ `tuMnistClassifierAg` tutorial. The
output below is from the Rust run with OA 0.18.0 on Intel Iris Xe with
Mesa 26.3.1.

## Run in Rust

From the OASDK root, with `oa-dev` as a sibling checkout:

```sh
cargo run --release --example tu_ml_basics
```

## 1. Create the training pair

The model learns the line `y = 2x − 1` from five samples. `Runtime::current()`
returns this thread's lazily initialized default runtime, and constructors
that omit an explicit runtime upload on it.

```rust
let runtime = Runtime::current()?;
let x = Matrix::new_from_slice([SAMPLES, 1], &[-2.0, -1.0, 0.0, 1.0, 2.0])?;
let target = Matrix::new_from_slice([SAMPLES, 1], &[-3.0, -1.0, 1.0, 3.0, 5.0])?;
```

## 2. Build the model and measure the initial loss

`Linear::new_with_seed` initializes weight and bias with a seeded Glorot
uniform draw and a zero bias, giving two trainable parameters. AdamW receives
weight decay `0.0` so the update matches plain Adam. An eager forward pass and
`mse` give the loss before training.

```rust
let model = Linear::new_with_seed(1, 1, 0)?;
let mut optimizer = AdamW::new_with_hyperparameters(
	model.all_parameters()?, 0.05, 0.9, 0.999, 1e-7, 0.0,
)?;
let initial_loss = f64::from(metric::scalar_loss(&loss::mse(&model.forward(&x)?, &target)?)?);
```

## 3. Drive the schedule through ItTraining

`ItTraining::new` borrows the default runtime and requires an AdamW
optimizer: whole program capture is implemented for AdamW. The iterator owns
step counting, epoch accounting, callback dispatch, and the capture and
replay lifecycle. `ProgressBar` renders the progress bar for each epoch and
`TrainingSummary` prints the final totals.

```rust
let mut progress = ProgressBar::default();
let mut summary = TrainingSummary::default();
let mut training = ItTraining::new(
	&runtime,
	&mut optimizer,
	ItTrainingConfig {
		total_steps: STEPS,
		steps_per_epoch: STEPS_PER_EPOCH,
		batch_size: SAMPLES as u64,
		..ItTrainingConfig::default()
	},
)?;
training.add_callback(&mut progress);
training.add_callback(&mut summary);
```

## 4. Record one step, replay seventy nine

Each `step` uploads the batch's row indices through `prepare`, then `record`
gathers the batch, runs forward, `mse`, and `backward` under a `GradientTape`
and returns the loss. The first step runs eagerly; the second is captured
into a compiled command that replays for every subsequent step. `snapshot()`
reports the mean loss for the completed epoch.

```rust
let admitted = training.step(
	|| Matrix::new_from_slice([SAMPLES], &rows),
	|indices| {
		let batch = x.gather(&indices)?;
		let targets = target.gather(&indices)?;
		let tape = GradientTape::new();
		let prediction = model.forward(&batch)?;
		let loss = loss::mse(&prediction, &targets)?;
		tape.backward(&loss)?;
		Ok(loss)
	},
)?;
```

```text
epoch 1/5
  16/16 |██████████| 0.01s · 0.65 ms/step · 7.73K sample/s · loss: 0.0867
  epoch 1: mean loss 0.74184

…

epoch 5/5
  16/16 |██████████| 0.01s · 0.32 ms/step · 15.47K sample/s · loss: 0.0000
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
  Wall: 0.56 ms/step · 8.88K sample/s
  run: 0.05s · 80 steps · batch 5
Compiled step: 10 nodes · 79 cache hits
loss: 1.903632 -> 0.000018
[-3.0058205, -1.0049047, 0.9960111, 2.9969268, 4.997843]
```

The Python program runs the identical schedule and prints the same loss
trajectory. Seeded initialization and the donor update kernels match across
languages.

## Source

[Rust example](tu_ml_basics.rs) ·
[Full Python walkthrough](../../../../py/tutorials/ml/tu_ml_basics/tu_ml_basics.md)

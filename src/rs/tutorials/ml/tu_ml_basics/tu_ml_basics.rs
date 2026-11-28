//! Fit a one-layer regression model with OA autograd and the ItTraining iterator.

use anyhow::{Context, Result, ensure};
use oa::ml::nn::Linear;
use oa::ml::{
	AdamW, GradientTape, ItTraining, ItTrainingConfig, Module, ProgressBar, TrainingSummary, loss,
	metric,
};
use oa::{Matrix, Runtime};

const SAMPLES: usize = 5;
const EPOCHS: u64 = 5;
const STEPS_PER_EPOCH: u64 = 16;
const STEPS: u64 = EPOCHS * STEPS_PER_EPOCH;

fn main() -> Result<()> {
	let runtime = Runtime::current()?;
	let x = Matrix::new_from_slice([SAMPLES, 1], &[-2.0, -1.0, 0.0, 1.0, 2.0])?;
	let target = Matrix::new_from_slice([SAMPLES, 1], &[-3.0, -1.0, 1.0, 3.0, 5.0])?;

	let model = Linear::new_with_seed(1, 1, 0)?;
	let parameters = model.all_parameters()?;
	ensure!(
		parameters.len() == 2,
		"expected one weight and one bias parameter"
	);
	let mut optimizer = AdamW::new_with_hyperparameters(parameters, 0.05, 0.9, 0.999, 1e-7, 0.0)?;

	let initial_loss = f64::from(metric::scalar_loss(&loss::mse(
		&model.forward(&x)?,
		&target,
	)?)?);

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

	let rows = (0..SAMPLES as u32).collect::<Vec<_>>();
	for epoch in 1..=EPOCHS {
		for _ in 0..STEPS_PER_EPOCH {
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
			ensure!(admitted, "training stopped before the schedule finished");
		}
		println!(
			"  epoch {epoch}: mean loss {:.5}",
			training.snapshot().epoch_mean_loss()
		);
	}

	let diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let fallbacks = training.program_capture_fallback_count();
	let result = training.finish()?;
	let diagnostics = diagnostics.context("training capture did not produce a compiled step")?;
	ensure!(
		diagnostics.command_cache_hit_count() > 0 && fallbacks == 0,
		"expected cached training replay, observed {fallbacks} capture fallbacks"
	);
	ensure!(
		result.step_count() == STEPS && u64::from(optimizer.step_count()) == STEPS,
		"AdamW update count disagrees with the training schedule"
	);
	println!(
		"Compiled step: {} nodes · {} cache hits",
		diagnostics.node_count(),
		diagnostics.command_cache_hit_count()
	);

	let final_loss = f64::from(metric::scalar_loss(&loss::mse(
		&model.forward(&x)?,
		&target,
	)?)?);
	let values = model.forward(&x)?.read_f32()?;

	ensure!(
		final_loss < initial_loss * 0.01,
		"training loss did not fall"
	);
	let expected = [-3.0_f32, -1.0, 1.0, 3.0, 5.0];
	ensure!(
		values
			.iter()
			.zip(expected)
			.all(|(&a, b)| (a - b).abs() < 0.1),
		"predictions did not converge to the target line"
	);

	println!("loss: {initial_loss:.6} -> {final_loss:.6}");
	println!("{values:?}");
	Ok(())
}

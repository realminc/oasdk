//! Train and inspect the Fashion-MNIST dense classifier with OA and OADS.

mod plots;

use std::{fs, path::PathBuf, rc::Rc};

use anyhow::{Context, Result, bail, ensure};
use oa::ml::nn::{Flatten, Linear, Relu, Sequential};
use oa::ml::{
	AdamW, GradientTape, ItTraining, ItTrainingConfig, Module, ProgressBar, TrainingSummary, loss,
	metric,
};
use oa::plot::{Figure, LineStyle};
use oa::{Matrix, Runtime};
use oads::image::{FashionMnist, FashionMnistConfig, FashionMnistSplit};
use serde_json::json;

const SIDE: usize = 28;
const PIXELS: usize = SIDE * SIDE;
const CLASSES: [&str; 10] = [
	"T-shirt/top",
	"Trouser",
	"Pullover",
	"Dress",
	"Coat",
	"Sandal",
	"Shirt",
	"Sneaker",
	"Bag",
	"Ankle boot",
];
const BATCH_SIZE: usize = 32;
const SEED: u64 = 42;
type Scores = [f32; CLASSES.len()];

fn argmax(row: &Scores) -> usize {
	let mut best = 0;
	for index in 1..row.len() {
		if row[index] > row[best] {
			best = index;
		}
	}
	best
}

fn load_split(
	runtime: &Runtime,
	split: FashionMnistSplit,
	count: usize,
) -> Result<(Matrix, Matrix, Vec<usize>)> {
	let data = FashionMnist::load_on(
		runtime,
		&FashionMnistConfig {
			split,
			normalize: false,
			..FashionMnistConfig::default()
		},
	)?;
	ensure!(
		data.images().shape() == [count, 1, SIDE, SIDE],
		"unexpected {split:?} image shape"
	);
	ensure!(
		data.labels().shape() == [count],
		"unexpected {split:?} label shape"
	);
	let raw_labels = data.labels().read_f32()?;
	ensure!(
		raw_labels
			.iter()
			.all(|&value| (0.0..CLASSES.len() as f32).contains(&value) && value.fract() == 0.0),
		"invalid class index in {split:?}",
	);
	let labels = raw_labels
		.iter()
		.map(|&value| value as usize)
		.collect::<Vec<_>>();
	let mut frequencies = [0; CLASSES.len()];
	for &label in &labels {
		frequencies[label] += 1;
	}
	ensure!(
		frequencies == [count / CLASSES.len(); CLASSES.len()],
		"unexpected {split:?} class frequencies"
	);
	println!(
		"{split:?}: {:?}, {} labels, {:?}",
		data.images().shape(),
		labels.len(),
		&labels[..10]
	);
	Ok((
		data.images().reshape([count, PIXELS])?,
		data.labels().reshape([count, 1])?,
		labels,
	))
}

struct EpochMetrics {
	loss: f64,
	accuracy: f64,
}

fn fit(
	runtime: &Runtime,
	model: &Sequential,
	images: &Matrix,
	labels: &Matrix,
	optimizer: &mut AdamW,
	epochs: u32,
) -> Result<Vec<EpochMetrics>> {
	let count = images.shape()[0];
	let steps_per_epoch = (count / BATCH_SIZE) as u64;
	let total_steps = u64::from(epochs) * steps_per_epoch;
	let mut progress = ProgressBar::default();
	let mut summary = TrainingSummary::default();
	let mut training = ItTraining::new(
		runtime,
		optimizer,
		ItTrainingConfig {
			total_steps,
			steps_per_epoch,
			batch_size: BATCH_SIZE as u64,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_callback(&mut progress);
	training.add_callback(&mut summary);

	let mut history = Vec::new();
	let mut hits = None;
	for epoch in 1..=epochs {
		let mut order = (0..count as i64).collect::<Vec<_>>();
		oa::data::shuffle(&mut order, SEED + u64::from(epoch - 1))?;
		let mut correct = 0_u32;
		for rows in order.chunks(BATCH_SIZE) {
			let rows = rows.iter().map(|&row| row as u32).collect::<Vec<_>>();
			let admitted = training.step(
				|| Matrix::new_from_slice_on(runtime, [BATCH_SIZE], &rows),
				|indices| {
					let batch = images
						.gather(&indices)?
						.div_scalar(255.0)?
						.reshape([BATCH_SIZE, SIDE, SIDE])?;
					let targets = labels
						.gather(&indices)?
						.reshape([BATCH_SIZE])?
						.cast_f32_i32()?;
					let tape = GradientTape::new();
					let logits = model.forward(&batch)?;
					let loss = loss::cross_entropy(&logits, &targets)?;
					tape.backward(&loss)?;
					// Retain the captured counter for observation after each replay.
					hits = Some(logits.categorical_accuracy_count(&targets)?);
					Ok(loss)
				},
			)?;
			ensure!(admitted, "training stopped before the epoch finished");
			correct += hits
				.as_ref()
				.context("missing training accuracy counter")?
				.read::<u32>()?[0];
		}
		let loss = training.snapshot().epoch_mean_loss();
		let accuracy = f64::from(correct) / count as f64;
		ensure!(loss.is_finite(), "nonfinite training loss at epoch {epoch}");
		history.push(EpochMetrics { loss, accuracy });
		println!(
			"  mean loss: {loss:.5} · accuracy: {:.2}%",
			100.0 * accuracy
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
		result.step_count() == total_steps && u64::from(optimizer.step_count()) == total_steps,
		"AdamW update count disagrees with the training schedule"
	);
	println!(
		"Compiled step: {} nodes · {} cache hits",
		diagnostics.node_count(),
		diagnostics.command_cache_hit_count()
	);
	Ok(history)
}

fn evaluate(
	runtime: &Runtime,
	model: &Sequential,
	images: &Matrix,
	labels: &[usize],
) -> Result<(f64, f64, Vec<Scores>)> {
	let count = labels.len();
	let mut probabilities = Vec::with_capacity(count);
	let (mut loss_sum, mut correct) = (0.0, 0_u32);
	for (batch, rows) in labels.chunks(BATCH_SIZE).enumerate() {
		let start = batch * BATCH_SIZE;
		let end = start + rows.len();
		let target_values = rows.iter().map(|&row| row as u32).collect::<Vec<_>>();
		let targets = Matrix::new_from_slice_on(runtime, [rows.len()], &target_values)?;
		let input = images
			.slice(0, start as i64, end as i64)?
			.div_scalar(255.0)?
			.reshape([rows.len(), SIDE, SIDE])?;
		let logits = model.forward(&input)?;
		let loss = loss::cross_entropy(&logits, &targets)?;
		let hits = logits.categorical_accuracy_count(&targets)?;
		let scores = logits.softmax(-1)?.read_f32()?;
		ensure!(
			scores.len() == rows.len() * CLASSES.len(),
			"unexpected softmax shape"
		);
		for row in scores.as_chunks::<{ CLASSES.len() }>().0 {
			ensure!(
				row.iter().all(|value| (0.0..=1.0).contains(value))
					&& (row.iter().sum::<f32>() - 1.0).abs() <= 2e-5,
				"softmax did not return a normalized distribution",
			);
			probabilities.push(*row);
		}
		loss_sum += rows.len() as f64 * f64::from(metric::scalar_loss(&loss)?);
		correct += hits.read::<u32>()?[0];
	}
	let (mut host_correct, mut host_loss) = (0, 0.0);
	for (row, &label) in probabilities.iter().zip(labels) {
		host_correct += usize::from(argmax(row) == label);
		host_loss -= f64::from(row[label]).ln();
	}
	ensure!(
		host_correct == correct as usize && ((host_loss - loss_sum) / count as f64).abs() <= 2e-5,
		"device metrics disagree with host metrics from the probability rows",
	);
	Ok((
		loss_sum / count as f64,
		f64::from(correct) / count as f64,
		probabilities,
	))
}

fn main() -> Result<()> {
	let mut epochs = 10_u32;
	let mut output = PathBuf::from("var/tutorials/fashion_mnist/rust");
	let mut args = std::env::args().skip(1);
	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--epochs" => epochs = args.next().context("missing --epochs value")?.parse()?,
			"--output" => output = args.next().context("missing --output path")?.into(),
			"--help" | "-h" => {
				println!("Usage: tu_ml_fashion_mnist [--epochs N] [--output DIR]");
				return Ok(());
			}
			_ => bail!("unknown option {arg}"),
		}
	}
	ensure!(epochs > 0, "--epochs must be positive");
	fs::create_dir_all(&output)?;
	let runtime = Runtime::new()?;
	let save = |figure: Figure, name: &str| -> Result<()> {
		let path = output.join(format!("{name}.png"));
		figure.save_to_on(&runtime, path.to_str().context("plot path must be UTF-8")?)?;
		Ok(())
	};

	// 1. Inspect both official OADS splits before scaling.
	let (train, train_labels, train_classes) =
		load_split(&runtime, FashionMnistSplit::Train, 60_000)?;
	let (test, _, test_classes) = load_split(&runtime, FashionMnistSplit::Test, 10_000)?;
	ensure!(
		train_classes[..3] == [9, 0, 0] && test_classes[..2] == [9, 2],
		"unexpected sample order"
	);
	let first = train.slice(0, 0, 1)?.read_f32()?;
	ensure!(
		first.contains(&0.0) && first.contains(&255.0),
		"expected the first raw image to span 0 to 255"
	);
	let mut raw_plot = plots::figure("First training image · raw pixel values", 1, 1, 407, 400);
	plots::image_plot(raw_plot.ax(0, 0), &first, true);
	save(raw_plot, "01_raw_image")?;

	// 2. Use the same OA scaling for each batch, then preview the first 25 training images.
	let preview = train.slice(0, 0, 25)?.div_scalar(255.0)?.read_f32()?;
	ensure!(
		first
			.iter()
			.zip(&preview)
			.all(|(&raw, &scaled)| (raw / 255.0 - scaled).abs() <= 1e-7),
		"raw and normalized pixels disagree",
	);
	let mut grid = plots::figure("First 25 training images", 5, 5, 990, 1135);
	for (index, pixels) in preview.as_chunks::<PIXELS>().0.iter().enumerate() {
		let axes = grid.ax(index as i32 / 5, index as i32 % 5);
		plots::image_plot(axes, pixels, false);
		axes.caption(CLASSES[train_classes[index]]);
	}
	save(grid, "02_training_images")?;

	// 3. Build Flatten → Dense(128) → ReLU → Dense(10) with trainable biases.
	let mut model = Sequential::new();
	model.add(Rc::new(Flatten::new(1, -1)))?;
	model.add(Rc::new(Linear::new_with_seed_on(
		&runtime, PIXELS, 128, SEED,
	)?))?;
	model.add(Rc::new(Relu::new()))?;
	model.add(Rc::new(Linear::new_with_seed_on(
		&runtime,
		128,
		CLASSES.len(),
		SEED + 1,
	)?))?;
	let parameters = model.all_parameters()?;
	let parameter_count: usize = parameters
		.iter()
		.map(|parameter| parameter.data().num_elements())
		.sum();
	ensure!(
		parameters.len() == 4 && parameter_count == 101_770,
		"unexpected classifier parameter count"
	);
	let mut optimizer = AdamW::new_with_hyperparameters(parameters, 0.001, 0.9, 0.999, 1e-7, 0.0)?;
	println!("Flatten (784) → Linear (128) → ReLU → Linear (10), {parameter_count} parameters");

	// 4. Fit on training logits, then evaluate the complete held-out split.
	let history = fit(
		&runtime,
		&model,
		&train,
		&train_labels,
		&mut optimizer,
		epochs,
	)?;
	let mut plot = plots::figure("Training history", 1, 2, 840, 300);
	let x = (1..=epochs).map(|epoch| epoch as f32).collect::<Vec<_>>();
	for (column, title) in ["Loss", "Accuracy"].into_iter().enumerate() {
		let axes = plot.ax(0, column as i32);
		axes.title(title);
		axes.x_label("Epoch");
		let y = history
			.iter()
			.map(|row| {
				if column == 0 {
					row.loss as f32
				} else {
					row.accuracy as f32
				}
			})
			.collect::<Vec<_>>();
		axes.plot_xy(&x, &y, LineStyle::default());
	}
	save(plot, "03_training_history")?;
	let (test_loss, test_accuracy, probabilities) = evaluate(&runtime, &model, &test, &test_classes)?;
	println!(
		"test: loss={test_loss:.5} accuracy={:.2}%",
		100.0 * test_accuracy
	);

	// 5. Inspect test probabilities. Captions show predicted class, confidence, and truth.
	println!(
		"test[0]: {:?} → {} ({})",
		probabilities[0],
		CLASSES[argmax(&probabilities[0])],
		CLASSES[test_classes[0]]
	);
	let preview = test.slice(0, 0, 15)?.div_scalar(255.0)?.read_f32()?;
	for index in [0, 12] {
		save(
			plots::prediction_grid(
				&preview,
				&test_classes,
				&probabilities,
				&[index],
				1,
				&format!("Test image {index}"),
			),
			&format!("04_prediction_{index:02}"),
		)?;
	}
	save(
		plots::prediction_grid(
			&preview,
			&test_classes,
			&probabilities,
			&(0..15).collect::<Vec<_>>(),
			3,
			"First 15 test predictions",
		),
		"05_prediction_grid",
	)?;

	// A fixed preview can contain only correct predictions. Show the most confident error too.
	let error = probabilities
		.iter()
		.zip(&test_classes)
		.enumerate()
		.filter_map(|(index, (row, &label))| {
			let predicted = argmax(row);
			(predicted != label).then_some((index, row[predicted]))
		})
		.max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
	if let Some((index, _)) = error {
		let pixels = test
			.slice(0, index as i64, index as i64 + 1)?
			.div_scalar(255.0)?
			.read_f32()?;
		save(
			plots::prediction_grid(
				&pixels,
				&test_classes[index..index + 1],
				&probabilities[index..index + 1],
				&[0],
				1,
				&format!("Most confident error · test image {index}"),
			),
			"05_confident_error",
		)?;
	}

	// 6. Insert an explicit batch axis and predict test image 1 on its own.
	let image = test
		.slice(0, 1, 2)?
		.div_scalar(255.0)?
		.reshape([SIDE, SIDE])?;
	let single = image.reshape([1, SIDE, SIDE])?;
	let single_scores: Scores = model
		.forward(&single)?
		.softmax(-1)?
		.read_f32()?
		.try_into()
		.map_err(|_| anyhow::anyhow!("unexpected singleton softmax shape"))?;
	ensure!(
		single_scores
			.iter()
			.zip(probabilities[1])
			.all(|(&a, b)| (a - b).abs() <= 2e-5),
		"singleton prediction disagrees with batched test row 1",
	);
	println!(
		"test[1]: {:?} → {:?} → {}",
		image.shape(),
		single.shape(),
		CLASSES[argmax(&single_scores)]
	);
	let mut plot = plots::figure("Single image · test image 1", 1, 1, 1100, 330);
	plots::probability_plot(plot.ax(0, 0), &single_scores, test_classes[1], true);
	save(plot, "06_single_image")?;

	let results = json!({
		"class_names": CLASSES, "parameter_count": parameter_count, "batch_size": BATCH_SIZE,
		"optimizer_steps": optimizer.step_count(), "history": history.iter().enumerate().map(|(index, row)| json!({"epoch": index + 1, "loss": row.loss, "accuracy": row.accuracy})).collect::<Vec<_>>(), "test_loss": test_loss,
		"test_accuracy": test_accuracy, "test_labels": test_classes, "probabilities": probabilities,
		"single_probabilities": single_scores,
	});
	fs::write(output.join("results.json"), serde_json::to_vec(&results)?)?;
	println!("Wrote plots and results to {}", output.display());
	if epochs == 10 {
		let first_loss = history[0].loss;
		let last_loss = history[9].loss;
		ensure!(
			last_loss < first_loss && test_accuracy >= 0.85 && test_loss < 0.6,
			"the 10-epoch run missed the Fashion-MNIST convergence gate"
		);
	}
	Ok(())
}

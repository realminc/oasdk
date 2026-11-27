use oa::Color;
use oa::plot::{Axes, BarStyle, Figure, FigureConfig, HeatmapStyle, Theme};

use super::{CLASSES, PIXELS, SIDE, Scores, argmax};

const BLUE: Color = Color::new(0.35, 0.62, 1.0, 1.0);
const RED: Color = Color::new(1.0, 0.40, 0.43, 1.0);
const GRAY: Color = Color::new(0.47, 0.47, 0.47, 1.0);
const CLASS_IDS: [&str; CLASSES.len()] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

pub(super) fn figure(title: &str, rows: i32, cols: i32, width: u32, height: u32) -> Figure {
	Figure::new(FigureConfig {
		title: title.into(),
		rows,
		cols,
		width,
		height,
		theme: Theme::Dark,
		..FigureConfig::default()
	})
}

pub(super) fn image_plot(axes: &mut Axes, pixels: &[f32], raw: bool) {
	let grayscale;
	let values = if raw {
		pixels
	} else {
		grayscale = pixels.iter().map(|pixel| 1.0 - pixel).collect::<Vec<_>>();
		&grayscale
	};
	axes.heatmap(
		values,
		SIDE as i32,
		SIDE as i32,
		HeatmapStyle {
			v_max: if raw { 255.0 } else { 1.0 },
			colormap: if raw { 1 } else { 3 },
			auto_scale: false,
			show_grid: false,
			colorbar: raw,
			..HeatmapStyle::default()
		},
	);
	axes.grid(false);
}

pub(super) fn probability_plot(axes: &mut Axes, scores: &Scores, label: usize, names: bool) {
	let predicted = argmax(scores);
	let mut bars = [[f32::NAN; CLASSES.len()]; 3];
	for (index, &score) in scores.iter().enumerate() {
		let series = if index == label {
			2
		} else if index == predicted {
			1
		} else {
			0
		};
		bars[series][index] = score;
	}
	for (values, color) in bars.iter().zip([GRAY, RED, BLUE]) {
		axes.bar(
			values,
			BarStyle {
				color,
				..BarStyle::default()
			},
		);
	}
	axes.categories(if names { &CLASSES } else { &CLASS_IDS });
	axes.limits(-0.5, 9.5, 0.0, 1.0);
	axes.yticks(&[0.0, 0.5, 1.0]);
	axes.grid(false);
	axes.legend(false);
}

pub(super) fn prediction_grid(
	images: &[f32],
	labels: &[usize],
	probabilities: &[Scores],
	indices: &[usize],
	columns: i32,
	title: &str,
) -> Figure {
	let rows = indices.len() as i32 / columns;
	let mut figure = figure(
		title,
		rows,
		2 * columns,
		600 * columns as u32,
		312 * rows as u32 + 54,
	);
	for (position, &index) in indices.iter().enumerate() {
		let (row, col) = (position as i32 / columns, position as i32 % columns);
		let (scores, label) = (&probabilities[index], labels[index]);
		let predicted = argmax(scores);
		let axes = figure.ax(row, 2 * col);
		image_plot(axes, &images[index * PIXELS..(index + 1) * PIXELS], false);
		axes.caption_color(
			&format!(
				"{} {:.0}% ({})",
				CLASSES[predicted],
				scores[predicted] * 100.0,
				CLASSES[label]
			),
			if predicted == label { BLUE } else { RED },
		);
		probability_plot(figure.ax(row, 2 * col + 1), scores, label, false);
	}
	figure
}

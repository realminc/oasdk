# Recognize speech commands in Rust

Train a CNN to recognize eight spoken commands. This walkthrough follows the
[TensorFlow simple audio tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)
using OA for signal processing and training and OADS for the complete 8,000-clip
mini Speech Commands pack used in that tutorial.

## Run in Rust

Run from the OASDK root.

```sh
cargo run --release --example tu_ml_audio_commands -- --epochs 10
```

The default schedule uses up to ten epochs and 64 clips per batch.
Set `OA_DATA_DIR` to share the OADS cache. Use `--preview-only` for waveform
and spectrogram inspection or `--steps 32 --batch 8` for a short replay check.
Select a specific Vulkan device with `--device INDEX`.

## 1. Load and inspect the audio

OADS verifies and loads every recording in the mini pack. It scales PCM to
`[-1, 1]` and pads each mono 16 kHz clip to 16,000 samples. Each command has
1,000 examples. The class IDs are `down`, `go`, `left`, `no`, `right`,
`stop`, `up`, and `yes`.

```rust
let data = MiniSpeechCommands::load_on(&runtime, &MiniSpeechCommandsConfig::default())?;
println!("{:?} {:?}", data.samples().shape(), data.labels().shape());
```

The input and label shapes are `[8000, 1, 16000]` and `[8000]`. Both
programs use OA's seed-42 shuffle to reserve 6,400 clips for training, 800
for validation, and 800 for the final test. This gives matching Rust and
Python splits with the same proportions as the reference. The TensorFlow
loader uses its own file order and shuffle.

![Nine one-second training waveforms labeled by word](asset/01_waveforms.jpg)

[Play clip 1](asset/clip_1.wav)

[Play clip 2](asset/clip_2.wav)

[Play clip 3](asset/clip_3.wav)

## 2. Convert waveforms to spectrograms

A short-time Fourier transform measures how a clip's frequency content
changes over time. Apply a periodic Hann window to 255 samples, advance by
128 samples and use a 256-point FFT. OA returns the magnitudes of 129
nonnegative-frequency bins in 124 frames. Add a channel axis so the CNN
receives `[batch, 1, 124, 129]` in OA's channel-first layout.

```rust
let magnitude = audio.stft(StftConfig {
    fft_size: 256, hop_size: 128, win_size: 255, center: false,
    ..Default::default()
})?;
let features = magnitude.reshape([batch, 1, 124, 129])?;
```

![The same clip as a waveform and as a time-frequency spectrogram](asset/02_stft.jpg)

The frequency plots use log magnitude to make quiet components easier to
spot. The CNN receives linear magnitudes.

![Nine labeled spectrograms from the same audio clips](asset/03_spectrograms.jpg)

## 3. Build the classifier

Fit a scalar mean and standard deviation on the training spectrograms.
Bilinearly resize each spectrogram to 32×32, then apply the fitted statistics
to training, validation, test and inference inputs. The tutorial caches the
resized features before training, so replay reuses the prepared batch shape.

| Layer | Output per clip |
| --- | --- |
| Bilinear resize and normalization | 1 × 32 × 32 |
| 3×3 convolution, ReLU | 32 × 30 × 30 |
| 3×3 convolution, ReLU | 64 × 28 × 28 |
| 2×2 max pooling and Dropout(0.25) | 64 × 14 × 14 |
| Flatten and dense ReLU | 128 |
| Dropout(0.5) and dense | 8 logits |

The model has 1,625,608 trainable values. The output logits go into sparse
cross-entropy. AdamW uses zero weight decay and the reference Adam settings
with a learning rate of 0.001 and epsilon of `1e-7`.

## 4. Train with compiled replay

`ItTraining` owns the optimizer updates, loss metric, progress bar, and
summary. The prepare callback uploads fresh data and integer labels. The
record callback builds the forward and backward pass. OA warms up eagerly,
captures a fixed-shape update, and replays the compiled step with new data.
The programs verify cached replay, a zero fallback count, and agreement
between completed steps and optimizer updates.

```rust
|(input, target)| {
    let tape = GradientTape::new();
    let logits = model.forward(&input)?;
    let loss = ml::loss::cross_entropy(&logits, &target)?;
    tape.backward(&loss)?;
    Ok(loss)
}
```

After each epoch, switch dropout off and evaluate the validation split.
Return to training mode for the next epoch. Stop after two consecutive epochs
without a lower validation loss, as in the reference.

The figures below come from the Rust run on Mesa llvmpipe. It completed 1,000
updates across 10 epochs with 64 clips per batch. Validation and test each
used all 800 clips in their respective splits.

```text
Split: 6400 train · 800 validation · 800 test

epoch 1/10
  100/100 |██████████| 640.66s · 6406.56 ms/step · 10 sample/s · 10 clip/s · loss: 1.4896
  100/100 — 640.7s 6407 ms/step · accuracy: 0.3695 · loss: 1.7173 · val_accuracy: 0.5687 · val_loss: 1.3222

epoch 2/10
  100/100 |██████████| 631.58s · 6315.83 ms/step · 10 sample/s · 10 clip/s · loss: 0.9025
  100/100 — 631.6s 6316 ms/step · accuracy: 0.5794 · loss: 1.1938 · val_accuracy: 0.6937 · val_loss: 0.9109

epoch 3/10
  100/100 |██████████| 647.30s · 6473.04 ms/step · 10 sample/s · 10 clip/s · loss: 0.7472
  100/100 — 647.3s 6473 ms/step · accuracy: 0.6656 · loss: 0.9184 · val_accuracy: 0.7550 · val_loss: 0.7556

epoch 4/10
  100/100 |██████████| 666.33s · 6663.31 ms/step · 10 sample/s · 10 clip/s · loss: 0.7240
  100/100 — 666.3s 6663 ms/step · accuracy: 0.7222 · loss: 0.7634 · val_accuracy: 0.7887 · val_loss: 0.6821

epoch 5/10
  100/100 |██████████| 650.52s · 6505.17 ms/step · 10 sample/s · 10 clip/s · loss: 0.7499
  100/100 — 650.5s 6505 ms/step · accuracy: 0.7631 · loss: 0.6610 · val_accuracy: 0.7875 · val_loss: 0.6187

epoch 6/10
  100/100 |██████████| 668.60s · 6686.01 ms/step · 10 sample/s · 10 clip/s · loss: 0.6433
  100/100 — 668.6s 6686 ms/step · accuracy: 0.7834 · loss: 0.5844 · val_accuracy: 0.8263 · val_loss: 0.5575

epoch 7/10
  100/100 |██████████| 772.45s · 7724.53 ms/step · 8 sample/s · 8 clip/s · loss: 0.4351
  100/100 — 772.5s 7725 ms/step · accuracy: 0.8205 · loss: 0.5022 · val_accuracy: 0.8325 · val_loss: 0.5283

epoch 8/10
  100/100 |██████████| 624.43s · 6244.27 ms/step · 10 sample/s · 10 clip/s · loss: 0.4586
  100/100 — 624.4s 6244 ms/step · accuracy: 0.8341 · loss: 0.4664 · val_accuracy: 0.8462 · val_loss: 0.4924

epoch 9/10
  100/100 |██████████| 625.37s · 6253.73 ms/step · 10 sample/s · 10 clip/s · loss: 0.4627
  100/100 — 625.4s 6254 ms/step · accuracy: 0.8466 · loss: 0.4179 · val_accuracy: 0.8387 · val_loss: 0.4837

epoch 10/10
  100/100 |██████████| 626.66s · 6266.61 ms/step · 10 sample/s · 10 clip/s · loss: 0.2645
  100/100 — 626.7s 6267 ms/step · accuracy: 0.8633 · loss: 0.3874 · val_accuracy: 0.8438 · val_loss: 0.4744

Summary:
  loss: initial 2.169622 · final 0.264475 · mean 0.761196
  Wall: 6727.78 ms/step · 10 sample/s · 10 clip/s
  GPU (commands_training_step): mean 6552.149 ms/step · p50 6301.367 · p95 7600.801 · min 5730.167 · max 22208.496 · 10 sample/s · 10 clip/s · wall-GPU gap 3%
  run: 6727.78s · 1000 steps · batch 64

Compiled step: 33 nodes · 999 cache hits
Test accuracy: 83.50% · test loss: 0.520289
WAV prediction: no (74.30%)
```

TensorFlow reports 82.33% test accuracy and 0.513843 test loss for its
published run. It uses a different split, so compare these numbers as
separate experiments. [TensorFlow test results](https://www.tensorflow.org/tutorials/audio/simple_audio#evaluate_the_model_performance)

![Training and validation loss and accuracy from the Rust run](asset/04_training_history.jpg)

## 5. Evaluate on the test split

After training, compute sample-weighted loss and accuracy over the 800 held-out
clips. The confusion matrix counts true labels down the rows and predictions
across the columns. Its diagonal sum must match OA's device-side count of
correct predictions. A uniform eight-way guess has a cross-entropy of
`ln(8) ≈ 2.0794`.

![Counts of true versus predicted commands from the Rust run](asset/05_confusion.jpg)

## 6. Predict the command in a WAV file

The example writes a held-out `no` clip to `sample.wav` and decodes it
through the inference path. Pass `--wav PATH` to use a different recording.
OA decodes to mono, resamples to 16 kHz when needed, and pads or trims to one
second. Apply the same STFT, resize, and standardization, then use softmax
on the eight logits.

[Play the inference clip](asset/inference.wav)

![Softmax probabilities for the example WAV file](asset/06_wav_prediction.jpg)

## 7. Save and reload inference

The run writes `commands.oam` and `preprocessing.json` to
`var/tutorials/audio_commands/rust/` by default. It reloads the model,
optimizer, and normalization statistics and verifies that the same waveform
gives identical scores. Run the saved classifier on a WAV file without
loading the dataset or training again.

```sh
cargo run --release --example tu_ml_audio_commands -- --load var/tutorials/audio_commands/rust --wav path/to/clip.wav
```

To regenerate the figures from the Rust run, use the shared plot script
with an environment containing `oapython`.

```sh
python src/py/tutorials/ml/tu_ml_audio_commands/plots.py \
  var/tutorials/audio_commands/rust src/rs/tutorials/ml/tu_ml_audio_commands/asset
```

[Complete Rust source](tu_ml_audio_commands.rs)

## Attribution

The walkthrough adapts the experiment from Google's
[Simple audio recognition tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio).
Google distributes the tutorial prose under CC BY 4.0 and its code samples
under Apache 2.0. The OA examples and plots were generated independently.
Google's mini Speech Commands recordings are distributed under
[CC BY 4.0](../../../../../license/mini_speech_commands/CC-BY-4.0.txt).
See the [dataset paper](https://arxiv.org/abs/1804.03209) for collection details.

# Assignment 1 Solution — Digital Signal Processing & Audio ML

## Part 1: LogMelFilterBanks Implementation

### Signal Processing Pipeline

```
Audio (B, T)  →  STFT  →  Power Spectrum  →  × Mel Matrix  →  log(· + ε)
```

### `__init__`

All constructor parameters are stored as instance attributes. The critical
subtlety is `f_max_hz`: when passed as `None`, it defaults to
`float(samplerate // 2)` (Nyquist frequency), exactly matching the internal
behaviour of `torchaudio.transforms.MelSpectrogram`. Both `window` and
`mel_fbanks` are registered with `register_buffer` so they move to the correct
device automatically when the module is transferred with `.to(device)`.

### `_init_melscale_fbanks`

Calls `torchaudio.functional.melscale_fbanks` with
`n_freqs = n_fft // 2 + 1 = 201` (the number of unique frequency bins in a
one-sided STFT for the default `n_fft=400`). Returns a filterbank matrix of
shape `(201, 80)`.

### `spectrogram`

Calls `torch.stft(return_complex=True)` → complex tensor of shape
`(B, 201, n_frames)`. Using `center=True` with `reflect` padding is required to
match the default behaviour of `torchaudio.transforms.Spectrogram`.

### `forward`

1. **Reshape** input to 2D `(B, T)`, handling both mono `(1, T)` and batched
   `(B, T)` inputs via `x.reshape(-1, shape[-1])`.
2. **STFT** → `(B, 201, 101)` complex (for a 1-second 16 kHz signal).
3. **Power spectrum**: `torch.abs(·) ** 2.0` → real tensor `(B, 201, 101)`.
4. **Mel projection**:
   ```
   power_spec.transpose(-2, -1)  @ mel_fbanks
   (B, 101, 201)                   (201, 80)
   = (B, 101, 80)  →  .transpose(-2,-1)  →  (B, 80, 101)
   ```
5. **Log compression**: `log(mel_spec + 1e-6)` — the same epsilon used in the
   README assertion.
6. **Restore shape**: `reshape(shape[:-1] + log_mel.shape[-2:])`.

The result passes the assertion with **zero numerical difference**:
```python
assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)  # True
# max |diff| = 0.00e+00
```

---

## Part 2: Training Pipeline

### Dataset — `YesNoDataset`

`YesNoDataset` reads the official Google Speech Commands v0.02 split files
(`validation_list.txt`, `testing_list.txt`) directly and scans only the `yes/`
and `no/` subdirectories.

| Split      | Samples |
|------------|---------|
| Training   | 6 358   |
| Validation | 803     |
| Testing    | 824     |

Labels: `yes → 0`, `no → 1`.

### Model — `SpeechCNN`

A lightweight 1-D convolutional network operating on log-mel features of shape
`(B, n_mels, 101)`:

| Layer                     | Output shape       | Params (groups=1) |
|---------------------------|--------------------|-------------------|
| Conv1d(80→64, k=3) + BN + ReLU  | (B, 64, 101)  | 15 424            |
| Conv1d(64→64, k=3) + BN + ReLU  | (B, 64, 101)  | 12 352            |
| Conv1d(64→128, k=3) + BN + ReLU | (B, 128, 101) | 24 832            |
| AdaptiveAvgPool1d(1)      | (B, 128, 1)        | 0                 |
| Linear(128→2)             | (B, 2)             | 258               |
| **Total**                 |                    | **53 250** (<100 K) |

### Lightning Module — `SpeechClassifier`

Combines `LogMelFilterBanks` (frozen-parameter feature extractor) with
`SpeechCNN`. Tracked metrics:

| Metric         | How                                         |
|----------------|---------------------------------------------|
| `train_loss`   | Cross-entropy, averaged per epoch           |
| `val_acc`      | `torchmetrics.MulticlassAccuracy` per epoch |
| `epoch_time_s` | `time.perf_counter()` wall-clock            |
| `test_acc`     | Evaluated once after training               |

Parameter count is computed via `sum(p.numel() for p in parameters())`.
FLOPs are computed via `thop.profile` on the CNN head with a dummy input of
shape `(1, n_mels, 101)`.

---

## Part 3: Experiment 1 — Effect of `n_mels`

All models trained for 15 epochs, `groups=1`, `batch_size=64`, Adam lr=1e-3.

| n_mels | Params  | FLOPs  | Test Accuracy | Mean Epoch Time |
|--------|---------|--------|---------------|-----------------|
| 20     | 41 730  | 4.23 M | **98.8%**     | 3.43 s          |
| 40     | 45 570  | 4.62 M | **99.0%**     | 3.38 s          |
| 80     | 53 250  | 5.39 M | **98.1%**     | 3.48 s          |

### Conclusions

- All three configurations achieve **≥98% test accuracy** on this binary task.
  The phonemic distinction between "yes" and "no" is captured well even with
  only 20 Mel bins.
- Increasing `n_mels` slightly increases the parameter count (larger first
  Conv1d input) and FLOPs but does not improve accuracy — the task is too easy.
- **Epoch time is almost identical** across `n_mels` values; the dominating cost
  is the CPU convolution, not the feature extraction.
- For harder, multi-class problems a higher `n_mels` (80–128) would be expected
  to matter more, since finer frequency resolution helps separate similar words.

---

## Part 4: Experiment 2 — Effect of `groups`

All models trained for 15 epochs, `n_mels=80`, `batch_size=64`, Adam lr=1e-3.

| groups | Params  | FLOPs  | Test Accuracy | Mean Epoch Time |
|--------|---------|--------|---------------|-----------------|
| 1      | 53 250  | 5.39 M | 97.5%         | 3.48 s          |
| 2      | 27 138  | 2.75 M | 97.7%         | 3.25 s          |
| 4      | 14 082  | 1.44 M | **98.7%**     | 3.62 s          |
| 8      | 7 554   | 0.78 M | 96.6%         | 3.12 s          |
| 16     | 4 290   | 0.45 M | 92.4%         | 3.13 s          |

### Conclusions

**Parameters and FLOPs** decrease linearly with `groups`: each doubling of
`groups` halves both quantities. Going from `groups=1` to `groups=16` reduces
parameter count by ~12× (53 250 → 4 290) and FLOPs by ~12× (5.39 M → 0.45 M).

**Epoch time is roughly constant (~3.1–3.6 s) across all `groups` values on
CPU.** Standard PyTorch CPU Conv1d kernels do not efficiently exploit the
sparsity introduced by grouped convolutions, so the theoretical FLOPs reduction
does not translate to a wall-clock speedup. On a GPU the picture would be
different.

**Test accuracy degrades at high `groups`**: at `groups=16` each convolutional
group sees only 5 input channels (`80 / 16`), severely limiting the spatial
mixing between frequency bands. The model still learns but has less expressive
power. The sweet spot here is around `groups=4`: roughly ¼ of the parameters of
the standard model yet slightly higher test accuracy — the grouping acts as a
light implicit regulariser for this small dataset.

**`groups=4` is the recommended baseline**: it achieves the best test accuracy
(98.7%) with only 14 082 parameters and 1.44 M FLOPs.

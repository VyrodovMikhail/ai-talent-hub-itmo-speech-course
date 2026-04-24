# GP1 Solution

## Task Analysis

- The project is a low-resource Russian ASR task for spoken numbers with a strict **train-from-scratch** rule and a **<5M parameter** limit.
- The official README states the transcription range is `[1000 .. 999999]`, but the provided metadata in `train.csv` includes smaller values such as `14`, `19`, and `931`. The implemented text normalization therefore supports the full `[0 .. 999999]` range.
- `train` and `dev` are speaker-shifted: `train` contains `spk_A..spk_F`, while `dev` mixes seen speakers with unseen speakers such as `spk_H..spk_K`. The training code reports overall CER, per-speaker CER, and a harmonic mean over in-domain vs out-of-domain CER for validation.
- `train` audio is all `wav` at `22.05/24 kHz`; `dev` includes both `wav` and `mp3` at `16 kHz`. The pipeline always loads audio and converts it to **16 kHz**.

## Implemented Approach

- Acoustic frontend: custom 16 kHz log-mel filterbanks built with `torch.stft` and a mel matrix, reusing the structure of assignment 1 without depending on `torchaudio`.
- Default objective: **structured chunk classification** with two heads predicting `thousands` and `remainder` in `[0..999]`, combined into the final integer.
- Targets: **character-level CTC** over normalized Russian spoken-number text, followed by deterministic inverse normalization back to digits for evaluation and submission.
- Acoustic model: a QuartzNet-style 1D separable-convolution encoder with residual blocks, scaled to about **3.62M parameters** for the structured classifier.
- Decoding: greedy CTC, unconstrained beam search, and **grammar-constrained beam search** that only permits valid Russian spoken-number phrases. Optional KenLM rescoring/shallow-fusion-style scoring remains available.
- Data prep: archive extraction, manifest generation, vocabulary build, LM corpus export, and optional KenLM artifact generation.

## Files

- `prepare_data.py`: extracts `train.tar` and `dev.tar`, writes manifests, vocabulary, lexicon, and LM corpus.
- `train.py`: trains the model and saves the best checkpoint by harmonic CER on dev.
- `infer.py`: runs decoding from a checkpoint and writes predictions to CSV.
- `build_lm.py`: prepares LM artifacts and trains a KenLM ARPA/binary model if `lmplz` is installed.
- `other files`: normalization, audio, dataset, model, decoder, and checkpoint utilities.

The dataset loader accepts both:
- raw split-local manifests such as `data/train/train.csv` and `data/dev/dev.csv`
- enriched manifests such as `data/manifests/train_manifest.csv`

If a manifest has only `filename,transcription,spk_id,gender,ext,samplerate`, the loader resolves the audio path relative to the manifest location and generates `spoken_transcription` automatically.

## Quick Start

```bash
cd group-projects/gp1
python3 train.py --epochs 25 --batch-size 32
python3 build_lm.py
python3 infer.py --checkpoint artifacts/baseline/best.pt --manifest data/dev/dev.csv
```

To use KenLM rescoring after building the LM:

```bash
python3 train.py --task-mode ctc --decode-mode beam_lm_rescore --lm-path artifacts/lm/3gram.arpa
```

## Notes

- This is a stronger baseline intended to be trainable on CPU for smoke tests and on GPU for full experiments.
- The normalization layer is deliberately deterministic so the acoustic model predicts valid spoken-number text, while evaluation still happens on digit strings.
- `task_mode=chunked` is the default and recommended mode for this isolated-number task.
- `beam_constrained` is still available for the legacy CTC path via `--task-mode ctc`.
- The provided `QuartzNet/` directory remains untouched and can still be used as an external reference for larger experiments.

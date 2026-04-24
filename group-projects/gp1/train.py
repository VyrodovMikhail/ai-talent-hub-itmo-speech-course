from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import jiwer
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .audio import WaveformAugment
from .data_utils import CTCVocabulary, SpokenNumbersDataset, ctc_collate_fn, read_manifest
from .decoding import CTCDecoder
from .model import build_model, count_parameters
from .numbers import all_number_chars, number_to_text, safe_parse_number_words
from .utils import save_checkpoint, save_json, seed_everything


def build_argparser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"

    parser = argparse.ArgumentParser(description="Train a Russian spoken-numbers recognizer.")
    parser.add_argument("--train-manifest", type=Path, default=data_root / "train" / "train.csv")
    parser.add_argument("--dev-manifest", type=Path, default=data_root / "dev" / "dev.csv")
    parser.add_argument("--vocab", type=Path, default=data_root / "vocab.json")
    parser.add_argument("--output-dir", type=Path, default=project_root / "artifacts" / "baseline")
    parser.add_argument("--task-mode", choices=["factorized", "chunked", "ctc"], default="factorized")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--dev-max-samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-val-samples", type=int, default=50)
    parser.add_argument("--val-samples-path", type=Path, default=data_root / "validation_samples.csv")
    parser.add_argument("--best-val-samples-path", type=Path, default=data_root / "validation_samples_best.csv")
    parser.add_argument(
        "--decode-mode",
        choices=["greedy", "beam", "beam_constrained", "beam_lm", "beam_lm_rescore"],
        default="greedy",
    )
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--beam-size-token", type=int, default=None)
    parser.add_argument("--lm-path", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--disable-waveform-augment", action="store_true")
    return parser


def decode_utterance(decoder: CTCDecoder, emission: torch.Tensor, mode: str) -> str:
    if mode == "greedy":
        return decoder.greedy_decode(emission)
    if mode == "beam":
        return decoder.beam_search_decode(emission)
    if mode == "beam_constrained":
        return decoder.constrained_beam_search_decode(emission)
    if mode == "beam_lm":
        return decoder.beam_search_with_lm(emission)
    if mode == "beam_lm_rescore":
        beams = decoder.beam_search_decode(emission, return_beams=True)
        return decoder.lm_rescore(beams)
    raise ValueError(f"Unsupported decode mode: {mode}")


def mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def predict_chunked_numbers(thousands_logits: torch.Tensor, remainder_logits: torch.Tensor) -> torch.Tensor:
    return thousands_logits.argmax(dim=-1) * 1000 + remainder_logits.argmax(dim=-1)


def predict_factorized_numbers(
    thousands_hundreds_logits: torch.Tensor,
    thousands_last2_logits: torch.Tensor,
    remainder_hundreds_logits: torch.Tensor,
    remainder_last2_logits: torch.Tensor,
) -> torch.Tensor:
    thousands = thousands_hundreds_logits.argmax(dim=-1) * 100 + thousands_last2_logits.argmax(dim=-1)
    remainder = remainder_hundreds_logits.argmax(dim=-1) * 100 + remainder_last2_logits.argmax(dim=-1)
    return thousands * 1000 + remainder


def save_validation_samples(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "speaker_id",
                "audio_path",
                "predicted_text",
                "golden_text",
                "predicted_digits",
                "golden_digits",
                "cer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    task_mode: str,
    ctc_loss: torch.nn.CTCLoss | None,
    ce_loss: torch.nn.CrossEntropyLoss | None,
    decoder: CTCDecoder | None,
    decode_mode: str,
    device: torch.device,
    seen_speakers: set[str],
    max_saved_samples: int,
) -> dict:
    model.eval()
    losses: list[float] = []
    sample_cers: list[float] = []
    in_domain_cers: list[float] = []
    ood_cers: list[float] = []
    per_speaker_cers: dict[str, list[float]] = defaultdict(list)
    parse_failures = 0
    saved_samples: list[dict[str, str]] = []

    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)

        if task_mode == "factorized":
            thousands_hundreds_targets = batch["thousands_hundreds_targets"].to(device)
            thousands_last2_targets = batch["thousands_last2_targets"].to(device)
            remainder_hundreds_targets = batch["remainder_hundreds_targets"].to(device)
            remainder_last2_targets = batch["remainder_last2_targets"].to(device)
            (
                thousands_hundreds_logits,
                thousands_last2_logits,
                remainder_hundreds_logits,
                remainder_last2_logits,
            ) = model(waveforms, waveform_lengths, apply_spec_augment=False)
            loss = (
                ce_loss(thousands_hundreds_logits, thousands_hundreds_targets)
                + ce_loss(thousands_last2_logits, thousands_last2_targets)
                + ce_loss(remainder_hundreds_logits, remainder_hundreds_targets)
                + ce_loss(remainder_last2_logits, remainder_last2_targets)
            )
            losses.append(loss.item())
            predicted_numbers = predict_factorized_numbers(
                thousands_hundreds_logits,
                thousands_last2_logits,
                remainder_hundreds_logits,
                remainder_last2_logits,
            ).cpu().tolist()

            for index, predicted_number in enumerate(predicted_numbers):
                predicted_digits = str(predicted_number)
                predicted_text = number_to_text(predicted_number)
                reference_digits = batch["reference_digits"][index]
                reference_text = batch["reference_words"][index]
                speaker_id = batch["speaker_ids"][index]
                sample_cer = jiwer.cer(reference_digits, predicted_digits)
                sample_cers.append(sample_cer)
                per_speaker_cers[speaker_id].append(sample_cer)
                if speaker_id in seen_speakers:
                    in_domain_cers.append(sample_cer)
                else:
                    ood_cers.append(sample_cer)
                if len(saved_samples) < max_saved_samples:
                    saved_samples.append(
                        {
                            "epoch": "",
                            "speaker_id": speaker_id,
                            "audio_path": batch["audio_paths"][index],
                            "predicted_text": predicted_text,
                            "golden_text": reference_text,
                            "predicted_digits": predicted_digits,
                            "golden_digits": reference_digits,
                            "cer": f"{sample_cer:.6f}",
                        }
                    )
            continue

        if task_mode == "chunked":
            thousands_targets = batch["thousands_targets"].to(device)
            remainder_targets = batch["remainder_targets"].to(device)
            thousands_logits, remainder_logits = model(waveforms, waveform_lengths, apply_spec_augment=False)
            loss = ce_loss(thousands_logits, thousands_targets) + ce_loss(remainder_logits, remainder_targets)
            losses.append(loss.item())
            predicted_numbers = predict_chunked_numbers(thousands_logits, remainder_logits).cpu().tolist()

            for index, predicted_number in enumerate(predicted_numbers):
                predicted_digits = str(predicted_number)
                predicted_text = number_to_text(predicted_number)
                reference_digits = batch["reference_digits"][index]
                reference_text = batch["reference_words"][index]
                speaker_id = batch["speaker_ids"][index]
                sample_cer = jiwer.cer(reference_digits, predicted_digits)
                sample_cers.append(sample_cer)
                per_speaker_cers[speaker_id].append(sample_cer)
                if speaker_id in seen_speakers:
                    in_domain_cers.append(sample_cer)
                else:
                    ood_cers.append(sample_cer)
                if len(saved_samples) < max_saved_samples:
                    saved_samples.append(
                        {
                            "epoch": "",
                            "speaker_id": speaker_id,
                            "audio_path": batch["audio_paths"][index],
                            "predicted_text": predicted_text,
                            "golden_text": reference_text,
                            "predicted_digits": predicted_digits,
                            "golden_digits": reference_digits,
                            "cer": f"{sample_cer:.6f}",
                        }
                    )
            continue

        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        logits, logit_lengths = model(waveforms, waveform_lengths, apply_spec_augment=False)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        loss = ctc_loss(log_probs, targets, logit_lengths, target_lengths)
        losses.append(loss.item())

        for index, utterance_length in enumerate(logit_lengths.tolist()):
            emission = logits[index, :utterance_length].cpu()
            predicted_words = decode_utterance(decoder, emission, decode_mode)
            parsed_number = safe_parse_number_words(predicted_words)
            predicted_digits = "" if parsed_number is None else str(parsed_number)
            if parsed_number is None:
                parse_failures += 1

            reference_digits = batch["reference_digits"][index]
            reference_text = batch["reference_words"][index]
            speaker_id = batch["speaker_ids"][index]
            sample_cer = jiwer.cer(reference_digits, predicted_digits)
            sample_cers.append(sample_cer)
            per_speaker_cers[speaker_id].append(sample_cer)
            if len(saved_samples) < max_saved_samples:
                saved_samples.append(
                    {
                        "epoch": "",
                        "speaker_id": speaker_id,
                        "audio_path": batch["audio_paths"][index],
                        "predicted_text": predicted_words,
                        "golden_text": reference_text,
                        "predicted_digits": predicted_digits,
                        "golden_digits": reference_digits,
                        "cer": f"{sample_cer:.6f}",
                    }
                )

            if speaker_id in seen_speakers:
                in_domain_cers.append(sample_cer)
            else:
                ood_cers.append(sample_cer)

    overall_cer = mean_or_nan(sample_cers)
    in_domain_cer = mean_or_nan(in_domain_cers)
    ood_cer = mean_or_nan(ood_cers)
    harmonic_cer = overall_cer
    if in_domain_cers and ood_cers and (in_domain_cer + ood_cer) > 0:
        harmonic_cer = 2.0 * in_domain_cer * ood_cer / (in_domain_cer + ood_cer)

    return {
        "loss": mean_or_nan(losses),
        "cer": overall_cer,
        "ind_cer": in_domain_cer,
        "ood_cer": ood_cer,
        "harmonic_cer": harmonic_cer,
        "parse_fail_rate": parse_failures / max(1, len(sample_cers)),
        "speaker_cer": {
            speaker: sum(values) / len(values)
            for speaker, values in sorted(per_speaker_cers.items())
        },
        "saved_samples": saved_samples,
    }


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "train_args.json", vars(args) | {"device_resolved": str(device)})

    vocabulary: CTCVocabulary | None = None
    if args.task_mode == "ctc":
        if args.vocab.exists():
            vocabulary = CTCVocabulary.load(args.vocab)
            if vocabulary.token_type != "char":
                vocabulary = CTCVocabulary.from_tokens(all_number_chars(), token_type="char")
                vocabulary.save(args.vocab)
        else:
            vocabulary = CTCVocabulary.from_tokens(all_number_chars(), token_type="char")
            vocabulary.save(args.vocab)
    elif args.decode_mode != "greedy":
        print(f"Note: decode mode '{args.decode_mode}' is ignored for task_mode='{args.task_mode}'.")

    waveform_augment = None if args.disable_waveform_augment else WaveformAugment()

    train_dataset = SpokenNumbersDataset(
        manifest_path=args.train_manifest,
        vocabulary=vocabulary,
        sample_rate=16000,
        training=True,
        waveform_augment=waveform_augment,
        max_samples=args.train_max_samples,
    )
    dev_dataset = SpokenNumbersDataset(
        manifest_path=args.dev_manifest,
        vocabulary=vocabulary,
        sample_rate=16000,
        training=False,
        waveform_augment=None,
        max_samples=args.dev_max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        task_mode=args.task_mode,
        vocab_size=None if vocabulary is None else vocabulary.vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ctc_loss = None if args.task_mode != "ctc" else torch.nn.CTCLoss(blank=vocabulary.blank_id, zero_infinity=True)
    ce_loss = None if args.task_mode not in {"factorized", "chunked"} else torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    decoder = None
    if args.task_mode == "ctc":
        decoder = CTCDecoder(
            vocabulary=vocabulary,
            beam_width=args.beam_width,
            beam_size_token=args.beam_size_token,
            use_constraints=args.decode_mode == "beam_constrained",
            alpha=args.alpha,
            beta=args.beta,
            lm_path=str(args.lm_path) if args.lm_path else None,
        )

    train_rows = read_manifest(args.train_manifest)
    seen_speakers = {row["spk_id"] for row in train_rows}

    parameter_count = count_parameters(model)
    print(f"Device: {device}")
    print(f"Task mode: {args.task_mode}")
    print(f"Train samples: {len(train_dataset)} | Dev samples: {len(dev_dataset)}")
    if vocabulary is not None:
        print(f"Vocabulary size: {vocabulary.vocab_size}")
    print(f"Train speaker IDs: {sorted(seen_speakers)}")
    print(f"Model parameters: {parameter_count}")

    best_metric = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_losses: list[float] = []
        epoch_start = time.time()

        for batch in tqdm(train_loader):
            waveforms = batch["waveforms"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.task_mode == "factorized":
                thousands_hundreds_targets = batch["thousands_hundreds_targets"].to(device)
                thousands_last2_targets = batch["thousands_last2_targets"].to(device)
                remainder_hundreds_targets = batch["remainder_hundreds_targets"].to(device)
                remainder_last2_targets = batch["remainder_last2_targets"].to(device)
                (
                    thousands_hundreds_logits,
                    thousands_last2_logits,
                    remainder_hundreds_logits,
                    remainder_last2_logits,
                ) = model(waveforms, waveform_lengths, apply_spec_augment=True)
                loss = (
                    ce_loss(thousands_hundreds_logits, thousands_hundreds_targets)
                    + ce_loss(thousands_last2_logits, thousands_last2_targets)
                    + ce_loss(remainder_hundreds_logits, remainder_hundreds_targets)
                    + ce_loss(remainder_last2_logits, remainder_last2_targets)
                )
            elif args.task_mode == "chunked":
                thousands_targets = batch["thousands_targets"].to(device)
                remainder_targets = batch["remainder_targets"].to(device)
                thousands_logits, remainder_logits = model(waveforms, waveform_lengths, apply_spec_augment=True)
                loss = ce_loss(thousands_logits, thousands_targets) + ce_loss(remainder_logits, remainder_targets)
            else:
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                logits, logit_lengths = model(waveforms, waveform_lengths, apply_spec_augment=True)
                log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
                loss = ctc_loss(log_probs, targets, logit_lengths, target_lengths)

            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            running_losses.append(loss.item())

        scheduler.step()
        val_metrics = evaluate(
            model=model,
            loader=dev_loader,
            task_mode=args.task_mode,
            ctc_loss=ctc_loss,
            ce_loss=ce_loss,
            decoder=decoder,
            decode_mode=args.decode_mode,
            device=device,
            seen_speakers=seen_speakers,
            max_saved_samples=args.num_val_samples,
        )

        for sample in val_metrics["saved_samples"]:
            sample["epoch"] = str(epoch)
        save_validation_samples(args.val_samples_path, val_metrics["saved_samples"])

        train_loss = sum(running_losses) / max(1, len(running_losses))
        metric_to_track = val_metrics["harmonic_cer"]
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_cer={val_metrics['cer']:.4f} "
            f"ind_cer={val_metrics['ind_cer']:.4f} "
            f"ood_cer={val_metrics['ood_cer']:.4f} "
            f"harmonic={val_metrics['harmonic_cer']:.4f} "
            f"parse_fail={val_metrics['parse_fail_rate']:.2%} "
            f"time={epoch_time:.1f}s"
        )
        print(f"Speaker CER: {json.dumps(val_metrics['speaker_cer'], ensure_ascii=False)}")

        if metric_to_track < best_metric:
            best_metric = metric_to_track
            best_epoch = epoch
            epochs_without_improvement = 0
            save_validation_samples(args.best_val_samples_path, val_metrics["saved_samples"])
            save_checkpoint(
                checkpoint_path=args.output_dir / "best.pt",
                model=model,
                vocabulary=vocabulary,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    **val_metrics,
                    "parameter_count": parameter_count,
                    "decode_mode": args.decode_mode,
                },
                task_mode=args.task_mode,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    print(f"Best harmonic CER: {best_metric:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()

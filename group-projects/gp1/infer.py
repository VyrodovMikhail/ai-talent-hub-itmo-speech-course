from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import jiwer
import torch
from torch.utils.data import DataLoader

from .data_utils import SpokenNumbersDataset, ctc_collate_fn
from .decoding import CTCDecoder
from .numbers import safe_parse_number_words
from .train import decode_utterance, mean_or_nan, predict_chunked_numbers, predict_factorized_numbers
from .utils import load_checkpoint_bundle


def build_argparser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run inference on a manifest using a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=project_root / "artifacts" / "predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    bundle, model, vocabulary, _, _ = load_checkpoint_bundle(args.checkpoint, device=device)
    task_mode = bundle.get("task_mode", "ctc")
    model.eval()

    decoder = None
    if task_mode == "ctc":
        decoder = CTCDecoder(
            vocabulary=vocabulary,
            beam_width=args.beam_width,
            beam_size_token=args.beam_size_token,
            use_constraints=args.decode_mode == "beam_constrained",
            alpha=args.alpha,
            beta=args.beta,
            lm_path=str(args.lm_path) if args.lm_path else None,
        )
    elif args.decode_mode != "greedy":
        print(f"Note: decode mode '{args.decode_mode}' is ignored for task_mode='{task_mode}'.")

    dataset = SpokenNumbersDataset(
        manifest_path=args.manifest,
        vocabulary=vocabulary,
        sample_rate=16000,
        training=False,
        waveform_augment=None,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=device.type == "cuda",
    )

    predictions: list[dict[str, str]] = []
    sample_cers: list[float] = []
    per_speaker_cers: dict[str, list[float]] = defaultdict(list)
    parse_failures = 0

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveforms"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)

            if task_mode == "factorized":
                (
                    thousands_hundreds_logits,
                    thousands_last2_logits,
                    remainder_hundreds_logits,
                    remainder_last2_logits,
                ) = model(waveforms, waveform_lengths, apply_spec_augment=False)
                predicted_numbers = predict_factorized_numbers(
                    thousands_hundreds_logits,
                    thousands_last2_logits,
                    remainder_hundreds_logits,
                    remainder_last2_logits,
                ).cpu().tolist()

                for index, predicted_number in enumerate(predicted_numbers):
                    predicted_digits = str(predicted_number)
                    reference_digits = batch["reference_digits"][index]
                    speaker_id = batch["speaker_ids"][index]
                    if reference_digits:
                        sample_cer = jiwer.cer(reference_digits, predicted_digits)
                        sample_cers.append(sample_cer)
                        per_speaker_cers[speaker_id].append(sample_cer)

                    predictions.append(
                        {
                            "filename": Path(batch["audio_paths"][index]).name,
                            "prediction": predicted_digits,
                            "predicted_words": "",
                            "reference": reference_digits,
                            "speaker_id": speaker_id,
                        }
                    )
                continue

            if task_mode == "chunked":
                thousands_logits, remainder_logits = model(waveforms, waveform_lengths, apply_spec_augment=False)
                predicted_numbers = predict_chunked_numbers(thousands_logits, remainder_logits).cpu().tolist()

                for index, predicted_number in enumerate(predicted_numbers):
                    predicted_digits = str(predicted_number)
                    reference_digits = batch["reference_digits"][index]
                    speaker_id = batch["speaker_ids"][index]
                    if reference_digits:
                        sample_cer = jiwer.cer(reference_digits, predicted_digits)
                        sample_cers.append(sample_cer)
                        per_speaker_cers[speaker_id].append(sample_cer)

                    predictions.append(
                        {
                            "filename": Path(batch["audio_paths"][index]).name,
                            "prediction": predicted_digits,
                            "predicted_words": "",
                            "reference": reference_digits,
                            "speaker_id": speaker_id,
                        }
                    )
                continue

            logits, logit_lengths = model(waveforms, waveform_lengths, apply_spec_augment=False)
            for index, utterance_length in enumerate(logit_lengths.tolist()):
                emission = logits[index, :utterance_length].cpu()
                predicted_words = decode_utterance(decoder, emission, args.decode_mode)
                parsed_number = safe_parse_number_words(predicted_words)
                predicted_digits = "" if parsed_number is None else str(parsed_number)
                if parsed_number is None:
                    parse_failures += 1

                reference_digits = batch["reference_digits"][index]
                speaker_id = batch["speaker_ids"][index]
                if reference_digits:
                    sample_cer = jiwer.cer(reference_digits, predicted_digits)
                    sample_cers.append(sample_cer)
                    per_speaker_cers[speaker_id].append(sample_cer)

                predictions.append(
                    {
                        "filename": Path(batch["audio_paths"][index]).name,
                        "prediction": predicted_digits,
                        "predicted_words": predicted_words,
                        "reference": reference_digits,
                        "speaker_id": speaker_id,
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "prediction", "predicted_words", "reference", "speaker_id"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Wrote predictions to {args.output_csv}")
    if sample_cers:
        metrics = {
            "cer": mean_or_nan(sample_cers),
            "parse_fail_rate": parse_failures / len(sample_cers),
            "speaker_cer": {speaker: sum(values) / len(values) for speaker, values in sorted(per_speaker_cers.items())},
            "checkpoint_metrics": bundle.get("metrics", {}),
            "task_mode": task_mode,
        }
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

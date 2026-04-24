from __future__ import annotations

import argparse
import csv
import io
import tarfile
from collections import Counter
from pathlib import Path

from .data_utils import CTCVocabulary
from .numbers import all_number_chars, number_to_text


def _read_rows(archive_path: Path, metadata_name: str) -> list[dict[str, str]]:
    with tarfile.open(archive_path) as archive:
        metadata = archive.extractfile(metadata_name)
        if metadata is None:
            raise FileNotFoundError(f"{metadata_name} was not found in {archive_path}")
        text_stream = io.TextIOWrapper(metadata, encoding="utf-8")
        return list(csv.DictReader(text_stream))


def _extract_split_audio(
    archive_path: Path,
    extract_root: Path,
    rows: list[dict[str, str]],
    overwrite: bool,
) -> None:
    wanted_files = {row["filename"] for row in rows}
    with tarfile.open(archive_path) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name in wanted_files
        ]
        for member in members:
            target_path = extract_root / member.name
            if target_path.exists() and not overwrite:
                continue
            archive.extract(member, path=extract_root)


def _write_manifest(
    manifest_path: Path,
    extract_root: Path,
    rows: list[dict[str, str]],
    split_name: str,
) -> list[str]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    spoken_transcriptions: list[str] = []

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "filename",
                "audio_path",
                "utterance_id",
                "transcription",
                "spoken_transcription",
                "spk_id",
                "gender",
                "ext",
                "source_samplerate",
            ],
        )
        writer.writeheader()
        for row in rows:
            transcription = str(row["transcription"]).strip()
            spoken = number_to_text(int(transcription))
            spoken_transcriptions.append(spoken)
            audio_path = (extract_root / row["filename"]).resolve()
            writer.writerow(
                {
                    "split": split_name,
                    "filename": row["filename"],
                    "audio_path": str(audio_path),
                    "utterance_id": Path(row["filename"]).stem,
                    "transcription": transcription,
                    "spoken_transcription": spoken,
                    "spk_id": row["spk_id"],
                    "gender": row["gender"],
                    "ext": row["ext"],
                    "source_samplerate": row["samplerate"],
                }
            )
    return spoken_transcriptions


def prepare_split(
    archive_path: Path,
    split_name: str,
    output_root: Path,
    max_samples: int | None,
    overwrite: bool,
) -> tuple[Path, list[str], list[dict[str, str]]]:
    rows = _read_rows(archive_path, f"{split_name}.csv")
    if max_samples is not None:
        rows = rows[:max_samples]

    extract_root = output_root / "extracted"
    manifest_path = output_root / "manifests" / f"{split_name}_manifest.csv"

    _extract_split_audio(archive_path, extract_root, rows, overwrite=overwrite)
    spoken_transcriptions = _write_manifest(manifest_path, extract_root, rows, split_name)
    return manifest_path, spoken_transcriptions, rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract train/dev archives and build ASR manifests.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-archive", type=Path, default=None)
    parser.add_argument("--dev-archive", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or (project_root / "data")).resolve()
    train_archive = (args.train_archive or (project_root / "train.tar")).resolve()
    dev_archive = (args.dev_archive or (project_root / "dev.tar")).resolve()

    train_manifest, train_texts, train_rows = prepare_split(
        archive_path=train_archive,
        split_name="train",
        output_root=output_root,
        max_samples=args.max_train_samples,
        overwrite=args.overwrite,
    )
    dev_manifest, dev_texts, dev_rows = prepare_split(
        archive_path=dev_archive,
        split_name="dev",
        output_root=output_root,
        max_samples=args.max_dev_samples,
        overwrite=args.overwrite,
    )

    vocab = CTCVocabulary.from_tokens(all_number_chars(), token_type="char")
    vocab_path = output_root / "vocab.json"
    vocab.save(vocab_path)

    corpus_path = output_root / "lm_corpus.txt"
    corpus_path.write_text("\n".join(train_texts) + "\n", encoding="utf-8")

    tokens_path = output_root / "tokens.txt"
    tokens_path.write_text("\n".join(sorted(token for token in vocab.token_to_id if token != vocab.blank_token)) + "\n", encoding="utf-8")

    lexicon_path = output_root / "lexicon.txt"
    lexicon_lines = [
        f"{token} {token}"
        for token in sorted(token for token in vocab.token_to_id if token != vocab.blank_token)
    ]
    lexicon_path.write_text("\n".join(lexicon_lines) + "\n", encoding="utf-8")

    train_speakers = sorted({row["spk_id"] for row in train_rows})
    dev_speakers = sorted({row["spk_id"] for row in dev_rows})
    train_samplerates = Counter(row["samplerate"] for row in train_rows)
    dev_samplerates = Counter(row["samplerate"] for row in dev_rows)

    print(f"Prepared train manifest: {train_manifest}")
    print(f"Prepared dev manifest: {dev_manifest}")
    print(f"Saved vocabulary: {vocab_path} ({vocab.vocab_size} tokens including blank)")
    print(f"Saved LM corpus: {corpus_path}")
    print(f"Train speakers: {train_speakers}")
    print(f"Dev speakers: {dev_speakers}")
    print(f"Train samplerates: {dict(train_samplerates)}")
    print(f"Dev samplerates: {dict(dev_samplerates)}")
    print(f"Unique spoken tokens: {sorted(token for token in vocab.token_to_id if token != vocab.blank_token)}")


if __name__ == "__main__":
    main()

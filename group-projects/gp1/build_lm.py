from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .data_utils import read_manifest


def build_argparser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare LM artifacts and optionally train a KenLM n-gram model.")
    parser.add_argument("--train-manifest", type=Path, default=project_root / "data" / "train" / "train.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "artifacts" / "lm")
    parser.add_argument("--order", type=int, default=3)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.train_manifest)
    corpus_lines = [row["spoken_transcription"] for row in rows if row.get("spoken_transcription")]
    vocabulary = sorted({token for line in corpus_lines for token in line.split()})

    corpus_path = args.output_dir / "corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")

    tokens_path = args.output_dir / "tokens.txt"
    tokens_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")

    lexicon_path = args.output_dir / "lexicon.txt"
    lexicon_path.write_text("\n".join(f"{token} {token}" for token in vocabulary) + "\n", encoding="utf-8")

    lmplz_path = shutil.which("lmplz")
    build_binary_path = shutil.which("build_binary")
    arpa_path = args.output_dir / f"{args.order}gram.arpa"
    binary_path = args.output_dir / f"{args.order}gram.bin"

    print(f"Wrote corpus to {corpus_path}")
    print(f"Wrote tokens to {tokens_path}")
    print(f"Wrote lexicon to {lexicon_path}")

    if lmplz_path:
        with corpus_path.open("r", encoding="utf-8") as corpus_handle, arpa_path.open("w", encoding="utf-8") as arpa_handle:
            subprocess.run(
                [lmplz_path, "-o", str(args.order), "--discount_fallback"],
                stdin=corpus_handle,
                stdout=arpa_handle,
                check=True,
            )
        print(f"Wrote ARPA LM to {arpa_path}")

        if build_binary_path:
            subprocess.run([build_binary_path, str(arpa_path), str(binary_path)], check=True)
            print(f"Wrote KenLM binary to {binary_path}")
    else:
        print("KenLM binaries were not found in PATH. Corpus and lexicon were prepared, but ARPA training was skipped.")


if __name__ == "__main__":
    main()

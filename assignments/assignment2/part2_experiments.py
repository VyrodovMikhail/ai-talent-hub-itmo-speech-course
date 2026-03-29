import argparse
import csv
import gzip
import itertools
import json
import shutil
import subprocess
import tempfile
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jiwer
import kenlm
import matplotlib.pyplot as plt
import pandas as pd
import torchaudio

from wav2vec2decoder import Wav2Vec2Decoder


ROOT = Path(__file__).resolve().parent
DEFAULT_ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
DEFAULT_BETAS = [0.0, 0.5, 1.0, 1.5]
DEFAULT_TEMPERATURES = [0.5, 1.0, 1.5, 2.0]


@dataclass
class Sample:
    audio_path: Path
    reference: str


class DecoderPool:
    """Reuse a single acoustic model while swapping LM state/config."""

    def __init__(self, model_name: str):
        self.decoder = Wav2Vec2Decoder(model_name=model_name, lm_model_path=None)
        self.lm_cache: dict[str, kenlm.Model] = {}
        print(f"DecoderPool is initialized with the {model_name} model.")

    def configure(
        self,
        *,
        lm_model_path: str | None,
        beam_width: int,
        alpha: float,
        beta: float,
        temperature: float,
    ) -> Wav2Vec2Decoder:
        self.decoder.beam_width = beam_width
        self.decoder.alpha = alpha
        self.decoder.beta = beta
        self.decoder.temperature = temperature

        if lm_model_path:
            resolved = str((ROOT / lm_model_path).resolve()) if not Path(lm_model_path).is_absolute() else lm_model_path
            if resolved not in self.lm_cache:
                self.lm_cache[resolved] = kenlm.Model(resolved)
            self.decoder.lm_model = self.lm_cache[resolved]
        else:
            self.decoder.lm_model = None

        return self.decoder


def parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def load_manifest(manifest_path: str, limit: int | None = None) -> list[Sample]:
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = ROOT / manifest

    samples: list[Sample] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                Sample(
                    audio_path=ROOT / row["path"],
                    reference=row["text"].strip().lower(),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def ensure_parent(path: str | None) -> None:
    if path:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def evaluate_method(
    pool: DecoderPool,
    samples: Iterable[Sample],
    *,
    method: str,
    lm_model_path: str | None,
    beam_width: int,
    alpha: float,
    beta: float,
    temperature: float,
    store_predictions: bool = False,
):
    decoder = pool.configure(
        lm_model_path=lm_model_path,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
    )

    references: list[str] = []
    hypotheses: list[str] = []
    predictions: list[dict[str, str]] = []

    for sample in tqdm(samples):
        audio_input, sample_rate = torchaudio.load(sample.audio_path)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sample_rate} for {sample.audio_path}")

        hypothesis = decoder.decode(audio_input, method=method)
        references.append(sample.reference)
        hypotheses.append(hypothesis)

        if store_predictions:
            predictions.append(
                {
                    "path": str(sample.audio_path.relative_to(ROOT)),
                    "reference": sample.reference,
                    "hypothesis": hypothesis,
                }
            )

    result = {
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
    }
    if store_predictions:
        result["predictions"] = predictions
    return result


def save_dataframe(df: pd.DataFrame, output_csv: str | None) -> None:
    if output_csv:
        ensure_parent(output_csv)
        df.to_csv(output_csv, index=False)


def save_heatmap(df: pd.DataFrame, value_column: str, output_path: str, title: str) -> None:
    ensure_parent(output_path)
    pivot = df.pivot(index="beta", columns="alpha", values=value_column).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_column.upper())

    for row_idx, beta in enumerate(pivot.index):
        for col_idx, alpha in enumerate(pivot.columns):
            ax.text(col_idx, row_idx, f"{pivot.loc[beta, alpha]:.3f}", ha="center", va="center", color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_line_plot(df: pd.DataFrame, output_path: str, title: str) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, group in df.groupby("method"):
        group = group.sort_values("temperature")
        ax.plot(group["temperature"], group["wer"], marker="o", label=method)
    ax.set_xlabel("temperature")
    ax.set_ylabel("WER")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def qualitative_rows(baseline_predictions: list[dict[str, str]], candidate_predictions: list[dict[str, str]], limit: int) -> list[dict[str, str | float]]:
    rows = []
    for baseline, candidate in zip(baseline_predictions, candidate_predictions):
        if baseline["path"] != candidate["path"]:
            raise ValueError("Prediction alignment mismatch while building qualitative report")
        if baseline["hypothesis"] == candidate["hypothesis"]:
            continue

        baseline_wer = jiwer.wer(baseline["reference"], baseline["hypothesis"])
        candidate_wer = jiwer.wer(candidate["reference"], candidate["hypothesis"])

        if candidate_wer < baseline_wer:
            status = "improved"
        elif candidate_wer > baseline_wer:
            status = "worse"
        else:
            status = "changed_same_wer"

        rows.append(
            {
                "path": baseline["path"],
                "reference": baseline["reference"],
                "beam": baseline["hypothesis"],
                "candidate": candidate["hypothesis"],
                "baseline_wer": baseline_wer,
                "candidate_wer": candidate_wer,
                "status": status,
            }
        )

    priority = {"improved": 0, "changed_same_wer": 1, "worse": 2}
    rows.sort(key=lambda row: (priority[row["status"]], row["candidate_wer"] - row["baseline_wer"]))
    return rows[:limit]


def cmd_grid_search(args: argparse.Namespace) -> None:
    samples = load_manifest(args.manifest, args.limit)
    pool = DecoderPool(args.model_name)
    alphas = parse_list(args.alphas, float)
    betas = parse_list(args.betas, float)

    rows = []
    for alpha, beta in itertools.product(alphas, betas):
        metrics = evaluate_method(
            pool,
            samples,
            method=args.method,
            lm_model_path=args.lm_model_path,
            beam_width=args.beam_width,
            alpha=alpha,
            beta=beta,
            temperature=args.temperature,
        )
        rows.append(
            {
                "method": args.method,
                "manifest": args.manifest,
                "lm_model_path": args.lm_model_path,
                "beam_width": args.beam_width,
                "temperature": args.temperature,
                "alpha": alpha,
                "beta": beta,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["wer", "cer", "alpha", "beta"]).reset_index(drop=True)
    save_dataframe(df, args.output_csv)

    if args.heatmap_path:
        save_heatmap(df, "wer", args.heatmap_path, f"{args.method} WER grid")

    if args.qualitative_csv:
        best = df.iloc[0]
        baseline = evaluate_method(
            pool,
            samples,
            method="beam",
            lm_model_path=None,
            beam_width=args.beam_width,
            alpha=0.0,
            beta=0.0,
            temperature=args.temperature,
            store_predictions=True,
        )
        candidate = evaluate_method(
            pool,
            samples,
            method=args.method,
            lm_model_path=args.lm_model_path,
            beam_width=args.beam_width,
            alpha=float(best["alpha"]),
            beta=float(best["beta"]),
            temperature=args.temperature,
            store_predictions=True,
        )
        qualitative = pd.DataFrame(
            qualitative_rows(
                baseline["predictions"],
                candidate["predictions"],
                args.qualitative_count,
            )
        )
        save_dataframe(qualitative, args.qualitative_csv)

    print(df.to_string(index=False))


def cmd_temperature_sweep(args: argparse.Namespace) -> None:
    samples = load_manifest(args.manifest, args.limit)
    pool = DecoderPool(args.model_name)
    temperatures = parse_list(args.temperatures, float)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    rows = []
    for method, temperature in itertools.product(methods, temperatures):
        lm_model_path = args.lm_model_path if method != "greedy" else None
        metrics = evaluate_method(
            pool,
            samples,
            method=method,
            lm_model_path=lm_model_path,
            beam_width=args.beam_width,
            alpha=args.alpha,
            beta=args.beta,
            temperature=temperature,
        )
        rows.append(
            {
                "manifest": args.manifest,
                "method": method,
                "temperature": temperature,
                "alpha": args.alpha if method != "greedy" else 0.0,
                "beta": args.beta if method != "greedy" else 0.0,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["method", "temperature"]).reset_index(drop=True)
    save_dataframe(df, args.output_csv)

    if args.plot_path:
        save_line_plot(df, args.plot_path, f"Temperature sweep on {Path(args.manifest).stem}")

    print(df.to_string(index=False))


def parse_manifest_spec(raw: str) -> tuple[str, str]:
    label, path = raw.split("=", maxsplit=1)
    return label, path


def parse_config_spec(raw: str) -> dict[str, str | float | int | None]:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) != 7:
        raise ValueError(
            "Config must look like "
            "'name|method|lm_path_or_none|alpha|beta|temperature|beam_width'"
        )
    name, method, lm_path, alpha, beta, temperature, beam_width = parts
    return {
        "name": name,
        "method": method,
        "lm_model_path": None if lm_path.lower() == "none" else lm_path,
        "alpha": float(alpha),
        "beta": float(beta),
        "temperature": float(temperature),
        "beam_width": int(beam_width),
    }


def cmd_summary_table(args: argparse.Namespace) -> None:
    manifests = [parse_manifest_spec(raw) for raw in args.manifests]
    configs = [parse_config_spec(raw) for raw in args.configs]
    pool = DecoderPool(args.model_name)

    rows = []
    for config in configs:
        row = {"name": config["name"], "method": config["method"], "lm_model_path": config["lm_model_path"]}
        for label, manifest_path in manifests:
            samples = load_manifest(manifest_path, args.limit)
            print("Manifest was successfully initialized")
            metrics = evaluate_method(
                pool,
                samples,
                method=str(config["method"]),
                lm_model_path=config["lm_model_path"],
                beam_width=int(config["beam_width"]),
                alpha=float(config["alpha"]),
                beta=float(config["beta"]),
                temperature=float(config["temperature"]),
            )
            row[f"{label}_wer"] = metrics["wer"]
            row[f"{label}_cer"] = metrics["cer"]
        rows.append(row)

    df = pd.DataFrame(rows)
    save_dataframe(df, args.output_csv)
    print(df.to_string(index=False))


def cmd_train_lm(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lmplz_path = Path(args.lmplz)
    if not lmplz_path.is_absolute():
        lmplz_path = ROOT / lmplz_path

    with tempfile.NamedTemporaryFile(suffix=".arpa", delete=False) as tmp_file:
        tmp_arpa = Path(tmp_file.name)

    try:
        with corpus_path.open("rb") as stdin_handle, tmp_arpa.open("wb") as stdout_handle:
            subprocess.run(
                [str(lmplz_path), "-o", str(args.order), "--discount_fallback"],
                stdin=stdin_handle,
                stdout=stdout_handle,
                check=True,
            )

        with tmp_arpa.open("rb") as src, gzip.open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    finally:
        tmp_arpa.unlink(missing_ok=True)

    print(json.dumps({"output": str(output_path), "order": args.order}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assignment 2 Part 2 experiment helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid = subparsers.add_parser("grid-search", help="Run alpha/beta sweep for Task 4 or Task 6")
    grid.add_argument("--manifest", required=True)
    grid.add_argument("--method", choices=["beam_lm", "beam_lm_rescore"], required=True)
    grid.add_argument("--lm-model-path", required=True)
    grid.add_argument("--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS))
    grid.add_argument("--betas", default=",".join(str(value) for value in DEFAULT_BETAS))
    grid.add_argument("--beam-width", type=int, default=10)
    grid.add_argument("--temperature", type=float, default=1.0)
    grid.add_argument("--limit", type=int)
    grid.add_argument("--model-name", default="facebook/wav2vec2-base-100h")
    grid.add_argument("--output-csv")
    grid.add_argument("--heatmap-path")
    grid.add_argument("--qualitative-csv")
    grid.add_argument("--qualitative-count", type=int, default=10)
    grid.set_defaults(func=cmd_grid_search)

    temperature = subparsers.add_parser("temperature-sweep", help="Run Task 3 or Task 7b temperature sweep")
    temperature.add_argument("--manifest", required=True)
    temperature.add_argument("--methods", default="greedy,beam_lm")
    temperature.add_argument("--temperatures", default=",".join(str(value) for value in DEFAULT_TEMPERATURES))
    temperature.add_argument("--lm-model-path")
    temperature.add_argument("--alpha", type=float, default=0.1)
    temperature.add_argument("--beta", type=float, default=0.0)
    temperature.add_argument("--beam-width", type=int, default=10)
    temperature.add_argument("--limit", type=int)
    temperature.add_argument("--model-name", default="facebook/wav2vec2-base-100h")
    temperature.add_argument("--output-csv")
    temperature.add_argument("--plot-path")
    temperature.set_defaults(func=cmd_temperature_sweep)

    summary = subparsers.add_parser("summary-table", help="Build Task 7 or Task 9 comparison tables")
    summary.add_argument("--manifests", nargs="+", required=True, help="label=manifest_path")
    summary.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="name|method|lm_path_or_none|alpha|beta|temperature|beam_width",
    )
    summary.add_argument("--limit", type=int)
    summary.add_argument("--model-name", default="facebook/wav2vec2-base-100h")
    summary.add_argument("--output-csv")
    summary.set_defaults(func=cmd_summary_table)

    train = subparsers.add_parser("train-lm", help="Train Task 8 financial KenLM model")
    train.add_argument("--corpus", default="data/earnings22_train/corpus.txt")
    train.add_argument("--output", default="lm/financial-3gram.arpa.gz")
    train.add_argument("--lmplz", default="/tmp/kenlm_build/build/bin/lmplz")
    train.add_argument("--order", type=int, default=3)
    train.set_defaults(func=cmd_train_lm)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

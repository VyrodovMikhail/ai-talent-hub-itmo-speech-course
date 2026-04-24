from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .audio import WaveformAugment

from .numbers import number_to_text


@dataclass
class CTCVocabulary:
    token_to_id: dict[str, int]
    token_type: str = "word"

    blank_token: str = "<blank>"

    @classmethod
    def from_texts(cls, texts: list[str], token_type: str = "word") -> "CTCVocabulary":
        if token_type == "char":
            tokens = sorted({char for text in texts for char in text})
        else:
            tokens = sorted({token for text in texts for token in text.split()})
        return cls.from_tokens(tokens, token_type=token_type)

    @classmethod
    def from_tokens(cls, tokens: list[str], token_type: str = "word") -> "CTCVocabulary":
        mapping = {cls.blank_token: 0}
        for token in tokens:
            if token == cls.blank_token:
                continue
            mapping[token] = len(mapping)
        return cls(mapping, token_type=token_type)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {value: key for key, value in self.token_to_id.items()}

    @property
    def blank_id(self) -> int:
        return self.token_to_id[self.blank_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        if self.token_type == "char":
            return [self.token_to_id[token] for token in text]
        return [self.token_to_id[token] for token in text.split()]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> list[str]:
        id_to_token = self.id_to_token
        return [id_to_token[token_id] for token_id in token_ids if token_id != self.blank_id]

    def tokens_to_text(self, tokens: list[str] | tuple[str, ...]) -> str:
        if self.token_type == "char":
            return "".join(tokens).strip()
        return " ".join(tokens).strip()

    def ids_to_text(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return self.tokens_to_text(self.decode(token_ids))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "token_to_id": self.token_to_id,
            "token_type": self.token_type,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CTCVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "token_to_id" in payload:
            mapping = payload["token_to_id"]
            token_type = payload.get("token_type", "word")
        else:
            mapping = payload
            token_type = "word"
        return cls({str(key): int(value) for key, value in mapping.items()}, token_type=token_type)


def _infer_split_name(manifest_path: Path, row: dict[str, str]) -> str:
    if row.get("split"):
        return row["split"]
    if row.get("filename"):
        return Path(row["filename"]).parts[0]
    stem = manifest_path.stem.lower()
    if "train" in stem:
        return "train"
    if "dev" in stem:
        return "dev"
    if "test" in stem:
        return "test"
    return ""


def _resolve_audio_path(manifest_path: Path, row: dict[str, str]) -> str:
    audio_path = row.get("audio_path", "").strip()
    if audio_path:
        return str(Path(audio_path).expanduser().resolve())

    filename = row.get("filename", "").strip()
    if not filename:
        raise ValueError(f"Manifest row in {manifest_path} has neither audio_path nor filename")

    candidate_paths = [
        manifest_path.parent / filename,
        manifest_path.parent.parent / filename,
        manifest_path.parent.parent / "extracted" / filename,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate.resolve())

    return str((manifest_path.parent / filename).resolve())


def _canonicalize_manifest_row(manifest_path: Path, row: dict[str, str]) -> dict[str, str]:
    transcription = str(row.get("transcription", "")).strip()
    spoken_transcription = str(row.get("spoken_transcription", "")).strip()
    if not spoken_transcription and transcription.isdigit():
        spoken_transcription = number_to_text(int(transcription))

    filename = row.get("filename", "").strip()
    utterance_id = str(row.get("utterance_id", "")).strip() or Path(filename).stem

    return {
        "split": _infer_split_name(manifest_path, row),
        "filename": filename,
        "audio_path": _resolve_audio_path(manifest_path, row),
        "utterance_id": utterance_id,
        "transcription": transcription,
        "spoken_transcription": spoken_transcription,
        "spk_id": str(row.get("spk_id", "")).strip(),
        "gender": str(row.get("gender", "")).strip(),
        "ext": str(row.get("ext", "")).strip(),
        "source_samplerate": str(row.get("source_samplerate", row.get("samplerate", ""))).strip(),
    }


def read_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [_canonicalize_manifest_row(manifest_path, row) for row in rows]


class SpokenNumbersDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        vocabulary: CTCVocabulary | None,
        sample_rate: int,
        training: bool = False,
        waveform_augment: WaveformAugment | None = None,
        max_samples: int | None = None,
    ):
        rows = read_manifest(manifest_path)
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows
        self.vocabulary = vocabulary
        self.sample_rate = sample_rate
        self.training = training
        self.waveform_augment = waveform_augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from .audio import load_audio

        row = self.rows[index]
        waveform, _ = load_audio(row["audio_path"], sample_rate=self.sample_rate)
        if self.training and self.waveform_augment is not None:
            waveform = self.waveform_augment(waveform)

        spoken_transcription = row.get("spoken_transcription", "").strip()
        transcription = row.get("transcription", "").strip()
        number_value = int(transcription) if transcription else 0
        target_ids = torch.tensor(
            self.vocabulary.encode(spoken_transcription) if (self.vocabulary is not None and spoken_transcription) else [],
            dtype=torch.long,
        )
        return {
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform.numel(), dtype=torch.long),
            "target_ids": target_ids,
            "target_length": torch.tensor(target_ids.numel(), dtype=torch.long),
            "reference_digits": transcription,
            "reference_words": spoken_transcription,
            "speaker_id": row.get("spk_id", ""),
            "audio_path": row["audio_path"],
            "number_value": torch.tensor(number_value, dtype=torch.long),
            "thousands_target": torch.tensor(number_value // 1000, dtype=torch.long),
            "remainder_target": torch.tensor(number_value % 1000, dtype=torch.long),
            "thousands_hundreds_target": torch.tensor((number_value // 1000) // 100, dtype=torch.long),
            "thousands_last2_target": torch.tensor((number_value // 1000) % 100, dtype=torch.long),
            "remainder_hundreds_target": torch.tensor((number_value % 1000) // 100, dtype=torch.long),
            "remainder_last2_target": torch.tensor((number_value % 1000) % 100, dtype=torch.long),
        }


def ctc_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = [item["waveform"] for item in batch]
    waveform_lengths = torch.stack([item["waveform_length"] for item in batch])
    target_lengths = torch.stack([item["target_length"] for item in batch])
    targets = torch.cat([item["target_ids"] for item in batch], dim=0)

    padded_waveforms = pad_sequence(waveforms, batch_first=True)

    return {
        "waveforms": padded_waveforms,
        "waveform_lengths": waveform_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "number_values": torch.stack([item["number_value"] for item in batch]),
        "thousands_targets": torch.stack([item["thousands_target"] for item in batch]),
        "remainder_targets": torch.stack([item["remainder_target"] for item in batch]),
        "thousands_hundreds_targets": torch.stack([item["thousands_hundreds_target"] for item in batch]),
        "thousands_last2_targets": torch.stack([item["thousands_last2_target"] for item in batch]),
        "remainder_hundreds_targets": torch.stack([item["remainder_hundreds_target"] for item in batch]),
        "remainder_last2_targets": torch.stack([item["remainder_last2_target"] for item in batch]),
        "reference_digits": [item["reference_digits"] for item in batch],
        "reference_words": [item["reference_words"] for item in batch],
        "speaker_ids": [item["speaker_id"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
    }

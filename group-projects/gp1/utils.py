from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .audio import AudioConfig
from .data_utils import CTCVocabulary
from .model import BaseSpeechModel, ModelConfig, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: BaseSpeechModel,
    vocabulary: CTCVocabulary | None,
    epoch: int,
    metrics: dict,
    task_mode: str,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "audio_config": model.audio_config.to_dict(),
            "model_config": model.model_config.to_dict(),
            "token_to_id": None if vocabulary is None else vocabulary.token_to_id,
            "token_type": None if vocabulary is None else vocabulary.token_type,
            "task_mode": task_mode,
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_checkpoint_bundle(checkpoint_path: str | Path, device: torch.device):
    bundle = torch.load(checkpoint_path, map_location=device)
    token_to_id = bundle.get("token_to_id")
    token_type = bundle.get("token_type")
    vocabulary = None if token_to_id is None else CTCVocabulary(token_to_id, token_type=token_type or "word")
    audio_config = AudioConfig(**bundle["audio_config"])
    model_config = ModelConfig(**bundle["model_config"])
    task_mode = bundle.get("task_mode", "ctc")
    model = build_model(
        task_mode=task_mode,
        vocab_size=None if vocabulary is None else vocabulary.vocab_size,
        audio_config=audio_config,
        model_config=model_config,
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    return bundle, model, vocabulary, audio_config, model_config

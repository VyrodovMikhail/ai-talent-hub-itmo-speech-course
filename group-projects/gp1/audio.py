from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch import nn


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min_hz: float = 0.0
    f_max_hz: float | None = None
    log_offset: float = 1e-6
    normalize: str = "per_feature"

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(audio_path: str, sample_rate: int) -> tuple[torch.Tensor, int]:
    audio, source_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_sr != sample_rate:
        divisor = math.gcd(source_sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, source_sr // divisor).astype(np.float32)
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float()
    return waveform, sample_rate


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float,
) -> np.ndarray:
    mel_min = _hz_to_mel(np.array([f_min_hz], dtype=np.float32))[0]
    mel_max = _hz_to_mel(np.array([f_max_hz], dtype=np.float32))[0]
    mel_points = np.linspace(mel_min, mel_max, num=n_mels + 2, dtype=np.float32)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)
    filterbank = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)

    for index in range(1, n_mels + 1):
        left = bins[index - 1]
        center = max(bins[index], left + 1)
        right = max(bins[index + 1], center + 1)

        for freq_bin in range(left, center):
            filterbank[freq_bin, index - 1] = (freq_bin - left) / (center - left)
        for freq_bin in range(center, min(right, filterbank.shape[0])):
            filterbank[freq_bin, index - 1] = (right - freq_bin) / (right - center)

    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-6)
    filterbank *= enorm[None, :]
    return filterbank


class WaveformAugment:
    def __init__(
        self,
        noise_prob: float = 0.3,
        max_noise_level: float = 0.02,
        gain_db_range: float = 6.0,
        speed_prob: float = 0.3,
        speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1),
    ):
        self.noise_prob = noise_prob
        self.max_noise_level = max_noise_level
        self.gain_db_range = gain_db_range
        self.speed_prob = speed_prob
        self.speed_factors = tuple(speed_factors)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        augmented = waveform.clone()

        if random.random() < self.speed_prob:
            speed = random.choice(self.speed_factors)
            target_len = max(1, int(round(augmented.numel() / speed)))
            augmented = F.interpolate(
                augmented.view(1, 1, -1),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).view(-1)

        gain_db = random.uniform(-self.gain_db_range, self.gain_db_range)
        augmented = augmented * (10.0 ** (gain_db / 20.0))

        if random.random() < self.noise_prob:
            scale = max(augmented.abs().max().item(), 1e-3)
            noise_level = random.uniform(0.001, self.max_noise_level) * scale
            augmented = augmented + noise_level * torch.randn_like(augmented)

        return augmented.clamp_(-1.0, 1.0)


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_masks: int = 2,
        max_freq_width: int = 8,
        time_masks: int = 2,
        max_time_width: int = 20,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.max_freq_width = max_freq_width
        self.time_masks = time_masks
        self.max_time_width = max_time_width

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        augmented = features.clone()
        batch, feat_bins, frames = augmented.shape
        for idx in range(batch):
            for _ in range(self.freq_masks):
                width = random.randint(0, self.max_freq_width)
                if width == 0 or width >= feat_bins:
                    continue
                start = random.randint(0, feat_bins - width)
                augmented[idx, start : start + width] = 0

            for _ in range(self.time_masks):
                width = random.randint(0, self.max_time_width)
                if width == 0 or width >= frames:
                    continue
                start = random.randint(0, frames - width)
                augmented[idx, :, start : start + width] = 0
        return augmented


class LogMelFilterBanks(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.register_buffer("window", torch.hann_window(config.win_length), persistent=False)
        fbanks = _build_mel_filterbank(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            f_min_hz=config.f_min_hz,
            f_max_hz=config.f_max_hz or (config.sample_rate / 2),
        )
        self.register_buffer("mel_fbanks", torch.tensor(fbanks, dtype=torch.float32), persistent=False)

    def feature_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(waveform_lengths, self.config.hop_length, rounding_mode="floor") + 1

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            waveforms,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(waveforms.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power_spec = spec.abs().pow(2.0)
        mel = torch.matmul(power_spec.transpose(-2, -1), self.mel_fbanks.to(waveforms.device)).transpose(-2, -1)
        features = torch.log(mel.clamp_min(self.config.log_offset))
        lengths = self.feature_lengths(waveform_lengths)

        if self.config.normalize == "per_feature":
            features = self._normalize(features, lengths)
        return features, lengths

    @staticmethod
    def _normalize(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        normalized = features.clone()
        for idx, length in enumerate(lengths.tolist()):
            valid = normalized[idx, :, :length]
            mean = valid.mean(dim=1, keepdim=True)
            std = valid.std(dim=1, keepdim=True).clamp_min(1e-5)
            normalized[idx, :, :length] = (valid - mean) / std
        return normalized


def pad_to_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.numel() >= target_length:
        return waveform[:target_length]
    return F.pad(waveform, (0, target_length - waveform.numel()))

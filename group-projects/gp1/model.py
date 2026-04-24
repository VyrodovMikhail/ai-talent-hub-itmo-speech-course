from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .audio import AudioConfig, LogMelFilterBanks, SpecAugment


@dataclass
class ModelConfig:
    stem_channels: int = 256
    encoder_channels: tuple[int, ...] = (256, 256, 320, 320, 384, 384)
    encoder_kernels: tuple[int, ...] = (11, 13, 15, 17, 19, 21)
    repeats_per_block: int = 3
    dropout: float = 0.2
    final_channels: int = 512
    classifier_hidden: int = 384
    stem_stride: int = 2
    activation: str = "relu"
    spec_freq_masks: int = 2
    spec_time_masks: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def conv_output_lengths(lengths: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> torch.Tensor:
    return torch.div(
        lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation_name: str):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = _activation(activation_name)

    def forward(self, x: torch.Tensor, apply_activation: bool = True) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        if apply_activation:
            x = self.activation(x)
        return x


class QuartzBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        repeats: int,
        dropout: float,
        activation_name: str,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(repeats):
            self.layers.append(SeparableConv1d(current_channels, out_channels, kernel_size, activation_name))
            current_channels = out_channels

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.activation = _activation(activation_name)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = x
        for index, layer in enumerate(self.layers):
            out = layer(out, apply_activation=index < len(self.layers) - 1)
            if index < len(self.layers) - 1:
                out = self.dropout(out)
        out = self.activation(out + residual)
        out = self.dropout(out)
        return out


class QuartzNetEncoder(nn.Module):
    def __init__(self, input_features: int, config: ModelConfig):
        super().__init__()
        padding = 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_features,
                config.stem_channels,
                kernel_size=5,
                stride=config.stem_stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(config.stem_channels),
            _activation(config.activation),
            nn.Dropout(config.dropout),
        )

        blocks = []
        in_channels = config.stem_channels
        for out_channels, kernel_size in zip(config.encoder_channels, config.encoder_kernels):
            blocks.append(
                QuartzBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    repeats=config.repeats_per_block,
                    dropout=config.dropout,
                    activation_name=config.activation,
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, config.final_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(config.final_channels),
            _activation(config.activation),
            nn.Dropout(config.dropout),
        )
        self.stem_kernel = 5
        self.stem_padding = padding
        self.stem_stride = config.stem_stride
        self.output_channels = config.final_channels

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(features)
        output_lengths = conv_output_lengths(
            feature_lengths,
            kernel_size=self.stem_kernel,
            stride=self.stem_stride,
            padding=self.stem_padding,
        )
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x, output_lengths


class BaseSpeechModel(nn.Module):
    task_mode: str = "base"

    def __init__(self, audio_config: AudioConfig | None = None, model_config: ModelConfig | None = None):
        super().__init__()
        self.audio_config = audio_config or AudioConfig()
        self.model_config = model_config or ModelConfig()
        self.frontend = LogMelFilterBanks(self.audio_config)
        self.spec_augment = SpecAugment(
            freq_masks=self.model_config.spec_freq_masks,
            time_masks=self.model_config.spec_time_masks,
        )

    def extract_features(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        apply_spec_augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.frontend(waveforms, waveform_lengths)
        if apply_spec_augment:
            features = self.spec_augment(features)
        return features, feature_lengths


class SpeechRecognitionModel(BaseSpeechModel):
    task_mode = "ctc"

    def __init__(self, vocab_size: int, audio_config: AudioConfig | None = None, model_config: ModelConfig | None = None):
        super().__init__(audio_config=audio_config, model_config=model_config)
        self.encoder = QuartzNetEncoder(
            input_features=self.audio_config.n_mels,
            config=self.model_config,
        )
        self.head = nn.Conv1d(self.encoder.output_channels, vocab_size, kernel_size=1)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        apply_spec_augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.extract_features(
            waveforms,
            waveform_lengths,
            apply_spec_augment=apply_spec_augment,
        )
        encoded, output_lengths = self.encoder(features, feature_lengths)
        logits = self.head(encoded).transpose(1, 2)
        return logits, output_lengths


class MaskedStatisticsPooling(nn.Module):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = x.size(-1)
        indices = torch.arange(max_len, device=x.device).unsqueeze(0)
        mask = indices < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).to(dtype=x.dtype)

        masked = x * mask
        denom = lengths.clamp_min(1).to(dtype=x.dtype).view(-1, 1, 1)
        mean = masked.sum(dim=-1) / denom.squeeze(-1)

        variance = ((x - mean.unsqueeze(-1)) * mask).pow(2).sum(dim=-1) / denom.squeeze(-1)
        std = torch.sqrt(variance.clamp_min(1e-6))
        return torch.cat([mean, std], dim=1)


class ChunkedNumberModel(BaseSpeechModel):
    task_mode = "chunked"

    def __init__(self, audio_config: AudioConfig | None = None, model_config: ModelConfig | None = None):
        super().__init__(audio_config=audio_config, model_config=model_config)
        self.encoder = QuartzNetEncoder(
            input_features=self.audio_config.n_mels,
            config=self.model_config,
        )
        pooled_dim = self.encoder.output_channels * 2
        self.pooling = MaskedStatisticsPooling()
        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, self.model_config.classifier_hidden, bias=False),
            nn.BatchNorm1d(self.model_config.classifier_hidden),
            _activation(self.model_config.activation),
            nn.Dropout(self.model_config.dropout),
        )
        self.thousands_head = nn.Linear(self.model_config.classifier_hidden, 1000)
        self.remainder_head = nn.Linear(self.model_config.classifier_hidden, 1000)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        apply_spec_augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.extract_features(
            waveforms,
            waveform_lengths,
            apply_spec_augment=apply_spec_augment,
        )
        encoded, output_lengths = self.encoder(features, feature_lengths)
        pooled = self.pooling(encoded, output_lengths)
        shared = self.shared(pooled)
        thousands_logits = self.thousands_head(shared)
        remainder_logits = self.remainder_head(shared)
        return thousands_logits, remainder_logits


class FactorizedNumberModel(BaseSpeechModel):
    task_mode = "factorized"

    def __init__(self, audio_config: AudioConfig | None = None, model_config: ModelConfig | None = None):
        super().__init__(audio_config=audio_config, model_config=model_config)
        self.encoder = QuartzNetEncoder(
            input_features=self.audio_config.n_mels,
            config=self.model_config,
        )
        pooled_dim = self.encoder.output_channels * 2
        self.pooling = MaskedStatisticsPooling()
        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, self.model_config.classifier_hidden, bias=False),
            nn.BatchNorm1d(self.model_config.classifier_hidden),
            _activation(self.model_config.activation),
            nn.Dropout(self.model_config.dropout),
        )
        self.thousands_hundreds_head = nn.Linear(self.model_config.classifier_hidden, 10)
        self.thousands_last2_head = nn.Linear(self.model_config.classifier_hidden, 100)
        self.remainder_hundreds_head = nn.Linear(self.model_config.classifier_hidden, 10)
        self.remainder_last2_head = nn.Linear(self.model_config.classifier_hidden, 100)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        apply_spec_augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.extract_features(
            waveforms,
            waveform_lengths,
            apply_spec_augment=apply_spec_augment,
        )
        encoded, output_lengths = self.encoder(features, feature_lengths)
        pooled = self.pooling(encoded, output_lengths)
        shared = self.shared(pooled)
        return (
            self.thousands_hundreds_head(shared),
            self.thousands_last2_head(shared),
            self.remainder_hundreds_head(shared),
            self.remainder_last2_head(shared),
        )


def build_model(
    task_mode: str,
    vocab_size: int | None = None,
    audio_config: AudioConfig | None = None,
    model_config: ModelConfig | None = None,
) -> BaseSpeechModel:
    if task_mode == "ctc":
        if vocab_size is None:
            raise ValueError("vocab_size is required for task_mode='ctc'")
        return SpeechRecognitionModel(vocab_size=vocab_size, audio_config=audio_config, model_config=model_config)
    if task_mode == "chunked":
        return ChunkedNumberModel(audio_config=audio_config, model_config=model_config)
    if task_mode == "factorized":
        return FactorizedNumberModel(audio_config=audio_config, model_config=model_config)
    raise ValueError(f"Unsupported task mode: {task_mode}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

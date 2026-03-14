"""
Binary YES/NO speech classification using Conv1d CNN on features built with LogMelFilterBanks.

Usage:
    python train.py                           # single run with defaults
    python train.py --n_mels 40 --groups 1   # custom run
    python train.py --experiment n_mels       # sweep n_mels in [20,40,80]
    python train.py --experiment groups       # sweep groups in [1,2,4,8,16]
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import soundfile as sf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Accuracy
from thop import profile as thop_profile

from melbanks import LogMelFilterBanks


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

LABELS = ['yes', 'no']
TARGET_LEN = 16_000         # 1 second of audio at 16 kHz

DATASET_NAME = 'SpeechCommands/speech_commands_v0.02'


def _load_wav(path: Path) -> torch.Tensor:
    data, _ = sf.read(str(path), dtype='float32')
    return torch.from_numpy(data).unsqueeze(0)


class YesNoDataset(Dataset):
    def __init__(self, root: str, subset: str, download: bool = True):
        super().__init__()
        sc_root = Path(root)
        archive_dir = sc_root / DATASET_NAME
        if not archive_dir.exists():
            SPEECHCOMMANDS(root=str(sc_root), download=download)

        # Build the split sets from the official text files
        val_set = set(line.strip() for line in archive_dir / 'validation_list.txt' if line.strip())
        test_set = set(line.strip() for line in archive_dir / 'testing_list.txt' if line.strip())

        self._samples: List[tuple] = []
        for label in LABELS:
            for wav_path in sorted((archive_dir / label).glob('*.wav')):
                rel = f"{label}/{wav_path.name}"
                if subset == 'training'   and rel not in val_set and rel not in test_set:
                    self._samples.append((wav_path, LABELS.index(label)))
                elif subset == 'validation' and rel in val_set:
                    self._samples.append((wav_path, LABELS.index(label)))
                elif subset == 'testing'    and rel in test_set:
                    self._samples.append((wav_path, LABELS.index(label)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        wav_path, label_idx = self._samples[idx]
        waveform = _load_wav(wav_path)
        n = waveform.shape[-1]
        if n < TARGET_LEN:
            waveform = TF.pad(waveform, (0, TARGET_LEN - n))
        else:
            waveform = waveform[..., :TARGET_LEN]
        return waveform, label_idx


def get_loaders(data_root: str = 'data', batch_size: int = 64,
                num_workers: int = 0) -> tuple:
    train_ds = YesNoDataset(data_root, subset='training')
    val_ds   = YesNoDataset(data_root, subset='validation')
    test_ds  = YesNoDataset(data_root, subset='testing')

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SpeechCNN(nn.Module):
    def __init__(self, n_mels: int = 80, n_classes: int = 2, groups: int = 1):
        super().__init__()
        if n_mels % groups != 0:
            raise ValueError(f"n_mels={n_mels} must be divisible by groups={groups}")
        if 128 % groups != 0:
            raise ValueError(f"Hidden dims (64, 128) must be divisible by groups={groups}")

        self.features = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (batch, 128, 1)
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_mels, n_frames)
        out = self.features(x).squeeze(-1)   # (batch, 128)
        return self.classifier(out)          # (batch, n_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, n_mels: int = 80, n_frames: int = 101) -> int:
        """Compute FLOPs for one forward pass via thop."""
        dummy = torch.zeros(1, n_mels, n_frames)
        flops, _ = thop_profile(self, inputs=(dummy,), verbose=False)
        return int(flops)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class SpeechClassifier(pl.LightningModule):
    """
    Tracked metrics:
        train_loss     – cross-entropy loss (per epoch)
        val_acc        – accuracy on the validation set (per epoch)
        epoch_time_s   – wall-clock seconds for the entire training epoch
        test_acc       – accuracy on the test set
    """

    def __init__(
        self,
        n_mels: int = 80,
        groups: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = LogMelFilterBanks(n_mels=n_mels)
        self.model = SpeechCNN(n_mels=n_mels, groups=groups)
        self.criterion = nn.CrossEntropyLoss()

        self.val_acc  = Accuracy(task='multiclass', num_classes=2)
        self.test_acc = Accuracy(task='multiclass', num_classes=2)

        self._epoch_start: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)                   # Squeeze the mono channel so the feature extractor sees (B, T)
        features = self.feature_extractor(x)   # (B, n_mels, n_frames)
        return self.model(features)            # (B, n_classes)

    def on_train_epoch_start(self):
        self._epoch_start = time.perf_counter()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        elapsed = time.perf_counter() - self._epoch_start
        self.log('epoch_time_s', elapsed, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        self.test_acc(preds, y)
        self.log('test_acc', self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self) -> int:
        """FLOPs of the CNN head only (feature extraction excluded)."""
        return self.model.count_flops(
            n_mels=self.hparams.n_mels, n_frames=101
        )


def build_trainer(log_dir: str, max_epochs: int = 15) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        logger=CSVLogger(save_dir=log_dir, name=''),
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        log_every_n_steps=1,
    )


def train_and_eval(
    n_mels: int,
    groups: int,
    data_root: str,
    log_dir: str,
    max_epochs: int = 15,
    batch_size: int = 64,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  n_mels={n_mels}  groups={groups}  epochs={max_epochs}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = get_loaders(
        data_root=data_root, batch_size=batch_size
    )

    model = SpeechClassifier(n_mels=n_mels, groups=groups)
    n_params = model.count_parameters()
    flops    = model.count_flops()
    print(f"  Parameters : {n_params:,}")
    print(f"  FLOPs (CNN): {flops:,}")

    trainer = build_trainer(log_dir=log_dir, max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader, verbose=False)
    test_acc = test_results[0]['test_acc']

    # Read per-epoch metrics from the CSV log
    log_path = Path(log_dir) / 'version_0' / 'metrics.csv'
    train_loss_curve, val_acc_curve, epoch_times = [], [], []
    if log_path.exists():
        import csv
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('train_loss'):
                    train_loss_curve.append(float(row['train_loss']))
                if row.get('val_acc'):
                    val_acc_curve.append(float(row['val_acc']))
                if row.get('epoch_time_s'):
                    epoch_times.append(float(row['epoch_time_s']))

    return {
        'n_mels': n_mels,
        'groups': groups,
        'n_params': n_params,
        'flops': flops,
        'test_acc': float(test_acc),
        'train_loss_curve': train_loss_curve,
        'val_acc_curve': val_acc_curve,
        'epoch_times': epoch_times,
        'mean_epoch_time': float(sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_mel_comparison(save_path: str = 'mel_comparison.png'):
    """
    Plot LogMelFilterBanks vs torch log melspectogram for a random 16 kHz signal.
    """
    torch.manual_seed(0)
    signal = torch.randn(1, 16000)

    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160, n_mels=80
    )(signal)
    logmelbanks = LogMelFilterBanks()(signal)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(
        torch.log(melspec[0] + 1e-6).numpy(),
        aspect='auto', origin='lower', cmap='magma'
    )
    axes[0].set_title('log(torchaudio.MelSpectrogram + ε)')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mel bin')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        logmelbanks[0].numpy(),
        aspect='auto', origin='lower', cmap='magma'
    )
    axes[1].set_title('LogMelFilterBanks (our implementation)')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mel bin')
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle('Log Mel Filterbank comparison  |  max |diff| = '
                 f'{(torch.log(melspec+1e-6) - logmelbanks).abs().max():.2e}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_n_mels_experiment(results: List[dict], save_path: str = 'exp_n_mels.png'):
    """Plots for the n_mels sweep."""
    n_mels_vals = [r['n_mels'] for r in results]
    test_accs   = [r['test_acc'] * 100 for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for r in results:
        axes[0].plot(r['train_loss_curve'], label=f"n_mels={r['n_mels']}")
    axes[0].set_title('Train Loss vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar([str(v) for v in n_mels_vals], test_accs, color='steelblue')
    axes[1].set_title('Test Accuracy vs n_mels')
    axes[1].set_xlabel('n_mels')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(test_accs):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)

    for r in results:
        axes[2].plot(r['val_acc_curve'], label=f"n_mels={r['n_mels']}")
    axes[2].set_title('Validation Accuracy vs Epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Experiment 1: Effect of n_mels on YES/NO Classification')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_groups_experiment(results: List[dict], save_path: str = 'exp_groups.png'):
    """Plots for the groups sweep."""
    groups_vals   = [r['groups'] for r in results]
    n_params      = [r['n_params'] for r in results]
    flops         = [r['flops'] for r in results]
    epoch_times   = [r['mean_epoch_time'] for r in results]
    test_accs     = [r['test_acc'] * 100 for r in results]
    labels        = [str(g) for g in groups_vals]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(groups_vals, epoch_times, 'o-', color='tomato')
    ax.set_title('Mean Epoch Training Time vs groups')
    ax.set_xlabel('groups')
    ax.set_ylabel('Time (s)')
    ax.set_xticks(groups_vals)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(labels, n_params, color='steelblue')
    ax.set_title('Model Parameters vs groups')
    ax.set_xlabel('groups')
    ax.set_ylabel('# Parameters')
    for i, v in enumerate(n_params):
        ax.text(i, v + max(n_params)*0.01, f'{v:,}', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 2]
    ax.bar(labels, flops, color='seagreen')
    ax.set_title('FLOPs (CNN head) vs groups')
    ax.set_xlabel('groups')
    ax.set_ylabel('FLOPs')
    for i, v in enumerate(flops):
        ax.text(i, v + max(flops)*0.01, f'{v/1e6:.2f}M', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    ax.bar(labels, test_accs, color='gold')
    ax.set_title('Test Accuracy vs groups')
    ax.set_xlabel('groups')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    for i, v in enumerate(test_accs):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    for r in results:
        ax.plot(r['train_loss_curve'], label=f"groups={r['groups']}")
    ax.set_title('Train Loss vs Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for r in results:
        ax.plot(r['val_acc_curve'], label=f"groups={r['groups']}")
    ax.set_title('Val Accuracy vs Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 2: Effect of Conv1d groups on YES/NO Classification  (n_mels=80)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mels',     type=int, default=80)
    parser.add_argument('--groups',     type=int, default=1)
    parser.add_argument('--epochs',     type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_root',  type=str, default='data')
    parser.add_argument('--log_dir',    type=str, default='logs')
    parser.add_argument(
        '--experiment',
        choices=['n_mels', 'groups', 'both', 'mel_plot'],
        default=None,
        help='Run a sweep experiment instead of a single training run.'
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Mel-bank comparison plot
    if args.experiment == 'mel_plot':
        plot_mel_comparison('mel_comparison.png')
        return

    # Training
    if args.experiment is None:
        result = train_and_eval(
            n_mels=args.n_mels,
            groups=args.groups,
            data_root=args.data_root,
            log_dir=os.path.join(args.log_dir, f'nm{args.n_mels}_g{args.groups}'),
            max_epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(f"\nResult: test_acc={result['test_acc']:.4f} "
              f"params={result['n_params']:,} flops={result['flops']:,}")
        return

    results_n_mels, results_groups = [], []

    # Experiment 1: different n_mels count
    if args.experiment in ('n_mels', 'both'):
        for n_mels in [20, 40, 80]:
            r = train_and_eval(
                n_mels=n_mels, groups=1,
                data_root=args.data_root,
                log_dir=os.path.join(args.log_dir, f'nm{n_mels}_g1'),
                max_epochs=args.epochs,
                batch_size=args.batch_size,
            )
            results_n_mels.append(r)

        with open('results_n_mels.json', 'w') as f:
            json.dump(results_n_mels, f, indent=2)
        plot_n_mels_experiment(results_n_mels)

    # Experiment 2: different groups count
    if args.experiment in ('groups', 'both'):
        for groups in [1, 2, 4, 8, 16]:
            r = train_and_eval(
                n_mels=80, groups=groups,
                data_root=args.data_root,
                log_dir=os.path.join(args.log_dir, f'nm80_g{groups}'),
                max_epochs=args.epochs,
                batch_size=args.batch_size,
            )
            results_groups.append(r)

        with open('results_groups.json', 'w') as f:
            json.dump(results_groups, f, indent=2)
        plot_groups_experiment(results_groups)


if __name__ == '__main__':
    main()

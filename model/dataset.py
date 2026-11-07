"""Loading features + labels as tensors, and the class <-> (string, fret) map.

Shared by train.py, baseline.py and evaluate.py, so all three agree on
exactly how a note becomes model input and how a position becomes a class.

Classes
-------
The model answers one multiple-choice question: "which of the 120 neck
positions was played?" Each (string, fret) pair gets a class number:

    class = (string - 1) * n_frets + fret        n_frets = max_fret + 1 = 20

so string 1 (high E) frets 0-19 are classes 0-19, string 2 is 20-39, ...,
string 6 (low E) is 100-119.

Why no DataLoader?
------------------
Our whole training set is ~4000 small spectrograms (~180 MB as float32), so
we put it on the GPU once and slice batches out of it with index tensors.
That's simpler and faster than PyTorch's Dataset/DataLoader machinery, which
pays off when data doesn't fit in memory or needs on-the-fly loading.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_ROOT / "synth"), str(PROJECT_ROOT / "features")]
import extract  # noqa: E402
import tuning  # noqa: E402

N_STRINGS = 6


@dataclass(frozen=True)
class PositionMap:
    """Converts between class numbers and (string, fret)."""

    max_fret: int = 19

    @property
    def n_frets(self) -> int:
        return self.max_fret + 1

    @property
    def n_classes(self) -> int:
        return N_STRINGS * self.n_frets

    def to_class(self, string: int, fret: int) -> int:
        return (string - 1) * self.n_frets + fret

    def to_position(self, cls: int) -> tuple[int, int]:
        return cls // self.n_frets + 1, cls % self.n_frets

    def class_midi(self) -> np.ndarray:
        """MIDI pitch of every class: lets us turn a predicted position into a
        predicted pitch."""
        return np.array(
            [tuning.string_fret_to_midi(*self.to_position(c)) for c in range(self.n_classes)]
        )


def frames_for_seconds(seconds: float, meta: dict) -> int:
    """How many CQT frames cover the first `seconds` of each note.
    (Frame k is centered at k * hop / sample_rate seconds.)"""
    return min(meta["n_frames"], 1 + math.floor(seconds * meta["sample_rate"] / meta["hop_length"]))


def load_norm_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads(Path(stats_path).read_text())
    if stats.get("computed_from") != "train":
        raise ValueError(f"{stats_path}: normalization stats must come from the train split")
    return np.asarray(stats["mean_per_bin"], np.float32), np.asarray(stats["std_per_bin"], np.float32)


def read_rows(data_dir: Path, split: str | None) -> list[dict]:
    """Label rows for one split (train/val/test), or all rows if split is None.
    Each row gets a `row` key = its index in features.npy."""
    data_dir = Path(data_dir)
    if split is None:
        rows = extract.read_labels(data_dir)
        return [{"row": str(i), **r} for i, r in enumerate(rows)]
    with open(data_dir / "splits" / f"{split}.csv", newline="") as f:
        return list(csv.DictReader(f))


def load_tensors(
    data_dir: Path,
    split: str | None,
    n_frames: int,
    norm_stats: Path,
    positions: PositionMap,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Return (X, y, rows) for one split.

    X: float32 (N, 1, n_bins, n_frames), normalized per frequency bin:
       x_norm = (x_dB - train_mean[bin]) / train_std[bin]
       The leading 1 is the "channel" axis: a CNN treats the spectrogram like
       a grayscale image (1 channel), as a photo would have 3 (RGB).
    y: int64 (N,) class numbers.

    `norm_stats` must be the *training* set's stats, even when loading val,
    test or a completely new dataset. The model only knows the scale it was
    trained on.
    """
    features, _ = extract.load_features(Path(data_dir))
    rows = read_rows(data_dir, split)
    idx = np.array([int(r["row"]) for r in rows])
    mean, std = load_norm_stats(norm_stats)

    x = np.asarray(features[idx, :, :n_frames], dtype=np.float32)
    x = (x - mean[None, :, None]) / std[None, :, None]
    y = np.array([positions.to_class(int(r["string"]), int(r["fret"])) for r in rows])
    if (y < 0).any() or (y >= positions.n_classes).any():
        raise ValueError("labels outside the model's position range (check max_fret)")
    X = torch.from_numpy(x).unsqueeze(1).to(device)
    return X, torch.from_numpy(y).long().to(device), rows


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():  # Apple GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

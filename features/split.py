"""Split a dataset into train / val / test and compute normalization stats.

    python features/split.py --data data/notes_v1

What the three splits are for
-----------------------------
* train (80%): the only data the model learns from.
* val (10%):   checked during training to pick settings (learning rate,
               model size, when to stop). Because we *choose* things based on
               it, it slowly stops being unbiased.
* test (10%):  locked away until the very end, then used once, to report how
               well the final model does on notes it has never seen in any way.

Why stratify by (string, fret)
------------------------------
A plain random 80/10/10 cut could, by chance, leave e.g. string 5 fret 17
out of the test set entirely. Then the test score says nothing about that
position, or the model never trains on it. Stratifying means splitting each
position's notes 80/10/10 *separately*, so every position is represented in
every split in the same proportion.

Outputs (in <data>/splits/)
---------------------------
train.csv, val.csv, test.csv: the labels.csv rows of that split, plus a `row`
    column = index into features.npy.
norm_stats.json: mean / std computed from TRAIN only (see below).
split_meta.json: seed, ratios, counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def stratified_split(
    rows: list[dict], ratios=DEFAULT_RATIOS, seed: int = 42
) -> dict[str, list[int]]:
    """Return {split name: sorted row indices}, split per (string, fret) group."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(int(row["string"]), int(row["fret"]))].append(i)

    # One generator for the whole split, walked through the groups in a fixed
    # (sorted) order: same seed + same labels -> same split, every time.
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
    out: dict[str, list[int]] = {name: [] for name in SPLITS}
    for position in sorted(groups):
        idx = groups[position]
        n = len(idx)
        if n < len(SPLITS):
            raise ValueError(
                f"position string={position[0]} fret={position[1]} has only {n} notes; "
                f"need at least {len(SPLITS)} so every split gets one. Generate more notes."
            )
        shuffled = [idx[k] for k in rng.permutation(n)]
        # Cut points from cumulative ratios, e.g. n=41: train 33, val 4, test 4.
        # Each split is guaranteed at least one note.
        n_train = min(n - 2, max(1, round(n * ratios[0])))
        n_val = min(n - n_train - 1, max(1, round(n * (ratios[0] + ratios[1])) - n_train))
        out["train"] += shuffled[:n_train]
        out["val"] += shuffled[n_train : n_train + n_val]
        out["test"] += shuffled[n_train + n_val :]
    return {name: sorted(v) for name, v in out.items()}


def compute_norm_stats(features: np.ndarray, train_rows: list[int], chunk: int = 256) -> dict:
    """Mean and standard deviation of the TRAIN features, overall and per bin.

    Why normalize: neural nets train best when inputs are centered around 0
    with a spread of ~1. Our dB values sit around -60 with a spread of ~15,
    so later we'll feed the model (x - mean) / std.

    Why TRAIN ONLY (data leakage): the val and test sets stand in for
    "notes the model has never seen". If their values helped compute the mean
    and std, a little information about them would be baked into the
    preprocessing, so the model would be (slightly) prepared for exactly those
    notes, and val/test scores would come out optimistic. In real use we can't
    compute stats from future audio; the honest setup is to fix every
    preprocessing number from training data and then apply it unchanged to
    val, test and real-world input. The same rule applies to anything that is
    *fitted*: stats, vocabularies, thresholds.

    Per-bin stats: each frequency row gets its own mean/std (over all train
    notes and frames). Low bins (where fundamentals live) and high bins (only
    faint harmonics and hiss) have very different typical levels; per-bin
    normalization puts them on equal footing. The global pair is saved too.
    Accumulated in float64 (sums of squares of ~60^2 over millions of values
    would lose precision in float16/32).
    """
    n_bins = features.shape[1]
    total = np.zeros(n_bins)
    total_sq = np.zeros(n_bins)
    count = 0
    rows = np.asarray(train_rows)
    for c0 in range(0, len(rows), chunk):
        x = np.asarray(features[rows[c0 : c0 + chunk]], dtype=np.float64)  # (b, bins, frames)
        total += x.sum(axis=(0, 2))
        total_sq += (x * x).sum(axis=(0, 2))
        count += x.shape[0] * x.shape[2]
    mean_bin = total / count
    std_bin = np.sqrt(np.maximum(total_sq / count - mean_bin**2, 0.0))
    mean = total.sum() / (count * n_bins)
    std = float(np.sqrt(max(total_sq.sum() / (count * n_bins) - mean**2, 0.0)))
    return {
        "computed_from": "train",
        "n_notes": len(rows),
        "values_per_bin": count,
        "mean": float(mean),
        "std": std,
        "mean_per_bin": [round(float(v), 6) for v in mean_bin],
        "std_per_bin": [round(float(v), 6) for v in std_bin],
        "usage": "x_norm = (x - mean_per_bin[:, None]) / std_per_bin[:, None]  (x: bins x frames)",
    }


def write_split_csv(path: Path, rows: list[dict], indices: list[int]) -> None:
    columns = ["row"] + list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for i in indices:
            writer.writerow({"row": i, **rows[i]})


def make_splits(data_dir: Path, seed: int = 42, ratios=DEFAULT_RATIOS) -> dict:
    data_dir = Path(data_dir)
    features, feat_meta = extract.load_features(data_dir)  # also checks alignment
    rows = extract.read_labels(data_dir)
    if len(rows) != features.shape[0]:
        raise RuntimeError(f"labels.csv has {len(rows)} rows but features.npy has {features.shape[0]}")

    splits = stratified_split(rows, ratios, seed)
    out_dir = data_dir / "splits"
    out_dir.mkdir(exist_ok=True)
    for name in SPLITS:
        write_split_csv(out_dir / f"{name}.csv", rows, splits[name])

    stats = compute_norm_stats(features, splits["train"])
    (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "stratified_by": ["string", "fret"],
        "counts": {name: len(splits[name]) for name in SPLITS},
        "positions": len({(r["string"], r["fret"]) for r in rows}),
        "labels_sha256": feat_meta["labels_sha256"],
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stratified train/val/test split + train-only norm stats.")
    p.add_argument("--data", required=True, help="dataset folder with features.npy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=float, nargs=3, default=DEFAULT_RATIOS, metavar=("TRAIN", "VAL", "TEST"))
    args = p.parse_args(argv)
    meta = make_splits(Path(args.data), args.seed, tuple(args.ratios))
    c = meta["counts"]
    n = sum(c.values())
    print(
        f"Splits in {args.data}/splits/: "
        + ", ".join(f"{k} {v} ({100 * v / n:.1f}%)" for k, v in c.items())
        + f"; {meta['positions']} positions in every split"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

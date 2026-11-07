"""Evaluate the trained CNN and the no-ML baseline on held-out notes.

    python model/evaluate.py --run runs/tiny_cnn_v1 --data data/notes_v1 --split test
    python model/evaluate.py --run runs/tiny_cnn_v1 --data data/notes_fresh1000 --split all

Writes to <run>/eval/<name>/ (name defaults to "<dataset>_<split>"):
  report.txt              the numbers below, CNN vs baseline side by side
  metrics.json            the same, machine-readable
  string_confusion.png    6x6 string confusion matrices (pitch-correct notes)
  predictions.csv         every note: true position, CNN guess, baseline guess

Only ever evaluate on notes the model never trained on (the test split, or
a freshly generated dataset). Train/val numbers are optimistic: the model
was fitted to train, and we picked the checkpoint by val.

The metrics, and why each one
-----------------------------
* position accuracy:  string AND fret right. The real goal.
* pitch accuracy:     the predicted position's pitch is right (e.g. guessing
                      s1 f0 for s2 f5 is still pitch-correct: both are E4).
* string accuracy when pitch is correct: of the notes where the pitch was
  right, how often was the string right too? THE key metric. Pitch is
  "easy" (even the baseline gets 100%); choosing among the 1-5 strings that
  can play a pitch requires hearing the string's timbre. We also report the
  chance level: guessing uniformly among the valid strings for that pitch.
* string confusion matrix: for pitch-correct notes, true string (rows) vs
  predicted string (columns). A perfect model is a diagonal. Off-diagonal
  cells show *which* strings get mixed up (usually neighbours).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import dataset as D  # noqa: E402
from tiny_cnn import load_checkpoint, predict_logits  # noqa: E402

FRET_REGIONS = [(0, 4), (5, 9), (10, 14), (15, 19)]


def position_name(positions: D.PositionMap, cls: int) -> str:
    s, f = positions.to_position(int(cls))
    return f"s{s} f{f}"


def compute_metrics(true: np.ndarray, pred: np.ndarray, positions: D.PositionMap) -> dict:
    true, pred = np.asarray(true), np.asarray(pred)
    midi = positions.class_midi()
    true_string = true // positions.n_frets + 1
    pred_string = pred // positions.n_frets + 1
    true_fret = true % positions.n_frets
    pos_ok = pred == true
    pitch_ok = midi[pred] == midi[true]

    # How many strings could play each true pitch -> chance level for the
    # "which string?" question.
    n_options = Counter(midi.tolist())
    chance = float(np.mean([1.0 / n_options[m] for m in midi[true]]))

    confusion = np.zeros((6, 6), dtype=int)  # [true string - 1, predicted string - 1]
    for t, p in zip(true_string[pitch_ok], pred_string[pitch_ok]):
        confusion[t - 1, p - 1] += 1

    mistakes = Counter(
        (position_name(positions, t), position_name(positions, p), bool(midi[p] == midi[t]))
        for t, p in zip(true[~pos_ok], pred[~pos_ok])
    )
    return {
        "n": int(len(true)),
        "position_acc": float(pos_ok.mean()),
        "pitch_acc": float(pitch_ok.mean()),
        "string_acc_given_pitch": float(pos_ok[pitch_ok].mean()) if pitch_ok.any() else 0.0,
        "string_acc_given_pitch_chance": chance,
        "per_string_acc": {
            str(s): float(pos_ok[true_string == s].mean()) for s in range(1, 7) if (true_string == s).any()
        },
        "per_fret_region_acc": {
            f"{lo}-{hi}": float(pos_ok[(true_fret >= lo) & (true_fret <= hi)].mean())
            for lo, hi in FRET_REGIONS
            if ((true_fret >= lo) & (true_fret <= hi)).any()
        },
        "string_confusion_pitch_correct": confusion.tolist(),
        "top_mistakes": [
            {"true": t, "pred": p, "pitch_correct": pc, "count": n}
            for (t, p, pc), n in mistakes.most_common(10)
        ],
    }


def plot_confusions(results: dict, path: Path, title: str) -> None:
    names = [f"{s} ({D.tuning.midi_to_name(D.tuning.OPEN_STRING_MIDI[s])})" for s in range(1, 7)]
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.6))
    for ax, (label, m) in zip(np.atleast_1d(axes), results.items()):
        cm = np.array(m["string_confusion_pitch_correct"])
        # Color = share of each true string's notes (row-normalized), so rare
        # and common rows are comparable; numbers = raw counts.
        frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                            color="white" if frac[i, j] > 0.5 else "black")
        ax.set_xticks(range(6), names, fontsize=8)
        ax.set_yticks(range(6), names, fontsize=8)
        ax.set_xlabel("predicted string")
        ax.set_ylabel("true string")
        ax.set_title(
            f"{label}\nstring accuracy when pitch is right: {100 * m['string_acc_given_pitch']:.1f}%",
            fontsize=10,
        )
    fig.suptitle(f"{title}: string confusion (pitch-correct notes only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def format_report(results: dict, header: str) -> str:
    labels = list(results)
    w = 26
    lines = [header, "", f"{'':{w}s}" + "".join(f"{lab:>14s}" for lab in labels)]

    def row(name, get):
        lines.append(f"{name:{w}s}" + "".join(f"{100 * get(results[lab]):13.1f}%" for lab in labels))

    row("position accuracy", lambda m: m["position_acc"])
    row("pitch accuracy", lambda m: m["pitch_acc"])
    row("string acc | pitch right", lambda m: m["string_acc_given_pitch"])
    lines.append(f"{'  (chance level)':{w}s}{100 * results[labels[0]]['string_acc_given_pitch_chance']:13.1f}%")
    lines += ["", "position accuracy by string:"]
    for s in results[labels[0]]["per_string_acc"]:
        row(f"  string {s}", lambda m, s=s: m["per_string_acc"][s])
    lines += ["", "position accuracy by fret region:"]
    for region in results[labels[0]]["per_fret_region_acc"]:
        row(f"  frets {region}", lambda m, r=region: m["per_fret_region_acc"][r])
    for lab in labels:
        lines += ["", f"{lab}: top mistakes"]
        for k in results[lab]["top_mistakes"]:
            tag = "(same pitch, wrong string)" if k["pitch_correct"] else "(wrong pitch)"
            lines.append(f"  {k['count']:4d}x  {k['true']:>7s} predicted as {k['pred']:<7s} {tag}")
        if not results[lab]["top_mistakes"]:
            lines.append("  none")
    return "\n".join(lines)


def evaluate(run: Path, data_dir: Path, split: str, name: str | None = None, device: str = "auto") -> dict:
    run, data_dir = Path(run), Path(data_dir)
    dev = D.pick_device(device)
    model, ckpt = load_checkpoint(run / "best.pt", dev)
    positions = D.PositionMap(ckpt["max_fret"])
    split_arg = None if split == "all" else split

    # Always normalize with the TRAINING set's stats from the checkpoint,
    # even for a brand-new dataset (that's how the model will meet real audio).
    X, y, rows = D.load_tensors(
        data_dir, split_arg, model.config.n_frames, Path(ckpt["norm_stats"]), positions, dev
    )
    true = y.cpu().numpy()
    cnn_pred = predict_logits(model, X).argmax(1).cpu().numpy()

    features, _ = D.extract.load_features(data_dir)
    base_pred = baseline.predict_classes(features[np.array([int(r["row"]) for r in rows])], positions)

    results = {
        "TinyCNN": compute_metrics(true, cnn_pred, positions),
        "baseline": compute_metrics(true, base_pred, positions),
    }

    name = name or f"{data_dir.name}_{split}"
    out = run / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    header = f"Evaluation: {run.name} on {data_dir} [{split}], {len(true)} notes (checkpoint epoch {ckpt['epoch']})"
    report = format_report(results, header)
    (out / "report.txt").write_text(report + "\n")
    (out / "metrics.json").write_text(json.dumps({"header": header, **results}, indent=2))
    plot_confusions(results, out / "string_confusion.png", f"{data_dir.name} [{split}]")
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row", "file", "true", "cnn", "baseline", "cnn_correct", "baseline_correct"])
        for r, t, c, b in zip(rows, true, cnn_pred, base_pred):
            w.writerow([r["row"], r["file"], position_name(positions, t), position_name(positions, c),
                        position_name(positions, b), int(c == t), int(b == t)])
    print(report)
    print(f"\n-> {out}")
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate TinyCNN vs the baseline.")
    p.add_argument("--run", default="runs/tiny_cnn_v1", help="training run folder with best.pt")
    p.add_argument("--data", required=True, help="dataset folder with features.npy")
    p.add_argument("--split", default="test", help="test (default), val, train, or all (for fresh datasets)")
    p.add_argument("--name", help="output subfolder name (default <dataset>_<split>)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    evaluate(Path(args.run), Path(args.data), args.split, args.name, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

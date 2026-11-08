"""Compare training runs side by side.

    python model/compare.py --runs runs/tiny_cnn_v1 runs/tiny_cnn_v2 runs/tiny_cnn_v1j \
        --evals notes_v1_test notes_fresh1000_all real_session1_all \
        --confusion real_session1_all --out runs/comparison

Reads what train.py and evaluate.py already wrote (config.json, history.csv,
eval/<name>/metrics.json); it doesn't re-run anything. Writes:
  <out>.md               the tables below (also printed)
  <out>_curves.png       all runs' learning curves on one plot
  <out>_confusion.png    string confusion on the --confusion eval set, one
                         panel per run + the baseline

The train/val gap
-----------------
At the best epoch, compare train and val loss/accuracy. A big gap means the
model fits its training notes much better than unseen ones (overfitting).
More data usually shrinks it: with 10x the examples, memorizing each one
stops being the easiest way to lower the loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402

METRICS = [
    ("position_acc", "position"),
    ("pitch_acc", "pitch"),
    ("string_acc_given_pitch", "string (when pitch right)"),
]


def load_run(run: Path) -> dict:
    cfg = json.loads((run / "config.json").read_text())
    with open(run / "history.csv") as f:
        history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    best = history[cfg["best_epoch"] - 1]
    return {"name": run.name, "cfg": cfg, "history": history, "best": best}


def load_eval(run: Path, name: str) -> dict | None:
    path = run / "eval" / name / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def build_tables(runs: list[Path], evals: list[str]) -> str:
    info = [load_run(r) for r in runs]
    names = [i["name"] for i in info]
    out = []

    out.append("## Training\n")
    out.append("| | " + " | ".join(names) + " |")
    out.append("|---|" + "---|" * len(names))
    out.append("| training data | " + " | ".join(i["cfg"]["train_config"]["data"] for i in info) + " |")
    out.append("| parameters | " + " | ".join(f"{i['cfg']['parameters']:,}" for i in info) + " |")
    out.append("| best epoch / epochs run | " + " | ".join(f"{i['cfg']['best_epoch']} / {i['cfg']['epochs_run']}" for i in info) + " |")
    out.append("| training time | " + " | ".join(f"{i['cfg']['train_seconds']:.0f} s" for i in info) + " |")
    out.append("| train loss (best epoch) | " + " | ".join(f"{i['best']['train_loss']:.4f}" for i in info) + " |")
    out.append("| val loss (best epoch) | " + " | ".join(f"{i['best']['val_loss']:.4f}" for i in info) + " |")
    out.append("| train acc (best epoch) | " + " | ".join(pct(i["best"]["train_acc"]) for i in info) + " |")
    out.append("| val acc (best epoch) | " + " | ".join(pct(i["best"]["val_acc"]) for i in info) + " |")
    out.append("| **train − val acc gap** | " + " | ".join(
        f"**{100 * (i['best']['train_acc'] - i['best']['val_acc']):.1f} pts**" for i in info) + " |")

    out.append("\n## Evaluation (held-out notes only)\n")
    cols = names + ["baseline"]
    out.append("| eval set | metric | " + " | ".join(cols) + " |")
    out.append("|---|---|" + "---|" * len(cols))
    for ev in evals:
        results = [load_eval(r, ev) for r in runs]
        base = next((res["baseline"] for res in results if res), None)
        n = next((res["TinyCNN"]["n"] for res in results if res), None)
        for key, label in METRICS:
            cells = [pct(res["TinyCNN"][key]) if res else "–" for res in results]
            cells.append(pct(base[key]) if base else "–")
            out.append(f"| {ev} ({n if n else '?'}) | {label} | " + " | ".join(cells) + " |")
    out.append("\n'–' = not evaluated (e.g. a run's own test split exists only for its dataset).")
    return "\n".join(out)


def plot_curves(runs: list[Path], path: Path) -> None:
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, run in enumerate(runs):
        info = load_run(run)
        ep = [h["epoch"] for h in info["history"]]
        color = f"C{k}"
        ax_l.plot(ep, [h["train_loss"] for h in info["history"]], "--", color=color, label=f"{info['name']} train")
        ax_l.plot(ep, [h["val_loss"] for h in info["history"]], "-", color=color, label=f"{info['name']} val")
        ax_a.plot(ep, [100 * h["train_acc"] for h in info["history"]], "--", color=color)
        ax_a.plot(ep, [100 * h["val_acc"] for h in info["history"]], "-", color=color, label=info["name"])
    ax_l.set_yscale("log")
    ax_l.set_ylabel("loss (log)")
    ax_a.set_ylabel("accuracy (%)")
    ax_a.set_ylim(0, 101)
    for ax in (ax_l, ax_a):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    ax_l.legend(fontsize=7, ncol=2)
    ax_a.legend(fontsize=8, title="solid = val, dashed = train", title_fontsize=8)
    ax_l.set_title("Loss: the gap between dashed and solid = overfitting")
    ax_a.set_title("Position accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(runs: list[Path], eval_name: str, path: Path) -> bool:
    results = {}
    for run in runs:
        res = load_eval(run, eval_name)
        if res:
            results[run.name] = res["TinyCNN"]
            results.setdefault("_baseline", res["baseline"])
    if not results:
        return False
    results["baseline"] = results.pop("_baseline")
    evaluate.plot_confusions(results, path, eval_name)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Side-by-side comparison of training runs.")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--evals", nargs="+", required=True, help="eval folder names under <run>/eval/")
    p.add_argument("--confusion", help="eval name to draw string confusions for")
    p.add_argument("--out", default="runs/comparison", help="output path prefix")
    args = p.parse_args(argv)

    runs = [Path(r) for r in args.runs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tables = build_tables(runs, args.evals)
    out.with_suffix(".md").write_text(f"# Run comparison\n\n{tables}\n")
    plot_curves(runs, out.parent / f"{out.name}_curves.png")
    print(tables)
    print(f"\n-> {out.with_suffix('.md')}, {out.parent / (out.name + '_curves.png')}")
    if args.confusion:
        cpath = out.parent / f"{out.name}_confusion.png"
        if plot_confusion(runs, args.confusion, cpath):
            print(f"-> {cpath}")
        else:
            print(f"(no evaluations named {args.confusion} yet; no confusion plot)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

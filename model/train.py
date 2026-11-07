"""Train TinyCNN to predict (string, fret) from a note's CQT.

    python model/train.py --data data/notes_v1 --run runs/tiny_cnn_v1

The training loop, in one paragraph
-----------------------------------
Show the model a batch of 64 spectrograms. It outputs 120 scores per note.
The *loss* (cross-entropy) measures how much probability it put on the right
answer: about 0 if it was confident and right, large if it was confident and
wrong. Backpropagation computes, for every parameter, which direction would
reduce that loss (the *gradient*), and the optimizer nudges each parameter a
small step that way. One pass over all training notes = one *epoch*. After
each epoch we score the model on the validation set, which it never trains on.

Choices
-------
* Cross-entropy: the standard loss for "pick one of N classes". It equals
  -log(probability assigned to the correct class).
* AdamW: Adam adapts the step size per parameter (big steps for parameters
  with small, consistent gradients; careful steps for noisy ones). The "W" is
  decoupled weight decay: each step also shrinks every weight slightly toward
  0, a mild push toward simpler solutions that generalize better.
* lr 1e-3: Adam's well-tested default step size.
* Early stopping: keep the checkpoint with the lowest *validation* loss and
  stop when it hasn't improved for `patience` epochs. Training longer past
  that point only memorizes the training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402
from tiny_cnn import TinyCNN, TinyCNNConfig, count_parameters, parameter_report, predict_logits  # noqa: E402


def set_seed(seed: int) -> None:
    """Fix every random number generator involved: weight initialization,
    batch order, etc. On the GPU some operations are still not bit-for-bit
    deterministic, so two runs can differ in the last decimals."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds MPS / CUDA


def loss_and_accuracy(model, X, y, batch_size=256) -> tuple[float, float]:
    logits = predict_logits(model, X, batch_size)
    loss = nn.functional.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def overfit_one_batch(config: TinyCNNConfig, X, y, lr: float, steps: int = 300, seed: int = 0) -> dict:
    """Sanity check: can the model memorize 64 examples?

    A model with ~280k parameters can trivially memorize 64 notes. If it
    *can't* drive their loss to ~0 and accuracy to 100%, something upstream
    is broken: labels misaligned with inputs, the loss wired to the wrong
    outputs, gradients not flowing, a learning rate wildly off, inputs that
    are all zeros or NaN... Each of those bugs can otherwise hide behind
    "the model just isn't learning well" for days. It costs seconds, so we do
    it before every real training run.
    """
    set_seed(seed)
    model = TinyCNN(config).to(X.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        model.train()
        loss = nn.functional.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            eval_loss, acc = loss_and_accuracy(model, X, y)
            if acc == 1.0 and eval_loss < 0.05:
                break
    eval_loss, acc = loss_and_accuracy(model, X, y)
    return {"steps": step, "batch_size": len(y), "final_loss": eval_loss, "final_accuracy": acc}


def plot_learning_curves(history: list[dict], best_epoch: int, path: Path) -> None:
    """How to read this plot (and spot overfitting):

    * Both losses falling together: the model is learning patterns that
      generalize. Keep going.
    * Train loss keeps falling while val loss flattens and then RISES, i.e. a
      widening gap between the curves: OVERFITTING. The model is memorizing
      quirks of the training notes (their particular noise, their exact
      pick strengths) instead of learning what a string sounds like. The best
      checkpoint is at the bottom of the val curve (the dashed line); early
      stopping is what keeps us from shipping the later, worse weights.
    * Both stuck high: UNDERFITTING. The model is too small, the lr is
      wrong, or the inputs don't contain the answer.
    * Val accuracy can keep creeping up for a bit while val loss rises: the
      model is getting more *overconfident* on the notes it gets wrong.
      Loss notices that; accuracy doesn't.
    Note the train numbers are averaged over the epoch while the weights were
    still changing (and in train mode), so early on train can even look worse
    than val.
    """
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], "o-", label="val")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("cross-entropy loss (log scale)")
    ax_acc.plot(epochs, [100 * h["train_acc"] for h in history], "o-", label="train")
    ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], "o-", label="val")
    ax_acc.set_ylabel("position accuracy (%)")
    ax_acc.set_ylim(0, 101)
    for ax in (ax_loss, ax_acc):
        ax.axvline(best_epoch, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    ax_loss.set_title(f"Loss (best val at epoch {best_epoch}, dashed)")
    ax_acc.set_title("Accuracy (string AND fret both right)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(args) -> dict:
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)
    device = D.pick_device(args.device)
    set_seed(args.seed)

    feat_meta = json.loads((data_dir / "features_meta.json").read_text())
    synth_meta = json.loads((data_dir / "metadata.json").read_text())
    positions = D.PositionMap(max_fret=int(synth_meta["max_fret"]))
    n_frames = D.frames_for_seconds(args.input_seconds, feat_meta)
    norm_stats = data_dir / "splits" / "norm_stats.json"

    X_train, y_train, _ = D.load_tensors(data_dir, "train", n_frames, norm_stats, positions, device)
    X_val, y_val, _ = D.load_tensors(data_dir, "val", n_frames, norm_stats, positions, device)

    config = TinyCNNConfig(
        n_bins=feat_meta["n_bins"],
        n_frames=n_frames,
        input_seconds=args.input_seconds,
        n_classes=positions.n_classes,
    )
    model = TinyCNN(config).to(device)
    print(f"Device: {device}   train {len(y_train)}  val {len(y_val)}   input {tuple(X_train.shape[1:])}")
    print(f"TinyCNN: {count_parameters(model):,} parameters\n{parameter_report(model)}\n")

    # --- 1. sanity check ------------------------------------------------------
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(len(y_train), generator=g)[:64].to(device)
    t0 = time.perf_counter()
    sanity = overfit_one_batch(config, X_train[pick], y_train[pick], args.lr, seed=args.seed)
    sanity["seconds"] = round(time.perf_counter() - t0, 2)
    print(
        f"Sanity check (overfit 1 batch of 64): accuracy {100 * sanity['final_accuracy']:.1f}%, "
        f"loss {sanity['final_loss']:.4f} after {sanity['steps']} steps ({sanity['seconds']}s)"
    )
    (run_dir / "sanity_check.json").write_text(json.dumps(sanity, indent=2))
    if sanity["final_accuracy"] < 0.99 and not args.skip_sanity:
        raise SystemExit("Sanity check FAILED: the model can't memorize one batch. Something is broken.")

    # --- 2. real training -----------------------------------------------------
    set_seed(args.seed)
    model = TinyCNN(config).to(device)  # fresh weights, not the memorized ones
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best_val, best_epoch, bad_epochs = [], float("inf"), 0, 0
    train_config = {k: v for k, v in vars(args).items()}
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Shuffle every epoch so batches differ each time: the gradient then
        # isn't biased by a fixed grouping of notes.
        order = torch.randperm(len(y_train), generator=g).to(device)
        total_loss, total_correct = 0.0, 0
        for b0 in range(0, len(order), args.batch_size):
            idx = order[b0 : b0 + args.batch_size]
            logits = model(X_train[idx])
            loss = nn.functional.cross_entropy(logits, y_train[idx])
            opt.zero_grad()  # gradients accumulate by default; clear last step's
            loss.backward()  # backpropagation: d(loss)/d(parameter) for all parameters
            opt.step()  # move every parameter a small step downhill
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(1) == y_train[idx]).sum().item()

        val_loss, val_acc = loss_and_accuracy(model, X_val, y_val)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(y_train),
            "train_acc": total_correct / len(y_train),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": round(time.perf_counter() - t_start, 2),
        }
        history.append(row)

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": config.to_dict(),
                    "train_config": train_config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "max_fret": positions.max_fret,
                    "norm_stats": str(norm_stats),
                    "data": str(data_dir),
                },
                run_dir / "best.pt",
            )
        else:
            bad_epochs += 1
        print(
            f"epoch {epoch:2d}  train loss {row['train_loss']:.4f} acc {100 * row['train_acc']:5.1f}%  |  "
            f"val loss {val_loss:.4f} acc {100 * val_acc:5.1f}%  {'*best*' if improved else ''}"
        )
        if bad_epochs >= args.patience:
            print(f"Early stop: val loss hasn't improved for {args.patience} epochs.")
            break

    train_seconds = time.perf_counter() - t_start
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    plot_learning_curves(history, best_epoch, run_dir / "learning_curves.png")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "parameters": count_parameters(model),
        "model_config": config.to_dict(),
        "train_config": train_config,
        "sanity_check": sanity,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_val_acc": history[best_epoch - 1]["val_acc"],
        "epochs_run": len(history),
        "train_seconds": round(train_seconds, 1),
        "torch": torch.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nBest epoch {best_epoch}: val loss {best_val:.4f}, val acc "
        f"{100 * summary['best_val_acc']:.1f}%.  Training took {train_seconds:.1f}s.  -> {run_dir}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train TinyCNN on CQT features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="dataset folder with features.npy and splits/")
    p.add_argument("--run", default="runs/tiny_cnn_v1", help="output folder for checkpoint, logs, plots")
    p.add_argument("--input-seconds", type=float, default=1.0, help="use the first N seconds of each note")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    p.add_argument("--skip-sanity", action="store_true", help="don't stop if the sanity check fails")
    return p


def main(argv=None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())

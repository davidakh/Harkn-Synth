"""Draw spectrograms to build intuition about what the model will see.

    python tools/show.py --data data/notes_v1

Writes two PNGs to <data>/figures/. Spectrograms are computed from the WAVs
with synth/spectrogram.py, the same CQT Electro CNN trains on, so no feature
extraction is needed first.

same_pitch_E4.png
    E4 (330 Hz) can be played in five places: string 1 open, 2 fret 5,
    3 fret 9, 4 fret 14, 5 fret 19. The pitch is identical, so the *only*
    thing that can tell a tab model which string was used is timbre: how
    bright the note is, how fast its harmonics fade. This is the hardest part
    of audio -> tab.
      row 1: rendered with identical playing parameters (same pick strength,
             same pluck position), so any difference is the string itself
      row 2: real examples from the dataset (random playing parameters), i.e.
             what the model actually has to cope with
      row 3: average spectrum of row 1 over the first 0.5 s, overlaid.
             Look at how quickly each curve drops from one harmonic to the next.

open_strings.png
    One open-string note per string from the dataset (low E -> high E).
    Note the fundamental (dashed line) and the ladder of harmonics above it.
    In a CQT the rungs get closer together going up (2f0 is 12 semitones
    above f0, 3f0 only 7 more, 4f0 only 5 more...), and that pattern is the
    same shape at every pitch.

Things to notice in both figures
    * The bright wedge in the bottom-left corner is the pluck's attack,
      smeared in time. Low CQT bins need long analysis windows (hundreds of
      ms) to resolve semitones at 80 Hz, so a sudden event blurs sideways
      there; high bins use short windows and stay sharp. That's the
      time/frequency trade-off: you can't have both at once.
    * The speckled purple background is the synth's background hiss
      (around -60 dB).

Axes: y is labeled in note names (each E and C), x in seconds. All panels in a
figure share one color scale, so brightness is comparable across panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "synth"))
import electro_synth as es  # noqa: E402
import spectrogram  # noqa: E402
import tuning  # noqa: E402

E4_POSITIONS = [(1, 0), (2, 5), (3, 9), (4, 14), (5, 19)]
VMIN_DB = -90.0  # bottom of the color scale (background hiss sits around -60)
SPECTRUM_FRAMES = slice(2, 22)  # ~0.05-0.5 s: after the pluck transient


def first_row_for(rows: list[dict], string: int, fret: int) -> int | None:
    for i, r in enumerate(rows):
        if int(r["string"]) == string and int(r["fret"]) == fret:
            return i
    return None


def draw_spectrogram(ax, spec_db, meta, vmax, title, midi=None, ylabel=True):
    seconds = spec_db.shape[1] * meta["hop_length"] / meta["sample_rate"]
    im = ax.imshow(
        spec_db,
        origin="lower",  # low frequencies at the bottom, like a piano roll
        aspect="auto",
        extent=[0, seconds, -0.5, spec_db.shape[0] - 0.5],
        cmap="magma",
        vmin=VMIN_DB,
        vmax=vmax,
        interpolation="nearest",
    )
    ticks, names = spectrogram.note_ticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=7)
    if ylabel:
        ax.set_ylabel("pitch (note)")
    else:
        # hide (not erase) the labels: with shared axes, erasing them on one
        # panel would erase them on all
        ax.tick_params(labelleft=False)
    ax.set_xticks(np.arange(0.0, seconds - 0.25, 0.5))
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    if midi is not None:
        ax.axhline(spectrogram.midi_to_bin(midi), color="cyan", lw=0.7, ls="--", alpha=0.7)
    ax.set_title(title, fontsize=8)
    return im


def controlled_e4(sample_rate: int, duration: float) -> list[np.ndarray]:
    specs = []
    for string, fret in E4_POSITIONS:
        params = es.default_params(string, fret)
        audio, _ = es.render_note(params, sample_rate, duration, es.sample_rng(0, 1))
        specs.append(spectrogram.cqt_db(audio, sample_rate))
    return specs


def plot_same_pitch(spectrograms, meta, rows, out_path: Path, duration: float) -> None:
    sr = meta["sample_rate"]
    controlled = controlled_e4(sr, duration)
    dataset = []
    for string, fret in E4_POSITIONS:
        i = first_row_for(rows, string, fret)
        dataset.append((i, None if i is None else np.asarray(spectrograms[i], dtype=np.float32)))
    vmax = max([s.max() for s in controlled] + [d.max() for _, d in dataset if d is not None])

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.8], hspace=0.45, wspace=0.08)
    e4 = tuning.string_fret_to_midi(1, 0)
    im = None
    for col, (string, fret) in enumerate(E4_POSITIONS):
        ax = fig.add_subplot(grid[0, col])
        im = draw_spectrogram(
            ax, controlled[col], meta, vmax,
            f"string {string}, fret {fret}\nsame playing params", e4, ylabel=col == 0,
        )
        ax = fig.add_subplot(grid[1, col])
        i, spec = dataset[col]
        if spec is None:
            ax.text(0.5, 0.5, "not in dataset", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        r = rows[i]
        draw_spectrogram(
            ax, spec, meta, vmax,
            f"string {string}, fret {fret}  ({r['file']})\n"
            f"pick {float(r['pick_strength']):.2f}  pos {float(r['pluck_position']):.2f}  "
            f"decay x{float(r['decay_scale']):.2f}",
            e4, ylabel=col == 0,
        )
    fig.colorbar(im, ax=fig.axes, shrink=0.5, pad=0.01, label="level (dB)")

    # Row 3: overlaid average spectra
    ax = fig.add_subplot(grid[2, :])
    for (string, fret), spec in zip(E4_POSITIONS, controlled):
        ax.plot(spec[:, SPECTRUM_FRAMES].mean(axis=1), lw=1.2, label=f"string {string}, fret {fret}")
    for k in range(1, 9):  # harmonic k sits 12*log2(k) semitones above f0
        b = spectrogram.midi_to_bin(e4 + 12 * math.log2(k))
        ax.axvline(b, color="gray", lw=0.6, ls=":")
        ax.text(b, 1.0, f"{k}f0" if k > 1 else "f0", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="gray")
    ticks, names = spectrogram.note_ticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_xlim(spectrogram.midi_to_bin(e4 - 7), spectrogram.midi_to_bin(e4 + 12 * math.log2(12)))
    ax.set_ylim(VMIN_DB, vmax + 3)
    ax.set_xlabel("pitch (note)")
    ax.set_ylabel("avg level, 0.05-0.5 s (dB)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        "Same pitch, different strings: thicker strings lose their upper harmonics faster",
        fontsize=9, pad=16,
    )

    fig.suptitle("E4 (330 Hz) on strings 1-5", fontsize=13)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_open_strings(spectrograms, meta, rows, out_path: Path) -> None:
    picks = []
    for string in (6, 5, 4, 3, 2, 1):  # low E first, like reading the neck
        i = first_row_for(rows, string, 0)
        if i is not None:
            picks.append((string, i, np.asarray(spectrograms[i], dtype=np.float32)))
    vmax = max(spec.max() for _, _, spec in picks)

    fig, axes = plt.subplots(1, len(picks), figsize=(3 * len(picks), 5.5), sharey=True)
    for col, (ax, (string, i, spec)) in enumerate(zip(np.atleast_1d(axes), picks)):
        r = rows[i]
        midi = int(r["midi"])
        im = draw_spectrogram(
            ax, spec, meta, vmax,
            f"string {string} open: {tuning.midi_to_name(midi)} ({float(r['freq_hz']):.1f} Hz)\n"
            f"{r['file']}, pick {float(r['pick_strength']):.2f}",
            midi, ylabel=col == 0,
        )
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), shrink=0.8, pad=0.01, label="level (dB)")
    fig.suptitle("Open strings (dashed line = labeled fundamental)", fontsize=12)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def read_labels(data_dir: Path) -> list[dict]:
    with open(Path(data_dir) / "labels.csv", newline="") as f:
        return list(csv.DictReader(f))


class NoteSpectrograms:
    """spectrograms[i] = CQT (dB) of the note on row i of labels.csv.

    Computed from the WAV when asked for, so this script only needs the audio
    the synth wrote. The figures pick about a dozen notes, so it's quick.
    """

    def __init__(self, data_dir: Path, rows: list[dict]):
        self.data_dir, self.rows = Path(data_dir), rows

    def __getitem__(self, i: int) -> np.ndarray:
        audio, sr = sf.read(str(self.data_dir / self.rows[i]["file"]), dtype="float64")
        return spectrogram.cqt_db(audio, sr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot spectrogram figures for learning.")
    p.add_argument("--data", required=True, help="dataset folder (audio/ + labels.csv)")
    p.add_argument("--out", help="output folder (default: <data>/figures)")
    args = p.parse_args(argv)

    data_dir = Path(args.data)
    rows = read_labels(data_dir)
    synth_meta = json.loads((data_dir / "metadata.json").read_text())
    meta = {"sample_rate": int(synth_meta["sample_rate"]), "hop_length": spectrogram.HOP_LENGTH}
    duration = float(synth_meta["duration_s"])
    spectrograms = NoteSpectrograms(data_dir, rows)
    out_dir = Path(args.out) if args.out else data_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_same_pitch(spectrograms, meta, rows, out_dir / "same_pitch_E4.png", duration)
    plot_open_strings(spectrograms, meta, rows, out_dir / "open_strings.png")
    print(f"Wrote {out_dir / 'same_pitch_E4.png'}\nWrote {out_dir / 'open_strings.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

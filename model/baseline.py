"""No-ML baseline: detect the pitch, then guess the lowest-fret position.

    python model/baseline.py --data data/notes_v1 --split test

Why have a baseline at all?
---------------------------
A number like "95% accurate" means nothing on its own. The question is
always "compared to what?". This baseline uses zero learning:
  1. pitch = the strongest CQT bin (averaged over 0.05-0.5 s)
  2. position = where a beginner would play that pitch: the lowest fret
     (e.g. E4 -> string 1 open, never string 5 fret 19).

It gets the *pitch* right nearly always, but it can only pick the right
*string* when the note happened to be played at its lowest-fret position.
With our balanced dataset that's 44 of the 120 positions (~37%). If the CNN
lands near that, it learned pitch plus a habit. If it lands far above it,
it really learned to hear which string was played (timbre).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402

PITCH_FRAMES = slice(2, 22)  # ~0.05-0.5 s: after the pluck's smeared attack


def strongest_bin_midi(features_db: np.ndarray) -> np.ndarray:
    """(N, bins, frames) raw dB features -> (N,) detected MIDI pitch.

    Rounds to the nearest semitone: 3 bins per semitone, so bin 3k is in
    tune and 3k +/- 1 are +/-33 cents.
    """
    avg = np.asarray(features_db[:, :, PITCH_FRAMES], dtype=np.float32).mean(axis=2)
    bins = avg.argmax(axis=1)
    return np.rint(D.extract.bin_to_midi(bins)).astype(int)


def lowest_fret_position(midi: int, max_fret: int = 19) -> tuple[int, int]:
    """The (string, fret) with the smallest fret that can play this pitch.
    Pitches outside the neck's range snap to the nearest end."""
    candidates = [
        (fret, string)
        for string, open_midi in D.tuning.OPEN_STRING_MIDI.items()
        if 0 <= (fret := midi - open_midi) <= max_fret
    ]
    if not candidates:
        return (6, 0) if midi < D.tuning.OPEN_STRING_MIDI[6] else (1, max_fret)
    fret, string = min(candidates)
    return string, fret


def predict_classes(features_db: np.ndarray, positions: D.PositionMap) -> np.ndarray:
    midis = strongest_bin_midi(features_db)
    return np.array([positions.to_class(*lowest_fret_position(m, positions.max_fret)) for m in midis])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Strongest-bin pitch + lowest-fret baseline.")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", help="train, val, test, or all")
    p.add_argument("--max-fret", type=int, default=19)
    args = p.parse_args(argv)

    positions = D.PositionMap(args.max_fret)
    rows = D.read_rows(Path(args.data), None if args.split == "all" else args.split)
    features, _ = D.extract.load_features(Path(args.data))
    idx = np.array([int(r["row"]) for r in rows])
    pred = predict_classes(features[idx], positions)
    true = np.array([positions.to_class(int(r["string"]), int(r["fret"])) for r in rows])
    midi = positions.class_midi()
    print(
        f"baseline on {args.data} [{args.split}, {len(rows)} notes]: "
        f"position {100 * np.mean(pred == true):.1f}%, pitch {100 * np.mean(midi[pred] == midi[true]):.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

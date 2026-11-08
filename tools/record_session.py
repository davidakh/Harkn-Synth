"""Make a play list for recording a real-guitar test set.

    python tools/record_session.py --out data/real_session1

Why real recordings?
--------------------
Everything so far was trained AND tested on our own synth. A 96% score only
says the model learned the synth. The question that matters: does
anything carry over to a real guitar? That gap between the training
distribution and the real world is called the "sim-to-real" or "domain"
gap. A small, carefully labeled real test set measures it honestly. We never
train on it: it's a test set only.

What this writes (in --out)
---------------------------
playlist.csv   take order, string, fret, midi, note (used by slice_session.py)
playlist.txt   the same as a printable checklist + tab, also printed here

60 positions: 10 per string, with frets spread over the neck (one fret
from each pair 0-1, 2-3, ..., 18-19), shuffled into a random order with a
fixed seed. The order is random so the model can't benefit from "the next
note is usually nearby", and so tiredness, tuning drift, etc. are spread
evenly over all positions.

How to record (also printed below)
----------------------------------
* Tune to standard tuning (A4 = 440 Hz) right before you start.
* Clean tone, no effects. Phone, audio interface or laptop mic all work.
* Play the notes in list order, one at a time, ~2 seconds apart. Let each
  note ring, mute it just before the next one, and don't let open strings
  ring on.
* If you fumble a note: STOP, and restart the take from the beginning (or
  note the number and re-record). The slicer matches notes to the list
  purely by order, so one extra or missing note shifts every label after it.
  It will refuse to guess if the counts don't match.
* Save as a WAV file (any sample rate, mono or stereo).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "synth"))
import tuning  # noqa: E402

STRING_NAMES = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}  # tab line labels, high to low
STRING_WORDS = {1: "high E", 2: "B", 3: "G", 4: "D", 5: "A", 6: "low E"}


def make_playlist(per_string: int = 10, max_fret: int = 19, seed: int = 42) -> list[tuple[int, int]]:
    """(string, fret) pairs in take order.

    Frets are spread by splitting 0..max_fret into `per_string` equal bands
    and picking one fret at random from each band.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,)))
    bands = np.array_split(np.arange(max_fret + 1), per_string)
    positions = [
        (string, int(rng.choice(band))) for string in tuning.STRINGS for band in bands
    ]
    return [positions[i] for i in rng.permutation(len(positions))]


def tab_block(items: list[tuple[int, int, int]]) -> list[str]:
    """ASCII tab for a group of (take number, string, fret): one column per note."""
    width = 5
    header = "   " + "".join(f"{n:^{width}d}" for n, _, _ in items)
    lines = [header]
    for s in tuning.STRINGS:  # 1 (high e) on top, like real tab
        cells = "".join(
            (f"{fret}".center(width, "-") if string == s else "-" * width) for _, string, fret in items
        )
        lines.append(f"{STRING_NAMES[s]} |{cells}|")
    return lines


def format_playlist(playlist: list[tuple[int, int]], per_line: int = 10) -> str:
    out = ["Electro real-guitar session: play these notes IN ORDER, ~2 s apart.", ""]
    out.append("Checklist")
    for n, (s, f) in enumerate(playlist, start=1):
        note = tuning.midi_to_name(tuning.string_fret_to_midi(s, f))
        fret = "open" if f == 0 else f"fret {f}"
        where = f"string {s} ({STRING_WORDS[s]}),"
        out.append(f"  [ ] {n:2d}. {where:20s} {fret:7s} -> {note}")
    out += ["", "Tab (numbers above each column = take order)"]
    numbered = [(n, s, f) for n, (s, f) in enumerate(playlist, start=1)]
    for i in range(0, len(numbered), per_line):
        out += [""] + tab_block(numbered[i : i + per_line])
    out += [
        "",
        "Tips: tune to A440 first; clean tone; mute each note before the next;",
        "if you fumble, restart the take (the slicer matches notes by order only).",
    ]
    return "\n".join(out)


def write_session(out_dir: Path, per_string: int = 10, max_fret: int = 19, seed: int = 42) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    playlist = make_playlist(per_string, max_fret, seed)
    with open(out_dir / "playlist.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["take", "string", "fret", "midi", "note"])
        for n, (s, fr) in enumerate(playlist, start=1):
            midi = tuning.string_fret_to_midi(s, fr)
            w.writerow([n, s, fr, midi, tuning.midi_to_name(midi)])
    (out_dir / "playlist.txt").write_text(format_playlist(playlist) + "\n")
    (out_dir / "session_meta.json").write_text(
        json.dumps({"seed": seed, "per_string": per_string, "max_fret": max_fret, "count": len(playlist)}, indent=2)
    )
    return playlist


def simulate_take(session: Path, seed: int = 0, spacing: float = 2.0) -> Path:
    """Write <session>/simulated_take.wav: the synth playing the play list,
    made deliberately messy like a real take, to dry-run the slicer before
    recording: uneven spacing, notes overlapping (not muted), a quiet
    muting click before each note, room noise, stereo 44.1 kHz, low level.
    """
    import soundfile as sf
    import librosa

    sys.path.insert(0, str(PROJECT_ROOT / "synth"))
    import electro_synth as es

    with open(Path(session) / "playlist.csv", newline="") as f:
        playlist = list(csv.DictReader(f))
    sr = 22050
    rng = np.random.default_rng(seed)
    starts = 0.7 + np.arange(len(playlist)) * spacing + rng.uniform(-0.25, 0.25, len(playlist))
    take = np.zeros(int((starts[-1] + 2.5) * sr))
    for k, (entry, t0) in enumerate(zip(playlist, starts)):
        note_rng = es.sample_rng(seed, k)
        params = es.random_params(note_rng, int(entry["string"]), int(entry["fret"]))
        audio, _ = es.render_note(params, sr, 2.0, note_rng)
        i = int(t0 * sr)
        take[i : i + len(audio)] += audio
        click = int((t0 - 0.12) * sr)  # finger touching the old note, 120 ms early
        take[click : click + 60] += 0.03 * rng.standard_normal(60)
    take += 10 ** (-58 / 20) * rng.standard_normal(len(take))
    take *= 10 ** (-14 / 20) / np.abs(take).max()  # a quiet phone-level recording
    stereo = librosa.resample(take, orig_sr=sr, target_sr=44100)
    stereo = np.stack([stereo, 0.9 * stereo], axis=1)
    path = Path(session) / "simulated_take.wav"
    sf.write(str(path), stereo, 44100, subtype="PCM_16")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate a play list for a real-guitar recording session.")
    p.add_argument("--out", default="data/real_session1", help="session folder")
    p.add_argument("--per-string", type=int, default=10)
    p.add_argument("--max-fret", type=int, default=19)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--simulate", action="store_true",
                   help="also write simulated_take.wav (the synth playing the list) to test the slicer")
    args = p.parse_args(argv)
    playlist = write_session(Path(args.out), args.per_string, args.max_fret, args.seed)
    print(format_playlist(playlist))
    print(f"\nSaved {args.out}/playlist.csv and playlist.txt")
    if args.simulate:
        print(f"Saved {simulate_take(Path(args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

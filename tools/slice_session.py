"""Cut a recorded take into labeled single notes (our usual dataset format).

    python tools/slice_session.py --session data/real_session1 --wav ~/Desktop/take1.wav

Reads <session>/playlist.csv (from record_session.py), finds every note in
the recording, pairs note k with play-list entry k, and writes
<session>/audio/000001.wav ..., labels.csv and metadata.json, so the same
extract.py / evaluate.py pipeline works on it unchanged.

It refuses to guess:
  * If the number of detected notes != the play list length, it switches to
    pitch alignment (below): extra events (handling noise, a string plucked
    twice) are dropped automatically and listed. Notes it can't find are
    listed as missing and left out. If too many are missing, nothing is
    written. (Matching purely by order would let one extra or missing note
    silently shift every label after it.)
  * Every note's pitch is measured and compared to its label. Mismatches
    are flagged in labels.csv and printed. The label is never changed.
  * It won't overwrite an already sliced session (e.g. the real-guitar exam)
    unless you pass --overwrite. Use --out to slice into a new folder.

Pitch alignment
---------------
When the count is off, we detect events more permissively (closer
together, so merged notes split), measure each event's pitch, and line up
the sequence of event pitches with the play list's pitches. Like a spell
checker lining up two words, it uses dynamic programming: every way to walk
through both lists in order has a cost, and we take the cheapest:
  * event matches play-list entry:  0 if the pitch agrees, small if it's off
                                    by exactly an octave (a common pitch-
                                    tracker error), large if it's another note
  * drop an event:                  cheap for quiet or unpitched events
                                    (noise), dearer for loud ones. Cheapest
                                    for a "repeat": an event with the same
                                    pitch as the one just before it. That's a
                                    re-pluck, or a knock while the note is
                                    still ringing (the pitch tracker hears the
                                    ringing string), so the first event wins.
  * a play-list entry has no event: most expensive (a missing note)
A note played wrong is still matched (and then flagged), never silently
swapped for something else.

Steps
-----
1. Load, mix to mono, resample to 22050 Hz (the training sample rate).
2. Onset detection. An "onset" is the moment a note starts. We use
   spectral flux (librosa's onset strength): how much the spectrum
   suddenly gains energy from one frame to the next. A pluck is a big jump.
   Picking then has two stages:
     a. find every local peak in onset strength (candidates);
     b. accept the strongest candidates first, skipping any within
        --min-gap seconds of one already accepted. A muting click or string
        squeak right before a pluck is weaker than the pluck, so the pluck wins.
   Finally, drop candidates whose note is much quieter than the loudest
   note (--quiet-db): stray noises, not plucks.
3. Cut each note from its onset to the next onset (at most --duration s),
   pad with silence to exactly --duration s, like the synth's notes.
4. Level: scale each note so its peak is -3 dBFS, the middle of the synth's
   range. The model was trained on absolute dB values; a quiet phone
   recording would otherwise look like nothing it has seen.
5. Pitch check with pYIN (probabilistic YIN, classic DSP, no learning).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import librosa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "synth"))
import tuning  # noqa: E402

SAMPLE_RATE = 22050
HOP = 256  # onset analysis hop: 11.6 ms resolution
TARGET_PEAK_DBFS = -3.0
PITCH_TOLERANCE_CENTS = 50  # more than half a semitone off = probably a different note
TUNING_WARN_CENTS = 20
ALIGN_MIN_GAP = 0.25  # seconds; permissive event detection used for alignment
MAX_MISSING_FRACTION = 0.1  # more missing notes than this = re-record

# alignment costs (see "Pitch alignment" above)
COST_OCTAVE = 0.3
COST_WRONG_NOTE = 1.5
COST_UNPITCHED_MATCH = 1.0
COST_DROP_BASE = 0.5  # + 0.5 * relative loudness (0 = quietest, 1 = loudest)
COST_DROP_REPEAT = 0.2  # same pitch as the previous event
COST_MISSING = 2.0


class SliceError(RuntimeError):
    pass


def load_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)  # stereo -> mono by averaging the channels
    if sr != sample_rate:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=sample_rate)
    return mono


def detect_onsets(
    y: np.ndarray, sr: int, min_gap: float = 0.8, quiet_db: float = 30.0
) -> tuple[np.ndarray, dict]:
    """Return onset sample positions (sorted) and diagnostics for plotting."""
    strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    # a. candidates: every local maximum that stands out a little
    # (onset_detect rescales the strength curve to 0..1 first, so delta is a
    # fraction of the strongest onset)
    cand = librosa.onset.onset_detect(
        onset_envelope=strength, sr=sr, hop_length=HOP, backtrack=False,
        wait=int(0.03 * sr / HOP), delta=0.05,
    )
    if len(cand) == 0:
        return np.array([], dtype=int), {"strength": strength, "dropped_quiet": 0, "levels_db": np.array([])}
    # b. strongest first, enforce a minimum gap
    gap = int(min_gap * sr / HOP)
    accepted: list[int] = []
    for c in sorted(cand, key=lambda c: -strength[c]):
        if all(abs(c - a) >= gap for a in accepted):
            accepted.append(c)
    accepted.sort()

    # Drop quiet events: peak level over the 250 ms after the onset, in dB
    # relative to the loudest event.
    win = int(0.25 * sr)
    starts = librosa.frames_to_samples(np.array(accepted, dtype=int), hop_length=HOP)
    levels = np.array([20 * np.log10(np.abs(y[s : s + win]).max() + 1e-12) for s in starts])
    keep = levels >= (levels.max() - quiet_db) if len(levels) else np.array([], bool)
    frames = np.array(accepted, dtype=int)[keep]
    if len(frames) == 0:
        return np.array([], dtype=int), {"strength": strength, "dropped_quiet": int((~keep).sum()), "levels_db": np.array([])}

    # Move each onset back to just before the energy starts rising
    # (backtracking), so the cut includes the whole attack.
    backtracked = librosa.onset.onset_backtrack(frames, strength)
    onsets = librosa.frames_to_samples(backtracked, hop_length=HOP)
    return onsets, {"strength": strength, "dropped_quiet": int((~keep).sum()), "levels_db": levels[keep]}


def estimate_pitch(note: np.ndarray, sr: int) -> float | None:
    """Median pYIN f0 over 0.05-0.8 s, or None if no frame looks pitched."""
    seg = note[int(0.05 * sr) : int(0.8 * sr)]
    f0, voiced, _ = librosa.pyin(seg, fmin=65.0, fmax=1100.0, sr=sr, frame_length=2048)
    f0 = f0[voiced & np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) else None


def match_cost(event_midi: float | None, label_midi: int) -> float:
    if event_midi is None:
        return COST_UNPITCHED_MATCH
    d = abs(event_midi - label_midi)
    if d <= 0.5:
        return 0.0
    if abs(d - 12) <= 0.5:
        return COST_OCTAVE
    return COST_WRONG_NOTE


def align_to_playlist(
    event_midis: list[float | None], event_loudness: list[float], playlist_midis: list[int]
) -> tuple[list[int | None], list[int]]:
    """Cheapest in-order alignment of detected events to play-list entries.

    Returns (match, dropped): match[i] = event index for play-list entry i
    (None = missing), dropped = event indices not used. event_loudness is
    0..1 relative to the loudest event.
    """
    n, m = len(playlist_midis), len(event_midis)
    INF = float("inf")
    drop_cost = []
    for j in range(m):
        prev, cur = (event_midis[j - 1] if j else None), event_midis[j]
        repeat = prev is not None and cur is not None and abs(cur - prev) <= 0.5
        drop_cost.append(COST_DROP_REPEAT if repeat else COST_DROP_BASE + 0.5 * event_loudness[j])
    # cost[i][j] = cheapest way to account for the first i entries and first j events
    cost = np.full((n + 1, m + 1), INF)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 = match, 2 = drop event, 3 = missing entry
    cost[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, how = INF, 0
            if i > 0 and j > 0:
                c = cost[i - 1, j - 1] + match_cost(event_midis[j - 1], playlist_midis[i - 1])
                if c < best:
                    best, how = c, 1
            if j > 0:
                c = cost[i, j - 1] + drop_cost[j - 1]
                if c < best:
                    best, how = c, 2
            if i > 0:
                c = cost[i - 1, j] + COST_MISSING
                if c < best:
                    best, how = c, 3
            cost[i, j], move[i, j] = best, how

    # walk back from the end to recover the choices
    match: list[int | None] = [None] * n
    dropped = []
    i, j = n, m
    while i > 0 or j > 0:
        how = move[i, j]
        if how == 1:
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif how == 2:
            dropped.append(j - 1)
            j -= 1
        else:
            i -= 1
    return match, sorted(dropped)


def cut_notes(
    y: np.ndarray, onsets: np.ndarray, sr: int, duration: float, ends: np.ndarray | None = None
) -> list[np.ndarray]:
    """Each note runs from its onset to the next onset (or `ends`), at most
    `duration` s, zero-padded to exactly `duration` s."""
    n = int(round(duration * sr))
    fade = int(0.01 * sr)
    notes = []
    for i, start in enumerate(onsets):
        nxt = ends[i] if ends is not None else (onsets[i + 1] if i + 1 < len(onsets) else len(y))
        end = min(start + n, nxt)
        note = np.zeros(n)
        seg = y[start:end].copy()
        if len(seg) > fade:  # fade the cut end so it doesn't click
            seg[-fade:] *= np.linspace(1.0, 0.0, fade)
        note[: len(seg)] = seg
        notes.append(note)
    return notes


def plot_slicing(y, sr, onsets, rows, path: Path, title: str) -> None:
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(max(12, len(y) / sr / 6), 3.5))
    ax.plot(t, y, lw=0.3, color="0.4")
    for i, s in enumerate(onsets):
        bad = rows is not None and i < len(rows) and rows[i]["flag"]
        ax.axvline(s / sr, color="red" if bad else "tab:blue", lw=0.8)
        ax.text(s / sr, 1.02, str(i + 1), transform=ax.get_xaxis_transform(), fontsize=6,
                ha="center", color="red" if bad else "tab:blue")
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_title(title, fontsize=9, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def slice_session(
    session: Path, wav: Path, duration: float = 2.0, min_gap: float = 0.8, quiet_db: float = 30.0,
    drop_flagged: bool = False, out: Path | None = None, overwrite: bool = False,
) -> list[dict]:
    session, wav = Path(session), Path(wav)
    out = Path(out) if out else session
    if (out / "labels.csv").exists() and not overwrite:
        raise SliceError(
            f"{out} already has a sliced dataset (labels.csv). Refusing to overwrite it; "
            "pass --overwrite, or --out to slice into a new folder."
        )
    out.mkdir(parents=True, exist_ok=True)
    with open(session / "playlist.csv", newline="") as f:
        playlist = list(csv.DictReader(f))

    y = load_mono(wav)
    onsets, diag = detect_onsets(y, SAMPLE_RATE, min_gap, quiet_db)
    alignment = {"used": False, "dropped_events": [], "missing_takes": []}
    entries = list(playlist)

    if len(onsets) != len(playlist):
        # Count is off: detect permissively, pitch every event, align by pitch.
        onsets, diag = detect_onsets(y, SAMPLE_RATE, ALIGN_MIN_GAP, quiet_db)
        segments = cut_notes(y, onsets, SAMPLE_RATE, duration)
        event_hz = [estimate_pitch(s, SAMPLE_RATE) for s in segments]
        event_midi = [None if hz is None else tuning.hz_to_midi(hz) for hz in event_hz]
        lv = diag["levels_db"]
        loud = (lv - lv.min()) / (lv.max() - lv.min() + 1e-9) if len(lv) else lv
        match, dropped = align_to_playlist(event_midi, list(loud), [int(e["midi"]) for e in playlist])

        missing = [int(playlist[i]["take"]) for i, j in enumerate(match) if j is None]
        alignment = {
            "used": True,
            "events_detected": int(len(onsets)),
            "dropped_events": [
                {"time_s": round(onsets[j] / SAMPLE_RATE, 3),
                 "pitch": "-" if event_midi[j] is None else tuning.midi_to_name(int(round(event_midi[j]))),
                 "level_db": round(float(lv[j] - lv.max()), 1)}
                for j in dropped
            ],
            "missing_takes": missing,
        }
        if len(missing) > MAX_MISSING_FRACTION * len(playlist):
            plot_slicing(y, SAMPLE_RATE, onsets, None, out / "slicing.png",
                         f"{wav.name}: {len(onsets)} events, {len(missing)} play-list notes not found")
            raise SliceError(
                f"Pitch alignment could not find {len(missing)} of {len(playlist)} notes "
                f"(takes {missing}). Nothing was written; see {out / 'slicing.png'}. Re-record the take."
            )
        # A kept note ends at the next kept or *pitched* event (e.g. a
        # re-pluck), but runs through unpitched noise events.
        kept = [(i, j) for i, j in enumerate(match) if j is not None]
        boundaries = sorted({j for _, j in kept} | {j for j in range(len(onsets)) if event_midi[j] is not None})
        ends = {}
        for j in (jj for _, jj in kept):
            later = [b for b in boundaries if b > j]
            ends[j] = onsets[later[0]] if later else len(y)
        entries = [playlist[i] for i, _ in kept]
        starts = np.array([onsets[j] for _, j in kept])
        cut_ends = np.array([ends[j] for _, j in kept])
    else:
        starts = onsets
        cut_ends = np.append(onsets[1:], len(y))

    notes = cut_notes(y, starts, SAMPLE_RATE, duration, ends=cut_ends)
    rows = []
    for k, (entry, note, start) in enumerate(zip(entries, notes, starts), start=1):
        midi = int(entry["midi"])
        f0 = estimate_pitch(note, SAMPLE_RATE)
        err = None if f0 is None else tuning.cents_between(f0, tuning.midi_to_hz(midi))
        peak = np.abs(note).max()
        gain_db = TARGET_PEAK_DBFS - 20 * np.log10(peak + 1e-12)
        rows.append({
            "file": f"audio/{k:06d}.wav",
            "string": int(entry["string"]),
            "fret": int(entry["fret"]),
            "midi": midi,
            "note": tuning.midi_to_name(midi),
            "freq_hz": f"{tuning.midi_to_hz(midi):.6g}",
            "take": int(entry["take"]),
            "onset_s": f"{start / SAMPLE_RATE:.3f}",
            "gain_db": f"{gain_db:.2f}",
            "est_freq_hz": "" if f0 is None else f"{f0:.2f}",
            "pitch_error_cents": "" if err is None else f"{err:.1f}",
            "_note": note * 10 ** (gain_db / 20),
            "_err": err,
        })

    # A guitar tuned a bit flat shifts *every* note the same way; judge each
    # note relative to that common offset, and report the offset separately.
    errors = [r["_err"] for r in rows if r["_err"] is not None]
    offset = float(np.median(errors)) if errors else 0.0
    for r in rows:
        if r["_err"] is None:
            r["flag"] = "no pitch detected"
        elif abs(r["_err"] - offset) > PITCH_TOLERANCE_CENTS:
            semis = (r["_err"] - offset) / 100
            r["flag"] = f"pitch off by {semis:+.1f} semitones"
        else:
            r["flag"] = ""

    plot_slicing(y, SAMPLE_RATE, starts, rows, out / "slicing.png",
                 f"{wav.name}: {len(starts)} notes (red = pitch flag)"
                 + (f", {len(alignment['dropped_events'])} events dropped by pitch alignment" if alignment["used"] else ""))

    kept = [r for r in rows if not (drop_flagged and r["flag"])]
    audio_dir = out / "audio"
    if audio_dir.exists():
        for old in audio_dir.glob("*.wav"):
            old.unlink()
    audio_dir.mkdir(exist_ok=True)
    columns = ["file", "string", "fret", "midi", "note", "freq_hz", "take", "onset_s", "gain_db",
               "est_freq_hz", "pitch_error_cents", "flag"]
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for k, r in enumerate(kept, start=1):
            r["file"] = f"audio/{k:06d}.wav"  # renumber if flagged notes were dropped
            pcm = np.round(np.clip(r["_note"], -1, 1) * 32767).astype(np.int16)
            sf.write(str(out / r["file"]), pcm, SAMPLE_RATE, subtype="PCM_16")
            w.writerow(r)

    meta = {
        "kind": "real_session",
        "source_wav": str(wav),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate": SAMPLE_RATE,
        "duration_s": duration,
        "count": len(kept),
        "max_fret": int(json.loads((session / "session_meta.json").read_text())["max_fret"])
        if (session / "session_meta.json").exists() else 19,
        "slicing": {"min_gap_s": min_gap, "quiet_db": quiet_db, "dropped_quiet_events": diag["dropped_quiet"]},
        "level": f"each note scaled to peak {TARGET_PEAK_DBFS} dBFS",
        "tuning_offset_cents": round(offset, 1),
        "flagged": sum(bool(r["flag"]) for r in rows),
        "dropped_flagged": drop_flagged,
        "pitch_alignment": alignment,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Slice a recorded take into a labeled dataset.")
    p.add_argument("--session", required=True, help="folder with playlist.csv")
    p.add_argument("--wav", required=True, help="the recorded take")
    p.add_argument("--duration", type=float, default=2.0, help="note length to write (match training)")
    p.add_argument("--min-gap", type=float, default=0.8, help="minimum seconds between two notes")
    p.add_argument("--quiet-db", type=float, default=30.0, help="ignore events this many dB below the loudest")
    p.add_argument("--drop-flagged", action="store_true", help="leave pitch-flagged notes out of the dataset")
    p.add_argument("--out", help="write the dataset here instead of into --session")
    p.add_argument("--overwrite", action="store_true", help="replace an existing sliced dataset")
    args = p.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.session)
    try:
        rows = slice_session(Path(args.session), Path(args.wav), args.duration, args.min_gap,
                             args.quiet_db, args.drop_flagged, out, args.overwrite)
    except SliceError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    meta = json.loads((out / "metadata.json").read_text())
    print(f"Sliced {len(rows)} notes -> {out}/audio, labels.csv (plot: {out}/slicing.png)")
    al = meta["pitch_alignment"]
    if al["used"]:
        print(f"Note count didn't match the play list, so notes were matched by pitch "
              f"({al['events_detected']} events detected).")
        for d in al["dropped_events"]:
            print(f"  DROPPED event at {d['time_s']:.2f}s: pitch {d['pitch']}, {d['level_db']:+.0f} dB vs loudest")
        if al["missing_takes"]:
            print(f"  MISSING (no matching note found, left out): takes {al['missing_takes']}")
    offset = meta["tuning_offset_cents"]
    if abs(offset) > TUNING_WARN_CENTS:
        print(f"WARNING: the guitar seems out of tune by {offset:+.0f} cents overall (A440 reference).")
    else:
        print(f"Tuning: {offset:+.1f} cents overall vs A440 (fine).")
    flagged = [r for r in rows if r["flag"]]
    for r in flagged:
        print(f"  FLAG take {r['take']:2d}: s{r['string']} f{r['fret']} ({r['note']}): {r['flag']} "
              f"(measured {r['est_freq_hz'] or '-'} Hz)")
    if flagged:
        print(f"{len(flagged)} note(s) flagged. Check them (wrong fret? open string ringing?). Labels were NOT changed"
              + ("; flagged notes were dropped." if args.drop_flagged else "; use --drop-flagged to exclude them."))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Turn a dataset's WAV files into spectrograms: the model's "eyes".

    python features/extract.py --data data/notes_v1

Why the model doesn't get raw audio
-----------------------------------
A 2 s note at 22050 Hz is 44,100 numbers that wiggle up and down. Pitch and
timbre are in there, but only as *patterns over time* (how often the wave
repeats, how its shape changes). A spectrogram does that pattern-finding up
front with fixed, well-understood math. It shows how much energy each
frequency has at each moment, as a 2-D image: frequency on one axis, time on
the other. A small model can then learn from the image instead of
rediscovering Fourier analysis from scratch.

Why a Constant-Q transform (CQT) instead of a regular FFT spectrogram
--------------------------------------------------------------------
An FFT spectrogram spaces its frequency bins *evenly in Hz*, e.g. every
10.8 Hz. But musical pitch is *logarithmic*: each semitone (each fret)
multiplies frequency by 2^(1/12) ~ 1.0595. So:
  * low E (82 Hz) -> F (87 Hz) is a 5 Hz step: less than one FFT bin, the
    two notes blur together;
  * high E fret 19 (988 Hz) -> fret 20 (1047 Hz) is a 59 Hz step: ~5 bins.
In an FFT image the same musical step looks different depending on where you
are on the neck.

The CQT spaces its bins *geometrically*: every bin is a fixed ratio above the
previous one (here 2^(1/36), i.e. 3 bins per semitone). So every semitone,
every fret, is exactly 3 bins apart everywhere, and a note's harmonic
pattern (f0, 2*f0, 3*f0, ...) has the *same shape* at every pitch, just
shifted up or down. That shift-invariance makes life much easier for a
convolutional model later: "moving up one fret" = "move the picture up 3
rows". The "constant Q" is the ratio frequency / bandwidth, which stays
the same for every bin. Low bins get long analysis windows (fine pitch
resolution, where it's needed); high bins get short ones (fine time
resolution).

Why 3 bins per semitone (36 per octave)? With 1 bin per semitone, a note
detuned by a few cents looks identical to an in-tune one, and a note 50 cents
sharp falls *between* bins. 3 bins (33 cents apart) let the image show where
inside the semitone the energy sits.

Why dB? Loudness perception, and the dynamic range of a decaying note, are
logarithmic. In linear amplitude the attack would be a bright spot and
everything else nearly black. In dB the quiet harmonics and the decaying tail
(which carry timbre, i.e. string identity) are visible too.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

import librosa

# Make the sibling `synth/` folder importable (for tuning helpers), whether
# this file is run as a script or imported by tests/tools.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "synth"))
import tuning  # noqa: E402

FEATURES_VERSION = "0.1.0"

# --- CQT parameters ---------------------------------------------------------
FMIN_NOTE = "D2"  # 73.4 Hz: a whole tone below low E, so E2 isn't on the edge
FMIN_MIDI = 38  # MIDI number of D2
N_OCTAVES = 7  # D2 .. D9 (~9.4 kHz): covers B5 (fret 19, high E) + many harmonics
BINS_PER_OCTAVE = 36  # 3 per semitone
N_BINS = N_OCTAVES * BINS_PER_OCTAVE  # 252
BINS_PER_SEMITONE = BINS_PER_OCTAVE // 12  # 3
HOP_LENGTH = 512  # samples between frames: 512 / 22050 Hz = 23.2 ms per frame

# --- dB conversion ----------------------------------------------------------
# ref=1.0 keeps *absolute* levels: a soft pick stays quieter than a hard one
# (we normalize later, from training statistics, not per note).
# amin is the floor: 1e-5 -> -100 dB, well below the synth's noise floor, so
# almost nothing gets clamped. top_db=None: no extra per-note clamping.
DB_REF = 1.0
DB_AMIN = 1e-5

BATCH_SIZE = 64  # librosa computes a stack of equal-length notes ~9x faster


def fmin_hz() -> float:
    return float(librosa.note_to_hz(FMIN_NOTE))


def bin_frequencies() -> np.ndarray:
    """Center frequency (Hz) of every CQT bin."""
    return librosa.cqt_frequencies(N_BINS, fmin=fmin_hz(), bins_per_octave=BINS_PER_OCTAVE)


def midi_to_bin(midi: float) -> float:
    """CQT row index whose center is this (possibly fractional) MIDI pitch."""
    return (midi - FMIN_MIDI) * BINS_PER_SEMITONE


def bin_to_midi(bin_index: float) -> float:
    return FMIN_MIDI + bin_index / BINS_PER_SEMITONE


def cqt_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Log-magnitude CQT of one note (n_samples,) or a batch (batch, n_samples).

    Returns float32 with shape (..., N_BINS, n_frames). The CQT gives complex
    numbers (amplitude + phase); we keep only the amplitude (np.abs). Phase
    mostly encodes *when* each wave crest happens, which says little about
    which string/fret it is.
    """
    spec = librosa.cqt(
        np.asarray(audio, dtype=np.float64),
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        fmin=fmin_hz(),
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    db = librosa.amplitude_to_db(np.abs(spec), ref=DB_REF, amin=DB_AMIN, top_db=None)
    return db.astype(np.float32)


def n_frames_for(n_samples: int) -> int:
    # librosa centers frames on hop multiples (center=True): frames at
    # 0, 512, 1024, ... up to the end of the signal.
    return 1 + n_samples // HOP_LENGTH


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_labels(data_dir: Path) -> list[dict]:
    with open(Path(data_dir) / "labels.csv", newline="") as f:
        return list(csv.DictReader(f))


def extract_features(data_dir: Path, batch_size: int = BATCH_SIZE, progress: bool = True) -> dict:
    """Write <data_dir>/features.npy and features_meta.json. Returns the meta.

    Row i of features.npy is the spectrogram of the file on row i of
    labels.csv: same order, no shuffling, no filtering. Everything downstream
    (splits, training) refers to notes by that row index.
    """
    data_dir = Path(data_dir)
    rows = read_labels(data_dir)
    synth_meta = json.loads((data_dir / "metadata.json").read_text())
    sample_rate = int(synth_meta["sample_rate"])
    n_samples = int(round(synth_meta["duration_s"] * sample_rate))
    n_frames = n_frames_for(n_samples)
    shape = (len(rows), N_BINS, n_frames)

    # Write straight to disk through a memory map, so a big dataset never has
    # to fit in RAM. A temporary name + rename at the end means a crash can't
    # leave a half-written features.npy that looks complete.
    final_path = data_dir / "features.npy"
    tmp_path = data_dir / "features.npy.partial"
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=shape)

    start = time.perf_counter()
    batches = range(0, len(rows), batch_size)
    for b0 in tqdm(batches, desc="CQT", unit="batch", disable=not progress):
        batch_rows = rows[b0 : b0 + batch_size]
        audio = np.empty((len(batch_rows), n_samples), dtype=np.float64)
        for j, row in enumerate(batch_rows):
            y, sr = sf.read(str(data_dir / row["file"]), dtype="float64")
            if sr != sample_rate or y.ndim != 1 or len(y) != n_samples:
                raise ValueError(
                    f"{row['file']}: expected mono {n_samples} samples @ {sample_rate} Hz, "
                    f"got shape {y.shape} @ {sr} Hz"
                )
            audio[j] = y
        # float16 keeps 3 significant digits: ~0.03-0.06 dB of precision in
        # the -100..+20 dB range. Far finer than anything audible, and it
        # halves the file size compared to float32.
        out[b0 : b0 + len(batch_rows)] = cqt_db(audio, sample_rate).astype(np.float16)
    out.flush()
    del out
    tmp_path.replace(final_path)
    elapsed = time.perf_counter() - start

    meta = {
        "features_version": FEATURES_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": "features.npy",
        "shape": list(shape),
        "axes": ["note (row of labels.csv)", "cqt_bin (low -> high freq)", "frame (time)"],
        "dtype": "float16",
        "units": "dB re amplitude 1.0 (librosa.amplitude_to_db, ref=1.0)",
        "transform": "librosa.cqt magnitude",
        "sample_rate": sample_rate,
        "duration_s": n_samples / sample_rate,
        "hop_length": HOP_LENGTH,
        "frame_seconds": HOP_LENGTH / sample_rate,
        "fmin_note": FMIN_NOTE,
        "fmin_midi": FMIN_MIDI,
        "fmin_hz": fmin_hz(),
        "n_octaves": N_OCTAVES,
        "bins_per_octave": BINS_PER_OCTAVE,
        "bins_per_semitone": BINS_PER_SEMITONE,
        "n_bins": N_BINS,
        "fmax_hz": float(bin_frequencies()[-1]),
        "n_frames": n_frames,
        "db_ref": DB_REF,
        "db_amin": DB_AMIN,
        "db_floor": float(20 * np.log10(DB_AMIN / DB_REF)),
        "top_db": None,
        "bin_to_midi": "midi = fmin_midi + bin / bins_per_semitone",
        # Fingerprint of the labels the rows were aligned to. If labels.csv
        # ever changes, this no longer matches and the features are stale.
        "labels_sha256": sha256_of(data_dir / "labels.csv"),
        "n_rows": len(rows),
        "first_file": rows[0]["file"] if rows else None,
        "last_file": rows[-1]["file"] if rows else None,
        "source_dataset": {
            "synth_version": synth_meta.get("synth_version"),
            "seed": synth_meta.get("seed"),
            "count": synth_meta.get("count"),
        },
        "extraction_seconds": round(elapsed, 3),
        "versions": {"librosa": librosa.__version__, "numpy": np.__version__},
    }
    (data_dir / "features_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_features(data_dir: Path, check: bool = True) -> tuple[np.ndarray, dict]:
    """Open features.npy read-only (memory-mapped) plus its meta.

    With check=True, refuses features that no longer match labels.csv.
    """
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "features_meta.json").read_text())
    if check and meta["labels_sha256"] != sha256_of(data_dir / "labels.csv"):
        raise RuntimeError(
            f"{data_dir}/features.npy is stale: labels.csv changed since extraction. "
            "Re-run features/extract.py."
        )
    return np.load(data_dir / "features.npy", mmap_mode="r"), meta


def note_ticks(step: str = "E_and_C") -> tuple[list[int], list[str]]:
    """Y-axis tick positions (bin indices) labeled with note names.

    Default: every E (the open-string note) and every C (octave landmark).
    """
    keep = {"E_and_C": {"E", "C"}, "E": {"E"}}[step]
    positions, names = [], []
    for b in range(0, N_BINS, BINS_PER_SEMITONE):
        midi = int(round(bin_to_midi(b)))
        name = tuning.midi_to_name(midi)
        if name.rstrip("0123456789") in keep:
            positions.append(b)
            names.append(name)
    return positions, names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute CQT spectrogram features for a dataset.")
    p.add_argument("--data", required=True, help="dataset folder, e.g. data/notes_v1")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    meta = extract_features(Path(args.data), batch_size=args.batch_size, progress=not args.quiet)
    size_mb = (Path(args.data) / "features.npy").stat().st_size / 1e6
    secs = meta["extraction_seconds"]
    print(
        f"Wrote {args.data}/features.npy  shape={tuple(meta['shape'])} float16  "
        f"{size_mb:.1f} MB  in {secs:.2f}s ({1000 * secs / max(1, meta['n_rows']):.2f} ms/note)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

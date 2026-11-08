"""The CQT spectrogram: how Electro turns a note's audio into a picture.

This is the one definition of "what a note looks like" that both projects
share. Electro Synth's tools (show.py, compare_real.py) draw it; Electro CNN's
features/extract.py computes it for every note and the model learns from it.
Keeping it here, in the installed synth package, means the two can't drift
apart: there is exactly one copy.

What a CQT is
-------------
A constant-Q transform is a spectrogram whose frequency bins are spaced like
piano keys (equal steps in *pitch*, not in Hz). Here: 3 bins per semitone,
36 per octave, 7 octaves from D2 (73.4 Hz). So bin = 3 * (midi - 38): moving
a note up one semitone moves its whole picture up exactly 3 rows. A plain FFT
spreads low notes over a few bins and high notes over hundreds; the CQT gives
every note the same shape, which is what a convolutional model wants.
"""

from __future__ import annotations

import numpy as np

import librosa

import tuning

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
# (the model normalizes later, from training statistics, not per note).
# amin is the floor: 1e-5 -> -100 dB, well below the synth's noise floor, so
# almost nothing gets clamped. top_db=None: no extra per-note clamping.
DB_REF = 1.0
DB_AMIN = 1e-5


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

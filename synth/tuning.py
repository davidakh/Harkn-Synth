"""Guitar tuning: (string, fret) -> MIDI note number -> frequency in Hz.

Concepts
--------
* MIDI note numbers are a linear "semitone index": every +1 is one semitone
  (one fret) higher. That is why a fretted note is simply `open_midi + fret`.
* Equal temperament: each semitone multiplies frequency by 2**(1/12), so 12
  semitones (one octave) exactly doubles the frequency. Anchored at A4 = MIDI 69
  = 440 Hz, the frequency of any MIDI note m is 440 * 2**((m - 69) / 12).
* Cents measure small pitch differences: 100 cents = 1 semitone, so 1200 cents
  = 1 octave. Humans notice roughly 5-10 cents of mistuning on sustained notes.

String numbering follows guitar convention (and our labels):
string 1 = high E (thinnest) ... string 6 = low E (thickest).
"""

from __future__ import annotations

import math

# MIDI note of each open string in standard tuning (E2 A2 D3 G3 B3 E4).
OPEN_STRING_MIDI: dict[int, int] = {
    6: 40,  # E2  (low E, wound)
    5: 45,  # A2  (wound)
    4: 50,  # D3  (wound)
    3: 55,  # G3  (wound on acoustic sets, plain on many electric sets)
    2: 59,  # B3  (plain)
    1: 64,  # E4  (high E, plain)
}

STRINGS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

A4_MIDI = 69
A4_HZ = 440.0


def string_fret_to_midi(string: int, fret: int) -> int:
    """MIDI note for a string (1-6) pressed at a fret (0 = open)."""
    if string not in OPEN_STRING_MIDI:
        raise ValueError(f"string must be 1..6, got {string}")
    if fret < 0:
        raise ValueError(f"fret must be >= 0, got {fret}")
    return OPEN_STRING_MIDI[string] + fret


def midi_to_hz(midi: float) -> float:
    """Equal-tempered frequency (A4 = 440 Hz). Accepts fractional MIDI."""
    return A4_HZ * 2.0 ** ((midi - A4_MIDI) / 12.0)


def hz_to_midi(hz: float) -> float:
    """Inverse of midi_to_hz; returns a fractional MIDI number."""
    return A4_MIDI + 12.0 * math.log2(hz / A4_HZ)


def string_fret_to_hz(string: int, fret: int) -> float:
    return midi_to_hz(string_fret_to_midi(string, fret))


def cents_between(f_measured: float, f_target: float) -> float:
    """Signed pitch error in cents (+ = measured is sharp)."""
    return 1200.0 * math.log2(f_measured / f_target)


def detune_hz(hz: float, cents: float) -> float:
    """Shift a frequency by a number of cents."""
    return hz * 2.0 ** (cents / 1200.0)


def midi_to_name(midi: int) -> str:
    """e.g. 40 -> 'E2', 69 -> 'A4' (MIDI 60 = C4 convention)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def all_positions(max_fret: int) -> list[tuple[int, int]]:
    """Every (string, fret) pair on the neck, frets 0..max_fret inclusive."""
    return [(s, f) for s in STRINGS for f in range(max_fret + 1)]

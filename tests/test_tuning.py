import pytest

import tuning


def test_open_strings_standard_tuning():
    assert [tuning.string_fret_to_midi(s, 0) for s in (6, 5, 4, 3, 2, 1)] == [40, 45, 50, 55, 59, 64]
    assert [tuning.midi_to_name(tuning.string_fret_to_midi(s, 0)) for s in (6, 5, 4, 3, 2, 1)] == [
        "E2", "A2", "D3", "G3", "B3", "E4"
    ]


def test_fret_adds_semitones():
    assert tuning.string_fret_to_midi(6, 5) == tuning.string_fret_to_midi(5, 0)  # A2
    assert tuning.string_fret_to_midi(3, 4) == tuning.string_fret_to_midi(2, 0)  # B3 (G string is the odd one)
    assert tuning.string_fret_to_midi(1, 12) == 76  # E5


def test_frequencies():
    assert tuning.string_fret_to_hz(1, 5) == pytest.approx(440.0)  # A4
    assert tuning.string_fret_to_hz(6, 0) == pytest.approx(82.4069, abs=1e-3)
    assert tuning.string_fret_to_hz(6, 12) == pytest.approx(2 * tuning.string_fret_to_hz(6, 0))
    assert tuning.hz_to_midi(440.0) == pytest.approx(69.0)


def test_cents():
    assert tuning.cents_between(880.0, 440.0) == pytest.approx(1200.0)
    assert tuning.cents_between(tuning.detune_hz(440.0, 3.0), 440.0) == pytest.approx(3.0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        tuning.string_fret_to_midi(7, 0)
    with pytest.raises(ValueError):
        tuning.string_fret_to_midi(1, -1)


def test_all_positions():
    positions = tuning.all_positions(19)
    assert len(positions) == 6 * 20
    assert len(set(positions)) == len(positions)

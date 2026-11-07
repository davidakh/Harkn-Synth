"""Pitch accuracy: every string/fret must sound at the right frequency.

This is the most important property of the dataset. A label that says
"A2, 110 Hz" on audio that is actually 112 Hz teaches the model the wrong
thing, and it would never tell us.
"""

import numpy as np
import pytest

import electro_synth as es
import tuning
from pitch_utils import estimate_f0

SR = 22050
DURATION = 2.0
POSITIONS = tuning.all_positions(19)


def _id(pos):
    s, f = pos
    return f"s{s}f{f:02d}"


@pytest.mark.parametrize("position", POSITIONS, ids=_id)
def test_pitch_default_params_within_5_cents(position):
    """No detune: the synth's own tuning error must stay within +-5 cents."""
    string, fret = position
    params = es.default_params(string, fret)
    audio, _ = es.render_note(params, SR, DURATION, es.sample_rng(0, 1))
    target = tuning.string_fret_to_hz(string, fret)
    error = tuning.cents_between(estimate_f0(audio, SR), target)
    assert abs(error) <= 5.0, f"{_id(position)}: {error:+.2f} cents"


@pytest.mark.parametrize("position", POSITIONS, ids=_id)
def test_pitch_random_params_within_10_cents(position):
    """Random playing params, including up to +-3 cents detune, vs the nominal
    label frequency: must be within +-10 cents."""
    string, fret = position
    rng = es.sample_rng(123, string * 100 + fret)
    params = es.random_params(rng, string, fret)
    audio, phys = es.render_note(params, SR, DURATION, rng)
    error = tuning.cents_between(estimate_f0(audio, SR), phys["nominal_hz"])
    assert abs(error) <= 10.0, f"{_id(position)}: {error:+.2f} cents ({params})"


@pytest.mark.parametrize("sample_rate", [16000, 44100])
def test_pitch_other_sample_rates(sample_rate):
    for string, fret in [(6, 0), (3, 7), (1, 19)]:
        audio, _ = es.render_note(es.default_params(string, fret), sample_rate, 1.5, es.sample_rng(0, 1))
        error = tuning.cents_between(
            estimate_f0(audio, sample_rate), tuning.string_fret_to_hz(string, fret)
        )
        assert abs(error) <= 5.0, f"sr={sample_rate} s{string}f{fret}: {error:+.2f} cents"


def test_estimator_sanity_on_pure_tone():
    """Check the checker: a pure sine must be measured almost exactly."""
    t = np.arange(int(1.5 * SR)) / SR
    for hz in (82.41, 196.0, 987.77):
        assert abs(tuning.cents_between(estimate_f0(np.sin(2 * np.pi * hz * t), SR), hz)) < 0.5

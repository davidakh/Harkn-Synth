"""The shared CQT definition (synth/spectrogram.py) and the figures drawn with it."""

import numpy as np
import pytest

import electro_synth as es
import spectrogram
import tuning


def test_strongest_bin_is_the_labeled_pitch():
    """Every position on the neck (random playing params): averaged over the
    first half-second, the strongest CQT bin is exactly the bin of the labeled
    note. (Our pluck model makes the fundamental the loudest partial; bins are
    33 cents apart and detune is at most 3 cents, so it must be the exact bin.)"""
    for string, fret in tuning.all_positions(19):
        rng = es.sample_rng(77, string * 100 + fret)
        params = es.random_params(rng, string, fret)
        audio, _ = es.render_note(params, 22050, 2.0, rng)
        spec = spectrogram.cqt_db(audio, 22050)
        strongest = int(np.argmax(spec[:, 2:22].mean(axis=1)))
        midi = tuning.string_fret_to_midi(string, fret)
        assert strongest == spectrogram.midi_to_bin(midi), (
            f"s{string}f{fret}: strongest bin is "
            f"{tuning.midi_to_name(round(spectrogram.bin_to_midi(strongest)))}, "
            f"expected {tuning.midi_to_name(midi)}"
        )


def test_bin_note_mapping():
    freqs = spectrogram.bin_frequencies()
    for midi in (40, 64, 83):  # E2, E4, B5
        assert freqs[int(spectrogram.midi_to_bin(midi))] == pytest.approx(tuning.midi_to_hz(midi), rel=1e-9)
    ticks, names = spectrogram.note_ticks()
    assert names[0] == "E2" and "E4" in names and "C5" in names


def test_shape():
    spec = spectrogram.cqt_db(np.zeros(22050), 22050)
    assert spec.shape == (252, spectrogram.n_frames_for(22050)) and spec.dtype == np.float32


def test_show_writes_pngs(tmp_path):
    import show

    data = tmp_path / "ds"
    es.generate_dataset(count=60, out_dir=data, seed=5, duration=1.0, max_fret=4, progress=False)
    show.main(["--data", str(data), "--out", str(tmp_path / "fig")])
    for name in ("same_pitch_E4.png", "open_strings.png"):
        png = tmp_path / "fig" / name
        assert png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

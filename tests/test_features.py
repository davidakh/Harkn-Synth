"""Step 2: spectrogram features, splits and normalization stats."""

import csv
import json
import shutil
from collections import Counter

import numpy as np
import pytest

import electro_synth as es
import extract
import split
import tuning

N_POSITIONS = 6 * 5  # max_fret=4 keeps the fixture small but still >= 3 notes/position
COUNT = 300
DURATION = 1.0


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A small dataset with features and splits, built once for this file."""
    d = tmp_path_factory.mktemp("ds")
    es.generate_dataset(count=COUNT, out_dir=d, seed=5, duration=DURATION, max_fret=4, progress=False)
    extract.extract_features(d, progress=False)
    split.make_splits(d, seed=42)
    return d


def _labels(d):
    return extract.read_labels(d)


def _split_rows(d, name):
    with open(d / "splits" / f"{name}.csv") as f:
        return list(csv.DictReader(f))


# --- features.npy ------------------------------------------------------------
def test_shape_dtype_and_meta(dataset):
    features, meta = extract.load_features(dataset)
    n_frames = 1 + int(DURATION * 22050) // extract.HOP_LENGTH
    assert features.shape == (COUNT, 252, n_frames)
    assert features.dtype == np.float16
    assert meta["shape"] == list(features.shape)
    assert meta["n_bins"] == 252 and meta["bins_per_octave"] == 36 and meta["hop_length"] == 512
    assert meta["fmin_hz"] == pytest.approx(73.416, abs=1e-2)
    assert np.isfinite(np.asarray(features, dtype=np.float32)).all()


def test_rows_line_up_with_labels(dataset):
    """Row i of features.npy must be the spectrogram of labels.csv row i's file.
    Recompute every row independently, one file at a time, and compare."""
    import soundfile as sf

    features, _ = extract.load_features(dataset)
    rows = _labels(dataset)
    recomputed = [extract.cqt_db(sf.read(str(dataset / r["file"]))[0], 22050) for r in rows]
    for i, spec in enumerate(recomputed):
        # float16 storage rounds to ~0.03-0.06 dB at these magnitudes
        np.testing.assert_allclose(features[i].astype(np.float32), spec, atol=0.07, err_msg=f"row {i}")
    # ...and the check is sensitive: a neighbouring row does NOT match
    assert np.abs(features[1].astype(np.float32) - recomputed[0]).max() > 5.0


def test_batch_size_does_not_change_features(dataset, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy, ignore=shutil.ignore_patterns("features*", "splits"))
    extract.extract_features(copy, batch_size=7, progress=False)  # 7 doesn't divide 300
    a, _ = extract.load_features(dataset)
    b, _ = extract.load_features(copy)
    np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), atol=0.07)


def test_stale_features_are_refused(dataset, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(dataset, copy)
    labels = (copy / "labels.csv").read_text().splitlines()
    labels[1], labels[2] = labels[2], labels[1]  # reorder two rows
    (copy / "labels.csv").write_text("\n".join(labels) + "\n")
    with pytest.raises(RuntimeError, match="stale"):
        extract.load_features(copy)


def test_strongest_bin_is_the_labeled_pitch():
    """Every position on the neck (random playing params): averaged over the
    first half-second, the strongest CQT bin is exactly the bin of the labeled
    note. (Our pluck model makes the fundamental the loudest partial; bins are
    33 cents apart and detune is at most 3 cents, so it must be the exact bin.)"""
    for string, fret in tuning.all_positions(19):
        rng = es.sample_rng(77, string * 100 + fret)
        params = es.random_params(rng, string, fret)
        audio, _ = es.render_note(params, 22050, 2.0, rng)
        spec = extract.cqt_db(audio, 22050)
        strongest = int(np.argmax(spec[:, 2:22].mean(axis=1)))
        midi = tuning.string_fret_to_midi(string, fret)
        assert strongest == extract.midi_to_bin(midi), (
            f"s{string}f{fret}: strongest bin is "
            f"{tuning.midi_to_name(round(extract.bin_to_midi(strongest)))}, "
            f"expected {tuning.midi_to_name(midi)}"
        )


def test_bin_note_mapping():
    freqs = extract.bin_frequencies()
    for midi in (40, 64, 83):  # E2, E4, B5
        assert freqs[int(extract.midi_to_bin(midi))] == pytest.approx(tuning.midi_to_hz(midi), rel=1e-9)
    ticks, names = extract.note_ticks()
    assert names[0] == "E2" and "E4" in names and "C5" in names


# --- splits ------------------------------------------------------------------
def test_splits_are_disjoint_and_cover_everything(dataset):
    sets = {name: {int(r["row"]) for r in _split_rows(dataset, name)} for name in split.SPLITS}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sets["train"] | sets["val"] | sets["test"] == set(range(COUNT))
    assert len(sets["train"]) == pytest.approx(0.8 * COUNT, abs=N_POSITIONS)


def test_every_position_in_every_split(dataset):
    all_positions = {(r["string"], r["fret"]) for r in _labels(dataset)}
    assert len(all_positions) == N_POSITIONS
    for name in split.SPLITS:
        positions = Counter((r["string"], r["fret"]) for r in _split_rows(dataset, name))
        assert set(positions) == all_positions, f"{name} is missing positions"


def test_split_rows_point_at_the_right_labels(dataset):
    labels = _labels(dataset)
    for name in split.SPLITS:
        for r in _split_rows(dataset, name):
            original = labels[int(r["row"])]
            assert {k: r[k] for k in original} == original


def test_split_is_deterministic():
    rows = [{"string": str(s), "fret": str(f)} for s, f in es.plan_positions(600, 19, seed=1)]
    assert split.stratified_split(rows, seed=3) == split.stratified_split(rows, seed=3)
    assert split.stratified_split(rows, seed=3) != split.stratified_split(rows, seed=4)


def test_too_few_notes_per_position_is_an_error():
    rows = [{"string": "1", "fret": "0"}] * 2
    with pytest.raises(ValueError, match="at least 3"):
        split.stratified_split(rows)


# --- normalization stats -----------------------------------------------------
def test_norm_stats_come_from_train_only(dataset):
    features, _ = extract.load_features(dataset)
    stats = json.loads((dataset / "splits" / "norm_stats.json").read_text())
    train = [int(r["row"]) for r in _split_rows(dataset, "train")]
    x_train = np.asarray(features[train], dtype=np.float64)
    assert stats["computed_from"] == "train" and stats["n_notes"] == len(train)
    assert stats["mean"] == pytest.approx(x_train.mean(), abs=1e-6)
    assert stats["std"] == pytest.approx(x_train.std(), abs=1e-6)
    np.testing.assert_allclose(stats["mean_per_bin"], x_train.mean(axis=(0, 2)), atol=1e-5)
    np.testing.assert_allclose(stats["std_per_bin"], x_train.std(axis=(0, 2)), atol=1e-5)
    # and they differ from stats over *all* notes (i.e. val/test weren't used)
    x_all = np.asarray(features, dtype=np.float64)
    assert abs(stats["mean"] - x_all.mean()) > 1e-6


# --- figures -------------------------------------------------------------------
def test_show_writes_pngs(dataset, tmp_path):
    import show

    show.main(["--data", str(dataset), "--out", str(tmp_path)])
    for name in ("same_pitch_E4.png", "open_strings.png"):
        png = tmp_path / name
        assert png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

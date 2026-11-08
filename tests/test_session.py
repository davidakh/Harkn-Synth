"""Step 4: play list, take slicing, and the synth's onset jitter."""

import csv
import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

import electro_synth as es
import record_session as rec
import slice_session as sl


# --- play list -----------------------------------------------------------------
def test_playlist_is_balanced_spread_and_shuffled():
    pl = rec.make_playlist(per_string=10, max_fret=19, seed=42)
    assert len(pl) == 60 and len(set(pl)) == 60
    assert set(Counter(s for s, _ in pl).values()) == {10}
    for string in range(1, 7):
        frets = sorted(f for s, f in pl if s == string)
        # one fret from each pair 0-1, 2-3, ..., 18-19
        assert [f // 2 for f in frets] == list(range(10))
    assert pl != sorted(pl)  # shuffled
    assert pl == rec.make_playlist(10, 19, 42)  # deterministic


def test_tab_shows_each_fret_on_its_string():
    block = rec.tab_block([(1, 5, 7), (2, 1, 12)])
    lines = {line[0]: line for line in block[1:]}
    assert "7" in lines["A"] and "12" in lines["e"]
    assert "7" not in lines["e"] and "12" not in lines["A"]


# --- slicing a (simulated) take --------------------------------------------------
@pytest.fixture
def session(tmp_path):
    d = tmp_path / "session"
    rec.write_session(d, per_string=2, max_fret=19, seed=3)  # 12 notes: quick
    rec.simulate_take(d, seed=1)
    return d


def test_slices_every_note_in_order(session):
    rows = sl.slice_session(session, session / "simulated_take.wav")
    playlist = list(csv.DictReader(open(session / "playlist.csv")))
    assert len(rows) == len(playlist) == 12
    for row, entry in zip(rows, playlist):
        assert (row["string"], row["fret"]) == (int(entry["string"]), int(entry["fret"]))
        assert row["flag"] == ""
    labels = list(csv.DictReader(open(session / "labels.csv")))
    assert len(labels) == 12
    for r in labels:
        info = sf.info(str(session / r["file"]))
        assert info.samplerate == 22050 and info.frames == 2 * 22050 and info.channels == 1


def test_too_many_missing_notes_writes_nothing(session):
    audio, sr = sf.read(str(session / "simulated_take.wav"))
    sf.write(str(session / "short.wav"), audio[: int(len(audio) * 0.75)], sr)  # lose the last 3 notes
    with pytest.raises(sl.SliceError, match="could not find [23] of 12"):
        sl.slice_session(session, session / "short.wav")
    assert not (session / "labels.csv").exists()
    assert (session / "slicing.png").exists()  # diagnostic plot instead


def test_refuses_to_overwrite_a_sliced_session(session):
    sl.slice_session(session, session / "simulated_take.wav")
    with pytest.raises(sl.SliceError, match="Refusing to overwrite"):
        sl.slice_session(session, session / "simulated_take.wav")
    sl.slice_session(session, session / "simulated_take.wav", out=session / "copy")  # a new folder is fine
    assert (session / "copy" / "labels.csv").exists()


# --- pitch alignment ---------------------------------------------------------------
def test_align_drops_extras_and_marks_missing():
    playlist = [40, 45, 50, 55, 59]
    # events: noise (no pitch), 40, 45, 45 again (quieter re-pluck), 50, 59 (55 is missing)
    events = [None, 40.1, 44.9, 45.0, 50.0, 59.2]
    loud = [0.2, 1.0, 1.0, 0.4, 0.9, 0.8]
    match, dropped = sl.align_to_playlist(events, loud, playlist)
    assert match == [1, 2, 4, None, 5]
    assert dropped == [0, 3]  # the noise and the quieter of the two plucks


def test_align_keeps_a_wrong_note_matched():
    """A wrong note is matched (and flagged later), not dropped + marked missing."""
    match, dropped = sl.align_to_playlist([40.0, 47.0, 50.0], [1, 1, 1], [40, 45, 50])
    assert match == [0, 1, 2] and dropped == []


def test_align_tolerates_octave_errors():
    match, dropped = sl.align_to_playlist([52.0, 45.0], [1, 1], [40, 45])
    assert match == [0, 1]


def _onsets(session):
    sl.slice_session(session, session / "simulated_take.wav", out=session / "clean")
    return [float(r["onset_s"]) for r in csv.DictReader(open(session / "clean" / "labels.csv"))]


def test_extra_events_are_dropped_by_pitch_alignment(session):
    onsets = _onsets(session)
    audio, sr = sf.read(str(session / "simulated_take.wav"))
    peak = np.abs(audio).max()
    rng = np.random.default_rng(0)
    # a loud knock (35 ms thump) halfway between notes 3 and 4
    knock_s = (onsets[2] + onsets[3]) / 2
    t, n = int(knock_s * sr), int(0.035 * sr)
    audio[t : t + n] += peak * rng.standard_normal((n, 1)) * np.hanning(n)[:, None]
    # note 7 plucked twice: its first 0.5 s again, 0.9 s later, quieter
    a, b = int(onsets[6] * sr), int((onsets[6] + 0.9) * sr)
    seg = audio[a : a + int(0.5 * sr)].copy()
    audio[b : b + len(seg)] += 0.6 * seg
    sf.write(str(session / "messy.wav"), audio, sr)

    rows = sl.slice_session(session, session / "messy.wav", out=session / "messy")
    playlist = list(csv.DictReader(open(session / "playlist.csv")))
    assert len(rows) == 12
    for row, entry in zip(rows, playlist):
        assert (row["string"], row["fret"]) == (int(entry["string"]), int(entry["fret"]))
    meta = json.loads((session / "messy" / "metadata.json").read_text())
    al = meta["pitch_alignment"]
    assert al["used"] and al["missing_takes"] == []
    times = [d["time_s"] for d in al["dropped_events"]]
    assert any(abs(x - (onsets[6] + 0.9)) < 0.1 for x in times)  # the re-pluck was dropped
    # the knock is either not detected at all or dropped; it never becomes a note
    assert all(abs(float(r["onset_s"]) - knock_s) > 0.1 for r in rows)


def test_missing_note_is_left_out_and_listed(session):
    onsets = _onsets(session)
    audio, sr = sf.read(str(session / "simulated_take.wav"))
    # silence note 5 completely (the player skipped it)
    audio[int((onsets[4] - 0.05) * sr) : int((onsets[5] - 0.05) * sr)] *= 0.0005
    sf.write(str(session / "skipped.wav"), audio, sr)
    rows = sl.slice_session(session, session / "skipped.wav", out=session / "skipped")
    playlist = list(csv.DictReader(open(session / "playlist.csv")))
    expected = [e for k, e in enumerate(playlist) if k != 4]
    assert [(r["string"], r["fret"]) for r in rows] == [(int(e["string"]), int(e["fret"])) for e in expected]
    meta = json.loads((session / "skipped" / "metadata.json").read_text())
    assert meta["pitch_alignment"]["missing_takes"] == [5]


def test_wrong_note_is_flagged_not_relabeled(session):
    # Pretend the player read take 4 as 3 frets higher than they played it.
    path = session / "playlist.csv"
    rows = list(csv.DictReader(open(path)))
    s, f = int(rows[3]["string"]), int(rows[3]["fret"])
    new_fret = f + 3 if f <= 16 else f - 3
    rows[3]["fret"] = str(new_fret)
    rows[3]["midi"] = str(es.tuning.string_fret_to_midi(s, new_fret))
    with open(path, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    result = sl.slice_session(session, session / "simulated_take.wav")
    flagged = [r["take"] for r in result if r["flag"]]
    assert flagged == [4]
    assert result[3]["fret"] == new_fret  # the label is kept as listed

    dropped = sl.slice_session(session, session / "simulated_take.wav", drop_flagged=True, overwrite=True)
    assert len(list(csv.DictReader(open(session / "labels.csv")))) == len(dropped) - 1


# --- onset jitter ------------------------------------------------------------------
def test_positive_delay_is_a_pure_shift():
    rng = es.sample_rng(0, 1)
    audio, _ = es.render_note(es.default_params(3, 5), 22050, 1.0, rng)
    shifted = es.apply_onset_delay(audio, 10.0, -60.0, 22050, np.random.default_rng(0))
    n = int(round(0.010 * 22050))
    assert len(shifted) == len(audio)
    np.testing.assert_array_equal(shifted[n:-300], audio[: len(audio) - n][:-300])
    assert np.abs(shifted[:n]).max() < 0.01  # the lead-in is only faint noise


def test_jitter_off_has_no_column_and_on_is_in_range(tmp_path):
    es.generate_dataset(count=20, out_dir=tmp_path / "off", seed=4, duration=0.5, progress=False)
    es.generate_dataset(count=20, out_dir=tmp_path / "on", seed=4, duration=0.5, progress=False,
                        onset_jitter_ms=60.0)
    off = list(csv.DictReader(open(tmp_path / "off" / "labels.csv")))
    on = list(csv.DictReader(open(tmp_path / "on" / "labels.csv")))
    assert "onset_delay_ms" not in off[0]
    delays = [float(r["onset_delay_ms"]) for r in on]
    assert all(es.ONSET_EARLIEST_MS <= d <= 60.0 for d in delays)
    # same notes otherwise: every other label column matches
    for a, b in zip(off, on):
        assert {k: a[k] for k in a} == {k: b[k] for k in a}


def test_jitter_is_deterministic_across_workers(tmp_path):
    kw = dict(count=16, seed=9, duration=0.5, progress=False, onset_jitter_ms=40.0)
    es.generate_dataset(out_dir=tmp_path / "a", workers=1, **kw)
    es.generate_dataset(out_dir=tmp_path / "b", workers=2, **kw)
    for name in ["labels.csv"] + [f"audio/{i:06d}.wav" for i in range(1, 17)]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()

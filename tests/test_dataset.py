"""Dataset-level guarantees: determinism, file format, no clipping, labels."""

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

import electro_synth as es
import tuning

SCRIPT = Path(__file__).resolve().parent.parent / "synth" / "electro_synth.py"


def _files(folder: Path) -> dict[str, bytes]:
    out = {"labels.csv": (folder / "labels.csv").read_bytes()}
    for wav in sorted((folder / "audio").glob("*.wav")):
        out[wav.name] = wav.read_bytes()
    return out


def test_same_seed_is_byte_identical_even_with_workers(tmp_path):
    kw = dict(count=24, seed=7, duration=0.5, progress=False)
    es.generate_dataset(out_dir=tmp_path / "a", workers=1, **kw)
    es.generate_dataset(out_dir=tmp_path / "b", workers=1, **kw)
    es.generate_dataset(out_dir=tmp_path / "c", workers=3, **kw)
    a, b, c = (_files(tmp_path / x) for x in "abc")
    assert len(a) == 25
    assert a == b, "same seed, same settings must give identical bytes"
    assert a == c, "worker count must not change the output"


def test_different_seed_differs(tmp_path):
    kw = dict(count=6, duration=0.5, progress=False)
    es.generate_dataset(out_dir=tmp_path / "a", seed=1, **kw)
    es.generate_dataset(out_dir=tmp_path / "b", seed=2, **kw)
    assert _files(tmp_path / "a") != _files(tmp_path / "b")


def test_format_duration_and_sample_rate(tmp_path):
    for sr, dur in [(22050, 2.0), (16000, 1.0), (44100, 0.75)]:
        out = tmp_path / f"sr{sr}"
        es.generate_dataset(count=4, out_dir=out, seed=3, sample_rate=sr, duration=dur, progress=False)
        for wav in (out / "audio").glob("*.wav"):
            info = sf.info(str(wav))
            assert info.samplerate == sr
            assert info.channels == 1
            assert info.subtype == "PCM_16"
            assert info.frames == round(sr * dur)


def test_no_clipping_even_at_extremes(tmp_path):
    """Hardest pick + longest decay at every position: still under -1 dBFS."""
    limit = int(round(32767 * 10 ** (es.PEAK_DBFS_MAX / 20))) + 1
    for string, fret in tuning.all_positions(19):
        params = es.NoteParams(string, fret, pick_strength=1.0, pluck_position=0.08,
                               decay_scale=1.4, detune_cents=3.0, noise_db=-50.0)
        audio, _ = es.render_note(params, 22050, 2.0, es.sample_rng(0, string * 100 + fret))
        pcm = es.to_int16(audio)
        assert np.max(np.abs(pcm.astype(np.int32))) <= limit
        assert np.all(np.isfinite(audio))
    # and through the full write path
    es.generate_dataset(count=30, out_dir=tmp_path, seed=11, progress=False)
    for wav in (tmp_path / "audio").glob("*.wav"):
        pcm, _ = sf.read(str(wav), dtype="int16")
        assert np.max(np.abs(pcm.astype(np.int32))) <= limit
        assert np.max(np.abs(pcm)) > 1000, "note should not be silent"


def test_positions_are_balanced():
    for count in (120, 200, 240, 1000):
        counts = Counter(es.plan_positions(count, 19, seed=42))
        assert len(counts) == min(count, 120)
        assert max(counts.values()) - min(counts.values()) <= 1, count
    assert set(Counter(es.plan_positions(240, 19, 5)).values()) == {2}


def test_labels_and_metadata(tmp_path):
    meta = es.generate_dataset(count=10, out_dir=tmp_path, seed=42, max_fret=12, progress=False)
    with open(tmp_path / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert list(rows[0].keys()) == es.LABEL_COLUMNS
    for i, row in enumerate(rows, start=1):
        assert row["file"] == f"audio/{i:06d}.wav"
        assert (tmp_path / row["file"]).exists()
        s, f = int(row["string"]), int(row["fret"])
        assert 1 <= s <= 6 and 0 <= f <= 12
        assert int(row["midi"]) == tuning.string_fret_to_midi(s, f)
        assert abs(float(row["freq_hz"]) - tuning.string_fret_to_hz(s, f)) < 1e-3
        for name, (lo, hi) in es.PARAM_RANGES.items():
            assert lo <= float(row[name]) <= hi
    on_disk = json.loads((tmp_path / "metadata.json").read_text())
    assert on_disk["seed"] == 42 and on_disk["synth_version"] == es.SYNTH_VERSION
    assert on_disk["count"] == meta["count"] == 10


def test_cli_end_to_end(tmp_path):
    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--count", "3", "--out", str(out), "--seed", "1",
         "--duration", "0.5", "--quiet"],
        check=True,
    )
    assert len(list((out / "audio").glob("*.wav"))) == 3
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["cli_args"]["count"] == 3

    preview = tmp_path / "preview"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--preview", "--out", str(preview), "--duration", "0.5"],
        check=True,
    )
    names = {p.name for p in preview.glob("*.wav")}
    assert len(names) == 24  # 6 strings x frets {0, 5, 12, 19}
    assert "s6_f00_E2.wav" in names and "s1_f05_A4.wav" in names

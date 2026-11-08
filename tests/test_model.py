"""Step 3: model, training loop, baseline and metrics."""

import json

import numpy as np
import pytest
import torch

import baseline
import dataset as D
import electro_synth as es
import evaluate
import extract
import split
import train
from tiny_cnn import TinyCNN, TinyCNNConfig, count_parameters, load_checkpoint, predict_logits

POS = D.PositionMap(19)


# --- classes -------------------------------------------------------------------
def test_position_map_round_trip():
    assert POS.n_classes == 120
    assert POS.to_class(1, 0) == 0 and POS.to_class(6, 19) == 119
    for c in range(POS.n_classes):
        assert POS.to_class(*POS.to_position(c)) == c
    midi = POS.class_midi()
    assert midi[POS.to_class(1, 0)] == midi[POS.to_class(2, 5)] == 64  # both E4
    assert len(set(midi.tolist())) == 44  # distinct pitches on a 20-fret neck


def test_frames_for_seconds():
    meta = {"n_frames": 87, "sample_rate": 22050, "hop_length": 512}
    assert D.frames_for_seconds(1.0, meta) == 44
    assert D.frames_for_seconds(10.0, meta) == 87  # capped at what exists


# --- model ---------------------------------------------------------------------
def test_model_shapes_and_parameter_count():
    model = TinyCNN(TinyCNNConfig())
    out = model(torch.zeros(3, 1, 252, 44))
    assert out.shape == (3, 120)
    assert count_parameters(model) == 283_944
    assert model.n_freq_out == 63  # frequency only pooled 4x: pitch survives


def test_config_round_trip():
    cfg = TinyCNNConfig(n_frames=30, channels=(8, 8, 8, 8))
    assert TinyCNNConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_overfit_one_batch_reaches_100_percent():
    """The sanity check itself works: 64 random inputs with random labels
    can be memorized (on CPU, with a small input to keep it quick)."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 1, 252, 16, generator=g)
    y = torch.randint(0, 120, (64,), generator=g)
    result = train.overfit_one_batch(TinyCNNConfig(n_frames=16), X, y, lr=1e-3)
    assert result["final_accuracy"] == 1.0


def test_too_short_input_is_a_clear_error():
    with pytest.raises(ValueError, match="too short"):
        TinyCNN(TinyCNNConfig(n_frames=8))


# --- baseline ------------------------------------------------------------------
@pytest.mark.parametrize(
    "midi, expected",
    [(64, (1, 0)), (59, (2, 0)), (60, (2, 1)), (40, (6, 0)), (44, (6, 4)), (45, (5, 0)),
     (83, (1, 19)), (90, (1, 19)), (30, (6, 0))],
)
def test_lowest_fret_position(midi, expected):
    assert baseline.lowest_fret_position(midi, 19) == expected


def test_strongest_bin_midi():
    feats = np.full((2, 252, 30), -80.0, dtype=np.float16)
    feats[0, extract.midi_to_bin(64)] = 0.0  # E4
    feats[1, extract.midi_to_bin(40) + 1] = 0.0  # E2, 33 cents sharp -> still E2
    assert baseline.strongest_bin_midi(feats).tolist() == [64, 40]


# --- metrics -------------------------------------------------------------------
def test_metrics_on_hand_made_predictions():
    t = [POS.to_class(2, 5), POS.to_class(2, 5), POS.to_class(6, 0), POS.to_class(3, 9)]
    p = [POS.to_class(2, 5), POS.to_class(1, 0), POS.to_class(6, 1), POS.to_class(3, 9)]
    m = evaluate.compute_metrics(np.array(t), np.array(p), POS)
    assert m["position_acc"] == 0.5
    assert m["pitch_acc"] == 0.75  # s2f5 -> s1f0 is still E4; s6f0 -> s6f1 is not
    assert m["string_acc_given_pitch"] == pytest.approx(2 / 3)
    cm = np.array(m["string_confusion_pitch_correct"])
    assert cm.sum() == 3 and cm[1, 0] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1
    assert m["top_mistakes"][0]["count"] == 1
    assert {(k["true"], k["pred"]) for k in m["top_mistakes"]} == {("s2 f5", "s1 f0"), ("s6 f0", "s6 f1")}


def test_chance_level_on_balanced_positions():
    every = np.arange(120)
    m = evaluate.compute_metrics(every, every, POS)
    assert m["position_acc"] == 1.0
    assert m["string_acc_given_pitch_chance"] == pytest.approx(44 / 120)


# --- end to end ----------------------------------------------------------------
def test_train_and_evaluate_end_to_end(tmp_path):
    """Tiny dataset -> features -> splits -> 2 epochs on CPU -> evaluation files."""
    d = tmp_path / "ds"
    es.generate_dataset(count=150, out_dir=d, seed=1, duration=1.0, max_fret=4, progress=False)
    extract.extract_features(d, progress=False)
    split.make_splits(d)
    run = tmp_path / "run"
    args = train.build_parser().parse_args(
        ["--data", str(d), "--run", str(run), "--epochs", "2", "--device", "cpu", "--skip-sanity"]
    )
    summary = train.train(args)
    assert summary["model_config"]["n_classes"] == 30
    for name in ("best.pt", "history.csv", "learning_curves.png", "config.json", "sanity_check.json"):
        assert (run / name).exists(), name

    model, ckpt = load_checkpoint(run / "best.pt")
    assert ckpt["max_fret"] == 4
    results = evaluate.evaluate(run, d, "test", device="cpu")
    assert results["baseline"]["pitch_acc"] == 1.0
    out = run / "eval" / "ds_test"
    for name in ("report.txt", "metrics.json", "string_confusion.png", "predictions.csv"):
        assert (out / name).exists(), name

    import compare

    prefix = tmp_path / "cmp" / "comparison"
    compare.main(["--runs", str(run), "--evals", "ds_test", "--confusion", "ds_test", "--out", str(prefix)])
    for name in ("comparison.md", "comparison_curves.png", "comparison_confusion.png"):
        assert (tmp_path / "cmp" / name).exists(), name

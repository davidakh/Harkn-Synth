# Electro: project context

Electro is a guitar **audio → tablature** model, trained entirely **from scratch**.
The author is learning ML by building it.

## Hard rules
- **No pretrained or third-party ML models, ever.** No downloaded weights, no
  pretrained feature extractors, no pitch-detection networks. Everything that
  learns is written and trained here. Classic DSP (FFT, filters, YIN) is fine.
- **Clarity over cleverness.** Prefer the readable version even when slower.
- **Comment the concepts (the *why*)**, not just the code: what physical or ML
  idea a block implements and why it's there. Explain terms on first use.
- Allowed dependencies (so far): numpy, scipy, soundfile, tqdm, numba (optional,
  speed only; the code must still run without it), librosa (signal processing
  only, e.g. CQT), matplotlib (figures only), pytest for tests. librosa pulls in
  scikit-learn as a dependency; never use it for models. Ask before adding
  anything else.
- Python 3.11+. Use the venv in `.venv/`.

## Layout
```
Electro Synth/             # repo root (GitHub: davidakh/Electro-Synth, private)
  CLAUDE.md, README.md, requirements.txt
  synth/electro_synth.py   # Step 1: dataset generator (Karplus-Strong synth + CLI)
  synth/tuning.py          # string/fret -> MIDI -> Hz, cents helpers
  features/extract.py      # Step 2: WAV -> CQT dB spectrograms (features.npy)
  features/split.py        # Step 2: stratified train/val/test + train-only norm stats
  tools/show.py            # learning figures (spectrogram PNGs, note-name axes)
  tests/                   # pytest; tests/pitch_utils.py = independent f0 estimator
  data/                    # generated datasets (gitignored, reproducible from seed)
```

## Conventions
- String numbering: **1 = high E (E4) … 6 = low E (E2)**. Standard tuning, MIDI
  E2=40 A2=45 D3=50 G3=55 B3=59 E4=64, `midi = open + fret`, A4 = 440 Hz.
- Datasets: `data/<name>/audio/000001.wav` (mono, 16-bit PCM),
  `labels.csv` (one row per file, ground truth + every randomized parameter),
  `metadata.json` (CLI args, seed, synth version, timestamp).
- **Reproducibility is a requirement**: same seed → byte-identical audio and
  labels, regardless of `--workers`. Per-sample randomness comes from
  `SeedSequence(seed, spawn_key=(0, index))`; the dataset layout from
  `spawn_key=(1,)`. Never use global/unseeded randomness.
- (string, fret) positions are balanced (each appears k or k+1 times), not iid.
- Bump `SYNTH_VERSION` whenever the audio a given seed produces changes.
- Features: `data/<name>/features.npy` is (N, 252 bins, frames) float16 dB;
  **row i == labels.csv row i**, always. Everything downstream refers to notes
  by that row index. `features_meta.json` stores the sha256 of labels.csv;
  `extract.load_features()` refuses stale features. CQT: fmin D2 (MIDI 38),
  36 bins/octave, 7 octaves, hop 512; `bin = 3 * (midi - 38)`.
- Splits: `data/<name>/splits/{train,val,test}.csv` (column `row` = features
  index), stratified per (string, fret), seed via `spawn_key=(2,)`.
  Normalization stats come from **train only** (`splits/norm_stats.json`,
  per-bin mean/std). Never fit anything on val/test.
- Bump `FEATURES_VERSION` whenever the features for given audio change.

## Roadmap
1. **Electro Synth v0.1** (done): labeled single plucked notes.
2. **Features** (done): CQT spectrograms, stratified splits, norm stats, figures.
3. Next (not started): model training on these features, then more realistic
   data (chords, real recordings for evaluation, etc.).

## Commands
```
.venv/bin/python -m pytest -q
.venv/bin/python synth/electro_synth.py --preview
.venv/bin/python synth/electro_synth.py --count 5000 --out data/notes_v1 --seed 42 --workers 4
.venv/bin/python features/extract.py --data data/notes_v1
.venv/bin/python features/split.py --data data/notes_v1
.venv/bin/python tools/show.py --data data/notes_v1
```
Git: `/usr/bin/git` is broken on this machine (xcode-select points at a
missing Xcode Beta). Use `/Library/Developer/CommandLineTools/usr/bin/git`.

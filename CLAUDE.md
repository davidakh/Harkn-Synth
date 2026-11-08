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
  only, e.g. CQT), matplotlib (figures only), torch (models we write and
  train ourselves; Apple GPU via MPS), pytest for tests. librosa pulls in
  scikit-learn as a dependency; never use it for models. Ask before adding
  anything else.
- Python 3.11+. Use the venv in `.venv/` (Homebrew Python 3.14, arm64). PyTorch
  2.15 ships 3.14 wheels and MPS works, so no separate training venv is needed.

## Layout
```
Electro Synth/             # repo root (GitHub: davidakh/Electro-Synth, private)
  CLAUDE.md, README.md, requirements.txt
  synth/electro_synth.py   # Step 1: dataset generator (Karplus-Strong synth + CLI)
  synth/tuning.py          # string/fret -> MIDI -> Hz, cents helpers
  synth/augment.py         # Step 5: recording-chain augmentation (--augment)
  features/extract.py      # Step 2: WAV -> CQT dB spectrograms (features.npy)
  features/split.py        # Step 2: stratified train/val/test + train-only norm stats
  tools/show.py            # learning figures (spectrogram PNGs, note-name axes)
  tools/record_session.py  # Step 4: real-guitar play list (+ --simulate dry-run take)
  tools/slice_session.py   # Step 4/5: recorded take -> labeled dataset (pitch alignment, never relabels)
  tools/compare_real.py    # Step 5: real vs synth diagnosis (spectra, decay, inharmonicity)
  model/dataset.py         # Step 3: features -> tensors, class <-> (string, fret)
  model/tiny_cnn.py        # Step 3: TinyCNN (4 conv blocks, time-only global pool)
  model/train.py           # Step 3: overfit-one-batch check, then AdamW + early stopping
  model/baseline.py        # Step 3: no-ML baseline (strongest bin + lowest fret)
  model/evaluate.py        # Step 3: CNN vs baseline metrics, confusion PNG
  model/compare.py         # Step 4: runs side by side (tables, curves, confusions)
  tests/                   # pytest; tests/pitch_utils.py = independent f0 estimator
  data/                    # generated datasets (gitignored, reproducible from seed)
  runs/                    # training runs: best.pt, logs, plots, eval/ (gitignored)
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
- **Shortcut warning**: synth notes start at exactly sample 0; tiny_cnn_v1 learned
  to depend on that (a 2 ms shift drops it from 96% to 40%). Synth v0.2 adds
  `--onset-jitter-ms` (random -5..+N ms start; separate rng stream 4, so jitter
  off = byte-identical v0.1 audio). Train anything meant for real audio with jitter.
- Real sessions: `data/real_session*/` (playlist.csv + recorded take -> audio/,
  labels.csv, metadata.json). Test-only: never train on them. The slicer never
  relabels: count mismatches go to pitch alignment (extras dropped and listed,
  missing notes left out; >10% missing aborts), pitch mismatches are flagged,
  and it refuses to overwrite a sliced session without --overwrite.
- **data/real_session1 is the locked real-world exam** (60 notes, phone mic, AAC).
  Never train on it, never re-slice it in place. Synth v0.3 was tuned *coarsely*
  against it (tools/compare_real.py): that's mild test-set leakage, so treat its
  score as slightly optimistic; record a separate dev session for finer tuning.
- Synth v0.3: string stiffness (dispersion allpasses; B_open per string, x2 every
  6 frets), body resonances (3 peaking modes), longer decay, brighter plain
  strings. `--augment` (rng stream 5): reverb, noise/hum/clicks, low/high cut,
  shelf, 3 bells, compression, AAC via afconvert at 44.1 kHz (MP3 fallback).
- Model classes: `class = (string - 1) * 20 + fret` (see `PositionMap`).
  Never pool the model over frequency: frequency position = pitch.
- Evaluate only on the test split or freshly generated data, always
  normalized with the *training* stats stored in the checkpoint. Always
  report against the baseline; the key metric is string accuracy when the
  pitch is right.

## Roadmap
1. **Electro Synth v0.1** (done): labeled single plucked notes.
2. **Features** (done): CQT spectrograms, stratified splits, norm stats, figures.
3. **TinyCNN** (done): 284k params, ~96% position accuracy on test vs 36.7%
   for the baseline; mistakes are mostly wound strings 5 <-> 6.
4. **Reality check + scale-up** (done except the real recording): play list
   + slicer, notes_v2 (50k, seed 1234) -> tiny_cnn_v2, onset-jitter fix
   (notes_v1j -> tiny_cnn_v1j). v2 99.8% aligned but 49% shifted; v1j ~86%
   either way, 77% on the simulated take. Real-guitar numbers pending.
5. **Sim-to-real** (done): compare_real diagnosis, synth v0.3 (stiffness, body),
   --augment recording chain, pitch-aligned slicing. tiny_cnn_v3 (notes_v3: 50k,
   seed 3003, jitter 60 + augment) = 63.3% position / 98.3% pitch on
   real_session1 (v1j: 33.3% / 83.3%). Errors: mostly "one string thinner".
6. Next: a separate real *dev* session for tuning; bigger/regularized model
   (v3 overfits after epoch 3); fix the one-string-thinner bias.

## Commands
```
.venv/bin/python -m pytest -q
.venv/bin/python synth/electro_synth.py --preview
.venv/bin/python synth/electro_synth.py --count 5000 --out data/notes_v1 --seed 42 --workers 4
.venv/bin/python features/extract.py --data data/notes_v1
.venv/bin/python features/split.py --data data/notes_v1
.venv/bin/python tools/show.py --data data/notes_v1
.venv/bin/python model/train.py --data data/notes_v1 --run runs/tiny_cnn_v1
.venv/bin/python model/evaluate.py --run runs/tiny_cnn_v1 --data data/notes_v1 --split test
```
Git: `/usr/bin/git` is broken on this machine (xcode-select points at a
missing Xcode Beta). Use `/Library/Developer/CommandLineTools/usr/bin/git`.

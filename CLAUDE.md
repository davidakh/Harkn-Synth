# Electro Synth: project context

Electro is a guitar **audio → tablature** model, trained entirely **from scratch**.
The author is learning ML by building it. Electro is two projects:

- **Electro Synth** (this folder): the data factory. Synth, augmentation, the
  shared CQT, real-session tools, and every dataset (`data/`).
- **Electro CNN** (`../Electro Models/Electro CNN`): features, splits, models,
  training runs. It installs this folder as a package (`pip install -e`) and
  imports `electro_synth`, `tuning`, `augment`, `spectrogram` from here.
  **Never copy synth code into it.** It finds datasets via `ELECTRO_DATA`.

## Hard rules
- **No pretrained or third-party ML models, ever.** No downloaded weights, no
  pretrained feature extractors, no pitch-detection networks. Everything that
  learns is written and trained here. Classic DSP (FFT, filters, YIN) is fine.
- **Clarity over cleverness.** Prefer the readable version even when slower.
- **Comment the concepts (the *why*)**, not just the code: what physical or ML
  idea a block implements and why it's there. Explain terms on first use.
- Allowed dependencies (so far): numpy, scipy, soundfile, tqdm, numba (optional,
  speed only; the code must still run without it), librosa (signal processing
  only, e.g. CQT), matplotlib (figures only), pytest for tests. (torch is
  Electro CNN's.) librosa pulls in scikit-learn as a dependency; never use it
  for models. Ask before adding anything else.
- Python 3.11+. Use the venv in `.venv/` (Homebrew Python 3.14, arm64).
- `pyproject.toml` installs `synth/*.py` as top-level modules
  (`py-modules`); a new module in `synth/` that Electro CNN needs must be added
  to that list. The package version is `SYNTH_VERSION`.

## Layout
```
Electro Synth/             # repo root (GitHub: davidakh/Electro-Synth, private)
  CLAUDE.md, README.md, requirements.txt, pyproject.toml
  synth/electro_synth.py   # dataset generator (Karplus-Strong synth + CLI)
  synth/tuning.py          # string/fret -> MIDI -> Hz, cents helpers
  synth/augment.py         # recording-chain augmentation (--augment)
  synth/spectrogram.py     # THE CQT (shared with Electro CNN's features/extract.py)
  tools/show.py            # learning figures (spectrogram PNGs from the WAVs)
  tools/record_session.py  # real-guitar play list (+ --simulate dry-run take)
  tools/slice_session.py   # recorded take -> labeled dataset (pitch alignment, never relabels)
  tools/compare_real.py    # real vs synth diagnosis (spectra, decay, inharmonicity)
  tests/                   # pytest; tests/pitch_utils.py = independent f0 estimator
  data/                    # ALL datasets (gitignored, reproducible from seed);
                           # Electro CNN writes features.npy + splits/ into them
  runs/real_vs_synth/      # compare_real.py output (gitignored)
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
- Changing `synth/spectrogram.py` changes Electro CNN's features: bump its
  `FEATURES_VERSION` and re-extract there.
- Features are made by Electro CNN but stored here, next to each dataset:
  `data/<name>/features.npy` (N, 252 bins, frames) float16 dB, **row i ==
  labels.csv row i**, plus `features_meta.json` (sha256 of labels.csv) and
  `splits/`. CQT: fmin D2 (MIDI 38), 36 bins/octave, 7 octaves, hop 512;
  `bin = 3 * (midi - 38)`.
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

## Roadmap (whole of Electro; model steps happen in Electro CNN)
1. **Electro Synth v0.1** (done): labeled single plucked notes.
2. **Features** (done): CQT spectrograms, stratified splits, norm stats, figures.
3. **TinyCNN** (done): 284k params, ~96% position accuracy on test vs 36.7%
   for the baseline; mistakes are mostly wound strings 5 <-> 6.
4. **Reality check + scale-up** (done): play list
   + slicer, notes_v2 (50k, seed 1234) -> tiny_cnn_v2, onset-jitter fix
   (notes_v1j -> tiny_cnn_v1j). v2 99.8% aligned but 49% shifted; v1j ~86%
   either way, 77% on the simulated take; v1j 33.3% on the real guitar.
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
.venv/bin/python tools/show.py --data data/notes_v1
.venv/bin/python tools/record_session.py --out data/real_session2
.venv/bin/python tools/slice_session.py --session data/real_session2 --wav take.wav
.venv/bin/python tools/compare_real.py --real data/real_session1 --out runs/real_vs_synth/x
```
Features, training and evaluation: run them from Electro CNN.
Git: `/usr/bin/git` is broken on this machine (xcode-select points at a
missing Xcode Beta). Use `/Library/Developer/CommandLineTools/usr/bin/git`.

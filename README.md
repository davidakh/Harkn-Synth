# Electro

A guitar audio → tablature model, built from scratch. So far:

1. **Electro Synth v0.1** (`synth/`): a generator for labeled training data,
   single plucked guitar notes where we know exactly which string and fret made
   each sound.
2. **Features** (`features/`, `tools/`): turns the audio into CQT spectrograms
   (the images the model will actually look at), splits the data into
   train/val/test, and computes normalization stats.
3. **TinyCNN** (`model/`): the first trained model. Predicts string + fret
   from a note's spectrogram, compared against a no-ML baseline.

## Setup

```bash
python3 -m venv .venv            # Python 3.11+
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# Listen first: 6 open strings + frets 5, 12, 19 on each -> data/preview/
.venv/bin/python synth/electro_synth.py --preview

# Generate a dataset
.venv/bin/python synth/electro_synth.py --count 5000 --out data/notes_v1 --seed 42

# Tests
.venv/bin/python -m pytest -q
```

| Option | Default | Meaning |
|---|---|---|
| `--count` | (required) | number of notes |
| `--out` | (required) | dataset folder; with `--preview`, overrides `data/preview` |
| `--seed` | 42 | global seed; same seed → byte-identical dataset |
| `--sample-rate` | 22050 | Hz |
| `--duration` | 2.0 | seconds per note |
| `--max-fret` | 19 | positions sampled are frets 0..max on all 6 strings |
| `--workers` | 1 | parallel processes (`0` = all cores); output is identical |
| `--preview` | off | write the listening set instead of a dataset |
| `--quiet` | off | no progress bar |

Speed (Apple Silicon, numba on): ~0.8 ms/note on one worker, so 5000 notes take
~4 s (~2 s with 4 workers). Below a few thousand notes, extra workers don't
help, because starting the processes costs more than they save. Each 2 s note
is ~86 KB, so 5000 notes take about 430 MB.

## Output

```
data/<name>/
  audio/000001.wav ...   mono, 16-bit PCM WAV
  labels.csv             one row per file
  metadata.json          how the dataset was made
```

### labels.csv columns

**Ground truth** (what the model will learn to predict):

| column | meaning |
|---|---|
| `file` | path relative to the dataset folder, e.g. `audio/000001.wav` |
| `string` | 1 = high E … 6 = low E |
| `fret` | 0 = open |
| `midi` | MIDI note number (open string + fret) |
| `note` | note name, e.g. `E2`, `C#4` |
| `freq_hz` | nominal equal-tempered frequency (A4 = 440) |

**Randomized playing parameters** (uniform within these ranges):

| column | range | meaning |
|---|---|---|
| `pick_strength` | 0.3 – 1.0 | soft to hard. Harder = brighter attack and louder |
| `pluck_position` | 0.08 – 0.30 | where on the string it's plucked, as a fraction of length from the bridge. Small = twangy, large = round |
| `decay_scale` | 0.6 – 1.4 | multiplies how long the string rings |
| `detune_cents` | −3 – +3 | slight mistuning (100 cents = 1 semitone) |
| `noise_db` | −70 – −50 | background hiss level relative to the note's peak |

**Derived values** (computed from the above; useful for analysis):

| column | meaning |
|---|---|
| `actual_freq_hz` | `freq_hz` after detune: the pitch actually in the audio |
| `t60_s` | seconds for the fundamental to fade by 60 dB |
| `loop_lowpass` | in-loop damping coefficient (0 = bright … 0.25 = darkest) |
| `loop_gain` | energy kept per round trip of the wave on the string |
| `delay_int`, `allpass_coef` | integer + fractional parts of the string length in samples |
| `comb_delay` | pluck position in samples (sets the comb-filter notches) |
| `peak_dbfs` | output peak level (−1 dBFS for hardest picks, −8 for softest) |

### metadata.json
All CLI args, seed, synth version, UTC timestamp, the parameter ranges and
per-string timbre table, generation time, and library versions. It's the
only file whose contents change between identical runs (because of the timestamp).

### Dataset layout guarantees
- **Balanced positions**: every (string, fret) appears either k or k+1 times
  (e.g. 5000 notes / 120 positions → 41 or 42 each), shuffled.
- **Reproducible**: each sample's randomness is derived from (seed, index), so
  the result doesn't depend on `--workers`. Note that the *layout* (which
  position goes at which index) depends on `--count` too, so the first 200
  notes of a 5000-note dataset differ from a 200-note dataset with the same seed.

## How the sound is made (short version)

Extended Karplus-Strong: a pluck-shaped burst circulates in a delay loop
whose length is one period of the note. A lowpass in the loop makes high
harmonics fade faster than low ones, like a real string. A fractional-delay
allpass tunes the loop to within ~0.01 cents. Wound strings get more in-loop
damping (darker) and longer sustain; higher frets ring shorter and darker.
Pluck position is a comb filter (notches at harmonics with a node at the pluck
point); pick strength controls attack brightness and level. Details are in
the comments of `synth/electro_synth.py`.

## Step 2: Features

```bash
.venv/bin/python features/extract.py --data data/notes_v1   # -> features.npy
.venv/bin/python features/split.py   --data data/notes_v1   # -> splits/
.venv/bin/python tools/show.py       --data data/notes_v1   # -> figures/*.png
```

On 5000 notes (Apple Silicon): extraction ~11 s, split + stats ~0.3 s,
figures ~2 s.

### What gets written (next to labels.csv)

| file | contents |
|---|---|
| `features.npy` | float16 array **(N notes, 252 bins, 87 frames)** of CQT magnitude in dB. **Row i = labels.csv row i.** ~209 MB for 5000 notes |
| `features_meta.json` | every CQT/dB parameter, shape, the sha256 of labels.csv (so stale features are detected), timing |
| `splits/train.csv`, `val.csv`, `test.csv` | the labels.csv rows of each split, plus a `row` column = index into features.npy |
| `splits/norm_stats.json` | mean/std from **train only**: overall, and per frequency bin (use the per-bin ones) |
| `splits/split_meta.json` | seed, ratios, counts |
| `figures/same_pitch_E4.png` | E4 played on strings 1–5: what tells strings apart |
| `figures/open_strings.png` | one open-string note per string |

### The spectrogram, in numbers

- **Frequency axis**: Constant-Q transform from D2 (73.4 Hz) up 7 octaves to
  ~9.2 kHz, 36 bins per octave = **3 bins per semitone**. Bin for a MIDI note:
  `bin = 3 * (midi - 38)`, so low E (40) is bin 6, A4 (69) is bin 93.
- **Time axis**: hop 512 samples = 23.2 ms per frame; a 2 s note = 87 frames.
- **Values**: `20·log10(|CQT|)` in dB, absolute (not per-note normalized),
  floored at −100 dB. Typical range: −90 (silence/hiss) to +20 (loud
  fundamental). Train mean ≈ −60.6 dB, std ≈ 14.7 dB.

### Splits

80/10/10, stratified per (string, fret): each position's notes are split
separately, so all 120 positions appear in every split. With 41–42 notes per
position that rounds to 33–34 / 4 / 4, i.e. 80.8% / 9.6% / 9.6%. Fixed seed
(`--seed`, default 42).

## Step 3: TinyCNN

```bash
.venv/bin/python model/train.py    --data data/notes_v1 --run runs/tiny_cnn_v1
.venv/bin/python model/evaluate.py --run runs/tiny_cnn_v1 --data data/notes_v1 --split test
.venv/bin/python model/baseline.py --data data/notes_v1 --split test      # baseline only
```

To check results on completely new notes, generate a set with another seed,
extract its features, and evaluate on all of it (no split needed):

```bash
.venv/bin/python synth/electro_synth.py --count 1000 --out data/notes_fresh1000 --seed 2026
.venv/bin/python features/extract.py --data data/notes_fresh1000
.venv/bin/python model/evaluate.py --run runs/tiny_cnn_v1 --data data/notes_fresh1000 --split all
```

- **Model**: 4 × (3×3 conv → batchnorm → ReLU → max-pool), average over
  time, then linear → 120 classes (6 strings × frets 0–19). 283,944
  parameters. Input: the first 1.0 s of the per-bin normalized CQT
  (`--input-seconds`).
- **Training**: cross-entropy, AdamW (lr 1e-3, weight decay 0.01), batch 64,
  ≤30 epochs, early stopping on val loss (patience 5), seed 42. Runs on the
  Apple GPU (MPS) in ~20 s. Before training it checks that a fresh model can
  memorize one batch of 64 to 100%, and refuses to continue if not.
- **Baseline**: pitch = strongest CQT bin, position = lowest fret for that
  pitch. No learning.

### Outputs (`runs/<run>/`)

| file | contents |
|---|---|
| `best.pt` | best checkpoint (lowest val loss) + model/train config + path of the norm stats |
| `config.json` | summary: parameters, device, sanity check, best epoch, timing |
| `history.csv`, `learning_curves.png` | per-epoch train/val loss and accuracy |
| `sanity_check.json` | result of the overfit-one-batch check |
| `eval/<dataset>_<split>/report.txt` | CNN vs baseline: position, pitch, string-given-pitch accuracy, per string, per fret region, top 10 mistakes |
| `eval/.../string_confusion.png` | 6×6 true vs predicted string (pitch-correct notes) |
| `eval/.../metrics.json`, `predictions.csv` | the same, machine-readable; one row per note |

### Results (seed 42)

| | test (480) | fresh set (1000, seed 2026) |
|---|---|---|
| TinyCNN position accuracy | 96.2% | 95.6% |
| baseline position accuracy | 36.7% | 36.7% |
| pitch accuracy (both) | 100% | 100% |

Every CNN mistake has the right pitch but the wrong string, mostly between the
two lowest wound strings (5 ↔ 6). Training with seed 7 instead gives 93.5% on
test, so expect a couple of points of variation between runs.

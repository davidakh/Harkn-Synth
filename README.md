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
4. **Reality check + scale-up** (`tools/record_session.py`,
   `tools/slice_session.py`, `model/compare.py`): a real-guitar test set
   workflow, a 50k-note dataset, and the onset-timing shortcut fix.
5. **Closing the sim-to-real gap** (`tools/compare_real.py`, `synth/augment.py`,
   synth v0.3): diagnose real vs synth, model string stiffness and the
   guitar body, randomize the recording chain, align takes by pitch.

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

## Step 4: Real-guitar test set

### 1. Make the play list
```bash
.venv/bin/python tools/record_session.py --out data/real_session1
```
This prints a numbered checklist and tab of 60 positions: 10 per string,
frets spread 0–19, in random order (fixed seed). The same list is saved as
`data/real_session1/playlist.txt` and `playlist.csv`.

### 2. Record
- **Before you start:** tune to standard tuning (A440). Use a clean tone, no effects.
- **Playing:** play the notes in list order, one at a time, about 2 s apart. Mute each note before the next one.
- **If you fumble:** restart the take. Notes are matched to the list *by order*.
- **Saving:** save as WAV (any sample rate, mono or stereo).

### 3. Slice, extract, evaluate
```bash
.venv/bin/python tools/slice_session.py --session data/real_session1 --wav path/to/take.wav
.venv/bin/python features/extract.py --data data/real_session1
.venv/bin/python model/evaluate.py --run runs/tiny_cnn_v1j --data data/real_session1 --split all
```
- **Count check:** if the number of detected notes differs from the play list,
  nothing is written. You get the onset times and `slicing.png` instead; adjust
  `--min-gap` / `--quiet-db` or re-record.
- **Pitch check:** every note is checked with pYIN. Notes more than half a
  semitone from their label (after removing the guitar's overall tuning
  offset) are **flagged** in `labels.csv` and printed, never relabeled.
  `--drop-flagged` leaves them out.
- **Level:** each note is scaled to a −3 dBFS peak, the middle of the synth's range.
- **Dry run without a guitar:** `record_session.py --simulate` writes
  `simulated_take.wav`, the synth playing the list with realistic messiness
  (uneven spacing, overlapping notes, muting clicks, room noise, 44.1 kHz stereo).

### Onset jitter (synth v0.2)
Synth notes always start at exactly sample 0, and tiny_cnn_v1 learned to rely
on that. Delaying a note by **2 ms** drops it from 96% to 40%. Real sliced
notes never line up like that.

`--onset-jitter-ms N` shifts every note by a random −5…+N ms. Off by default,
and with it off the output is byte-identical to v0.1.

`data/notes_v1j` is notes_v1 (same seed, same notes) with `--onset-jitter-ms 60`.

## Step 4 results: v1 vs v2 vs v1j

- **v1**: 5k notes.
- **v2**: 50k notes, same synth, same settings (seed 1234).
- **v1j**: v1's notes with onset jitter.

To rebuild the tables and plots:

```bash
.venv/bin/python model/compare.py --runs runs/tiny_cnn_v1 runs/tiny_cnn_v2 runs/tiny_cnn_v1j \
    --evals notes_v1_test notes_v2_test notes_v1j_test notes_fresh1000_all sim_session_all \
    --confusion sim_session_all --out runs/comparison_step4
```

| position accuracy | v1 | v2 | v1j | baseline |
|---|---|---|---|---|
| train − val accuracy gap | 5.0 pts | −0.1 pts | 10.4 pts | |
| synth test, aligned (notes_v1 test) | 96.2% | **99.8%** | 86.9% | 36.7% |
| synth test, shifted (notes_v1j test) | 44.2% | 49.0% | **86.2%** | 36.7% |
| fresh set (1000, aligned) | 95.6% | **99.9%** | 86.8% | 36.7% |
| simulated take (60) | 40.0% | 46.7% | **76.7%** | 36.7% |
| real guitar (60) | *record first* | | | |

- **More data:** 10× more data removed the overfitting gap and made v2
  near-perfect on aligned synth notes.
- **The shortcut:** v2 is still as dependent on exact note timing as v1
  (49% shifted). More of the *same* data doesn't fix a shortcut.
- **Changing the data:** jittered starts (v1j) give a model that scores the
  same whether notes are aligned or not, and nearly doubles v1 on the
  simulated take.

## Step 5: Closing the sim-to-real gap

### Diagnose
```bash
.venv/bin/python tools/compare_real.py --real data/real_session1 --out runs/real_vs_synth/before
.venv/bin/python tools/compare_real.py --augment --out runs/real_vs_synth/after_v03_aug
```
For each real note, this renders a synth "twin" at the same position and
measures both the same way:
- fundamental strength vs the strongest partial
- harmonic profile per string (pitch-aligned)
- decay T60
- noise floor
- inharmonicity B
- average spectrum vs frequency (the body + room + mic + codec coloration)

It writes `side_by_side.png`, `harmonic_profiles.png`, `coloration.png` and
`report.md`.

### Synth v0.3
- **String stiffness:** overtones run sharp, `f_k = k·f0·√(1 + B k²)`, via
  allpass filters in the string loop. B per open string (from the real
  recording) runs from 3e-4 (low E) to 0.2e-4 (high E). It doubles every
  6 frets and gets a random ×0.5–2 per note.
- **Guitar body:** three resonances (~105 / 210 / 350 Hz, randomized).
- **Retuned strings:** longer decay (T60 4–6 s open) and much brighter plain
  strings, both randomized per note.

### `--augment`: the recording chain
`synth/augment.py` applies, in order:
1. room reverb (synthetic impulse response)
2. room noise, mains hum, handling clicks
3. low cut, high cut, a ±12 dB shelf tilt, and three ±15 dB bell filters
4. phone-style compression
5. an AAC encode/decode at 32–128 kbps (via macOS `afconvert`; MP3 elsewhere)

Every parameter is random per note and logged in `labels.csv` (`aug_*` columns).

### Pitch-aligned slicing
If the onset count doesn't match the play list, `slice_session.py` detects
events more permissively, measures each one's pitch, and aligns them to the
play list with dynamic programming:
- **Extra events are dropped and listed:** knocks, re-plucks, and noise.
- **Missing notes are left out and listed.**
- **Wrong notes stay matched but flagged.**
- **Too many missing notes (>10%):** nothing is written.

It never overwrites an existing sliced session unless you pass `--overwrite`;
use `--out` to slice into a new folder.

### Step 5 results

`tiny_cnn_v3` is the same model and settings as before, trained on
`notes_v3`: 50k notes, seed 3003, synth v0.3, `--onset-jitter-ms 60 --augment`.

| position accuracy | v1 | v2 | v1j | **v3** | baseline |
|---|---|---|---|---|---|
| **real guitar (real_session1, 60)** | 30.0% | 13.3% | 33.3% | **63.3%** | 13.3% |
| real guitar: pitch | 81.7% | 43.3% | 83.3% | **98.3%** | 68.3% |
| real guitar: string when pitch right (chance 37%) | 36.7% | 30.8% | 40.0% | **64.4%** | 19.5% |
| own synth test | 96.2% | 99.7% | 86.2% | 89.2% | |
| old v0.1 synth (notes_v1 test) | 96.2% | 99.8% | 86.9% | 39.2% | 36.7% |

- **Real guitar:** v3 roughly doubles the real-guitar score and nearly solves
  pitch. With only 60 notes, ±12 points is the 95% range, and the synth was
  tuned coarsely on these same notes, so treat 63% as somewhat optimistic.
- **Old synth:** v3 is near chance on v0.1 synth notes. That synth lacks
  stiffness and has short decays, i.e. it isn't a realistic guitar any more,
  so those sets stopped being useful benchmarks.
- **Remaining real errors** almost all pick the next thinner string
  (D→G, B→E, A→D).

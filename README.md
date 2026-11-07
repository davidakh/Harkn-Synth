# Electro

A guitar audio → tablature model, built from scratch. This repo currently
contains **Step 1: Electro Synth v0.1**, a generator for labeled training data:
single plucked guitar notes where we know exactly which string and fret made
each sound.

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

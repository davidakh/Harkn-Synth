# Electro Synth

The data factory for **Electro**, a guitar audio → tablature model built from
scratch. This project makes the labeled notes and the real-guitar test sets;
the model that learns from them lives next door in
[`../Electro Models/Electro CNN`](../Electro%20Models/Electro%20CNN), which installs this
project as a dependency.

- **The synth** (`synth/electro_synth.py`): single plucked guitar notes where
  we know exactly which string and fret made each sound. Extended
  Karplus-Strong with string stiffness and a guitar body (v0.3).
- **Recording-chain augmentation** (`synth/augment.py`, `--augment`): room,
  noise, mic EQ, compression, AAC.
- **The CQT** (`synth/spectrogram.py`): the one definition of what a note
  "looks like", shared with Electro CNN so the two can't drift apart.
- **Real-guitar sessions** (`tools/record_session.py`, `tools/slice_session.py`):
  a play list to record, and a slicer that turns the take into a labeled dataset.
- **Diagnosis and figures** (`tools/compare_real.py`, `tools/show.py`).
- **Datasets** (`data/`): everything Electro CNN trains and tests on. They stay
  here; Electro CNN finds them through its `ELECTRO_DATA` setting.

## Setup

```bash
python3 -m venv .venv            # Python 3.11+
.venv/bin/pip install -r requirements.txt
```

`pyproject.toml` makes this folder installable, so another project can use it
with `pip install -e "<path to Electro Synth>[fast]"` (`[fast]` adds numba).
The modules are installed by name (`import electro_synth`, `import tuning`,
`import augment`, `import spectrogram`); editable means a change here is picked
up there with no reinstall.

## Run

```bash
# Listen first: 6 open strings + frets 5, 12, 19 on each -> data/preview/
.venv/bin/python synth/electro_synth.py --preview

# Generate a dataset
.venv/bin/python synth/electro_synth.py --count 5000 --out data/notes_v1 --seed 42

# The current training set: v0.3 + onset jitter + recording chain (~7 min)
.venv/bin/python synth/electro_synth.py --count 50000 --out data/notes_v3 --seed 3003 \
    --onset-jitter-ms 60 --augment --workers 0
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
| `--onset-jitter-ms` | 0 | start each note at a random −5…+N ms (see below) |
| `--augment` | off | apply the recording chain (see below) |
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

## Figures

```bash
.venv/bin/python tools/show.py --data data/notes_v1   # -> data/notes_v1/figures/*.png
```

| file | contents |
|---|---|
| `figures/same_pitch_E4.png` | E4 played on strings 1–5: what tells strings apart |
| `figures/open_strings.png` | one open-string note per string |

Spectrograms are computed from the WAVs with `synth/spectrogram.py`: CQT from
D2 (73.4 Hz) up 7 octaves, 3 bins per semitone (`bin = 3 * (midi - 38)`),
hop 512 samples (23.2 ms), absolute dB floored at −100. It's the same picture
Electro CNN trains on.

## Real-guitar sessions

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
- **If you fumble:** keep going. Extra events are dropped by pitch alignment (below); a re-take is only needed if more than 10% of notes are missing.
- **Saving:** save as WAV (any sample rate, mono or stereo).

### 3. Slice
```bash
.venv/bin/python tools/slice_session.py --session data/real_session1 --wav path/to/take.wav
```
Then extract features and evaluate from Electro CNN (see its README).
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

## Closing the sim-to-real gap (v0.3)

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

## Results

Training runs, scores and the model comparison are in Electro CNN's README.
Headline: `tiny_cnn_v3`, trained on `data/notes_v3` (50k notes, seed 3003,
synth v0.3, `--onset-jitter-ms 60 --augment`), gets **63.3%** string + fret
and 98.3% pitch on the real guitar (`data/real_session1`), against 33.3% /
83.3% before v0.3 and the augmentation.

## Tests

```bash
.venv/bin/python -m pytest -q
```

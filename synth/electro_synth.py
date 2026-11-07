"""Electro Synth v0.1: generate a labeled dataset of single plucked guitar notes.

Why a synthesizer?
------------------
To train an audio -> tablature model we need audio where we *know* exactly
which string and fret produced each sound. Recording and hand-labeling real
guitar takes forever; synthesizing gives perfect labels for free. The catch is
that a model trained only on synthetic audio learns synthetic quirks, so we
randomize the things that vary in real playing (how hard you pick, where you
pick, how long the string rings, slight mistuning, background noise). This
"domain randomization" pushes the model to rely on what actually identifies a
string/fret: pitch plus string-specific timbre.

How the sound is made: Karplus-Strong (KS)
------------------------------------------
A plucked string is a wave bouncing back and forth between the nut and the
bridge. Every round trip takes one period T = 1/f0 and loses a little energy,
more at high frequencies than low ones. KS models exactly that:

    excitation (a burst of noise = the pluck)
        |
        v
      (+)----------------------------------------> output y[n]
        ^                                   |
        |                                   v
    [allpass: fractional delay] <- [lowpass: energy loss] <- [delay line: N samples]

A signal that keeps circling a loop of total delay P samples repeats every P
samples, i.e. it has pitch fs / P. The lowpass in the loop makes high harmonics
die faster than low ones, which is the characteristic "pluck then mellow"
sound of a string.

"Extended" KS (Jaffe & Smith, 1983) adds the pieces we use here:
  * fractional-delay allpass so P need not be an integer (accurate pitch)
  * adjustable loop lowpass (string brightness) and loop gain (decay time)
  * pick-position comb filter and pick-strength (dynamics) filtering

Run `python synth/electro_synth.py --help` for CLI usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import lfilter
from tqdm import tqdm

# When run as a script, Python puts this file's folder (synth/) on sys.path,
# so the sibling module imports directly. Tests add synth/ to sys.path too.
import tuning

SYNTH_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "data" / "preview"

# ---------------------------------------------------------------------------
# Optional numba acceleration
# ---------------------------------------------------------------------------
# The string loop is inherently sequential (each output sample depends on
# earlier output samples), so numpy can't vectorize it. numba compiles the
# plain-Python loop to machine code. Without numba the same function runs as
# ordinary Python, roughly 50x slower but identical in behavior.
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


# ---------------------------------------------------------------------------
# Per-string timbre
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StringTimbre:
    """Physical character of one string, at the open position.

    t60_open:  seconds for the fundamental to decay by 60 dB (to 1/1000 of its
               amplitude). Thick wound strings store more energy and ring longer.
    loop_lowpass_open:  coefficient `b` of the in-loop lowpass (0 = no high-
               frequency loss, 0.25 = maximum loss). Wound strings have more
               internal friction, so they lose highs faster -> darker, warmer.
    """

    t60_open: float
    loop_lowpass_open: float


STRING_TIMBRE: dict[int, StringTimbre] = {
    6: StringTimbre(t60_open=4.0, loop_lowpass_open=0.20),  # low E, wound: darkest
    5: StringTimbre(t60_open=3.6, loop_lowpass_open=0.18),  # A, wound
    4: StringTimbre(t60_open=3.2, loop_lowpass_open=0.16),  # D, wound
    3: StringTimbre(t60_open=2.8, loop_lowpass_open=0.13),  # G, wound on acoustic
    2: StringTimbre(t60_open=2.4, loop_lowpass_open=0.09),  # B, plain
    1: StringTimbre(t60_open=2.0, loop_lowpass_open=0.07),  # high E, plain: brightest
}

# Fretting shortens the vibrating length. A shorter string at the same
# tension is stiffer relative to its length and has the player's fingertip
# damping it, so in practice it decays faster and sounds a bit duller.
# Every 18 frets halves T60; each fret adds a little extra high-frequency loss.
FRET_T60_HALVING = 18.0
FRET_LOWPASS_PER_FRET = 0.003
LOOP_LOWPASS_MAX = 0.25  # at 0.25 the loop filter has a zero at Nyquist

# Ranges for the per-sample randomized parameters (uniform distributions).
PARAM_RANGES: dict[str, tuple[float, float]] = {
    # 0.3 = soft fingertip-ish pluck, 1.0 = hard pick attack
    "pick_strength": (0.3, 1.0),
    # where the string is plucked, as a fraction of its length from the bridge
    # (~0.08 near the bridge = twangy, ~0.30 over the neck = round/soft)
    "pluck_position": (0.08, 0.30),
    # multiplies the string's T60 (different strings, guitars, players)
    "decay_scale": (0.6, 1.4),
    # slight mistuning of the instrument
    "detune_cents": (-3.0, 3.0),
    # background noise RMS relative to the note's peak, in dB
    "noise_db": (-70.0, -50.0),
}

# Mid-range values used for --preview and for the strict pitch test.
DEFAULT_PARAMS = {
    "pick_strength": 0.7,
    "pluck_position": 0.18,
    "decay_scale": 1.0,
    "detune_cents": 0.0,
    "noise_db": -60.0,
}

# Output level: never above -1 dBFS (headroom), and softer picks are quieter
# (down to -8 dBFS at the softest), so loudness varies like real playing.
PEAK_DBFS_MAX = -1.0
PEAK_DBFS_RANGE = 7.0
FADE_OUT_SECONDS = 0.010  # short fade so truncating a ringing note doesn't click


@dataclass(frozen=True)
class NoteParams:
    """Everything that defines one rendered note (besides the noise draws)."""

    string: int
    fret: int
    pick_strength: float
    pluck_position: float
    decay_scale: float
    detune_cents: float
    noise_db: float


def random_params(rng: np.random.Generator, string: int, fret: int) -> NoteParams:
    # Draw in a fixed order so a given rng state always yields the same note.
    values = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in PARAM_RANGES.items()}
    return NoteParams(string=string, fret=fret, **values)


def default_params(string: int, fret: int) -> NoteParams:
    return NoteParams(string=string, fret=fret, **DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------
def allpass_coefficient(delay: float, omega: float) -> float:
    """First-order allpass coefficient giving exactly `delay` samples of phase
    delay at angular frequency `omega` (radians/sample).

    Why an allpass? It passes every frequency at full volume (so it doesn't
    change the tone) but delays the signal by a *fractional* number of samples.
    That lets the loop length be e.g. 267.43 samples instead of only 267 or
    268. With integer delays only, a high note like 988 Hz at 22050 Hz
    (22.3 samples per period) could be off by up to ~40 cents.

    The allpass delay is frequency-dependent, so we solve for the coefficient
    that is exact at the fundamental. (The common approximation
    C = (1 - d) / (1 + d) is only exact as omega -> 0.)
    Keeping `delay` in [0.5, 1.5) keeps |C| small and the filter well behaved.
    """
    return math.sin(omega * (1.0 - delay) / 2.0) / math.sin(omega * (1.0 + delay) / 2.0)


@njit(cache=True)
def _string_loop(excitation, n_samples, delay_int, gain, lowpass_b, allpass_c):
    """Run the Karplus-Strong feedback loop sample by sample.

    y[n] = excitation[n] + allpass(lowpass(y[n - delay_int]))

    Loop lowpass: a symmetric 3-tap FIR  gain * (b, 1 - 2b, b).
      * Symmetric taps = linear phase = exactly 1 sample of delay at *every*
        frequency, so it never detunes the string (only the allpass does
        fractional tuning, and we account for it exactly).
      * At DC its response is `gain`; at Nyquist `gain * (1 - 4b)`. Bigger b
        means highs lose more energy per round trip -> darker tone.
    Total loop delay = delay_int + 1 (lowpass) + allpass delay = samples/period.
    """
    y = np.zeros(n_samples)
    n_exc = excitation.shape[0]
    tap_side = gain * lowpass_b
    tap_mid = gain * (1.0 - 2.0 * lowpass_b)
    ap_x_prev = 0.0  # allpass input one sample ago
    ap_y_prev = 0.0  # allpass output one sample ago
    for n in range(n_samples):
        i = n - delay_int  # sample leaving the delay line now
        d0 = y[i] if i >= 0 else 0.0
        d1 = y[i - 1] if i - 1 >= 0 else 0.0
        d2 = y[i - 2] if i - 2 >= 0 else 0.0
        lp = tap_side * d0 + tap_mid * d1 + tap_side * d2
        # first-order allpass: out = C*in + in[n-1] - C*out[n-1]
        ap = allpass_c * lp + ap_x_prev - allpass_c * ap_y_prev
        ap_x_prev = lp
        ap_y_prev = ap
        exc = excitation[n] if n < n_exc else 0.0
        y[n] = exc + ap
    return y


def make_excitation(
    rng: np.random.Generator, period: float, pick_strength: float, pluck_position: float
) -> tuple[np.ndarray, int]:
    """Build the "pluck": the initial energy injected into the string, about
    one period long (one period = the whole string length in the loop).

    1. Pluck shape. Just before release, the pick or finger pulls the string
       into a triangle with its corner at the pluck point. Harmonic n of that
       shape has amplitude ~ sin(n*pi*position) / n^2: a strong fundamental,
       falling off with harmonic number.
       (Classic KS uses pure white noise here. But then each harmonic's
       amplitude is random, and on some notes the fundamental comes out
       weaker than the octave above. Real plucks don't do that.)
    2. Pick texture: white noise adds the "scratch" of the pick and gives
       every sample a slightly different attack. Harder picks add more.
    3. Pluck position -> comb filter x[n] - x[n - M], M = position * period,
       applied to the texture. Plucking at 1/5 of the string length can't
       excite harmonics that have a node (a still point) there, so harmonics
       5, 10, 15, ... vanish. The comb puts notches at exactly those harmonics.
       The triangle already has the same notches built in: its sin(n*pi*pos)
       factor *is* that comb's frequency response. That's why picking near
       the bridge sounds thin and twangy and over the neck sounds round.
    4. Pick strength -> brightness. A soft pluck is a rounded, slow push; a
       hard pick snap is sharp, i.e. rich in high frequencies. A one-pole
       lowpass smooths soft plucks. (Loudness is set later, at the output.)
    """
    length = max(4, int(round(period)))
    comb_delay = min(length - 2, max(1, int(round(pluck_position * period))))

    # 1. triangle: rises 0 -> 1 up to the pluck point, falls back to 0
    n = np.arange(length + comb_delay, dtype=np.float64)
    shape = np.where(n < comb_delay, n / comb_delay, (length - n) / (length - comb_delay))
    shape = np.clip(shape, 0.0, None)

    # 2 + 3. comb-filtered noise texture
    noise = rng.uniform(-1.0, 1.0, length)
    texture = np.concatenate([noise, np.zeros(comb_delay)])
    texture[comb_delay:] -= noise
    texture_gain = 0.15 + 0.35 * pick_strength

    excitation = shape + texture_gain * texture
    excitation -= excitation.mean()  # remove DC: a string can't hold a constant offset

    # 4. pick-strength lowpass: smoothing 0 = untouched (bright), -> 1 = dull
    smoothing = 0.85 * (1.0 - pick_strength)
    excitation = lfilter([1.0 - smoothing], [1.0, -smoothing], excitation)
    return excitation, comb_delay


def string_physics(params: NoteParams, sample_rate: int) -> dict:
    """Turn musical parameters into the numbers the loop needs."""
    nominal_hz = tuning.string_fret_to_hz(params.string, params.fret)
    actual_hz = tuning.detune_hz(nominal_hz, params.detune_cents)
    period = sample_rate / actual_hz  # loop length in samples (fractional)
    omega = 2.0 * math.pi * actual_hz / sample_rate  # fundamental, rad/sample

    timbre = STRING_TIMBRE[params.string]
    t60 = timbre.t60_open * 2.0 ** (-params.fret / FRET_T60_HALVING) * params.decay_scale
    lowpass_b = min(
        LOOP_LOWPASS_MAX, timbre.loop_lowpass_open + FRET_LOWPASS_PER_FRET * params.fret
    )

    # Split the period: integer delay line + 1 sample (lowpass) + fractional
    # allpass delay in [0.5, 1.5).
    delay_int = int(math.floor(period - 1.5))
    allpass_delay = period - 1.0 - delay_int
    allpass_c = allpass_coefficient(allpass_delay, omega)

    # Loop gain from the desired decay time. Each trip around the loop (one
    # period) multiplies the fundamental by gain * |lowpass(f0)|. After t60
    # seconds = f0 * t60 trips it should be 10^(-60/20) = 10^-3, so the
    # per-trip factor is 10^(-3 / (f0 * t60)). Divide out the lowpass's own
    # attenuation at f0 so T60 means exactly "fundamental decays 60 dB".
    lowpass_mag_f0 = 1.0 - 2.0 * lowpass_b * (1.0 - math.cos(omega))
    gain = 10.0 ** (-3.0 / (actual_hz * t60)) / lowpass_mag_f0
    gain = min(gain, 0.99999)  # strictly < 1 keeps the loop stable

    return {
        "nominal_hz": nominal_hz,
        "actual_hz": actual_hz,
        "period_samples": period,
        "t60_s": t60,
        "loop_lowpass": lowpass_b,
        "loop_gain": gain,
        "delay_int": delay_int,
        "allpass_delay": allpass_delay,
        "allpass_coef": allpass_c,
    }


def render_note(
    params: NoteParams, sample_rate: int, duration: float, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Synthesize one note. Returns float audio in [-1, 1] and derived values.

    The `rng` supplies the pluck noise and background noise, so the same rng
    state + params always produce the same waveform.
    """
    n_samples = int(round(duration * sample_rate))
    phys = string_physics(params, sample_rate)

    excitation, comb_delay = make_excitation(
        rng, phys["period_samples"], params.pick_strength, params.pluck_position
    )
    y = _string_loop(
        excitation,
        n_samples,
        phys["delay_int"],
        phys["loop_gain"],
        phys["loop_lowpass"],
        phys["allpass_coef"],
    )

    # DC blocker (one-pole highpass ~20 Hz, below the lowest note at 82 Hz).
    # The loop passes DC with gain ~1, so any tiny offset would linger.
    r = math.exp(-2.0 * math.pi * 20.0 / sample_rate)
    y = lfilter([1.0, -1.0], [1.0, -r], y)
    y /= np.max(np.abs(y)) + 1e-12

    # Background noise: real recordings always have some hiss. Without it a
    # model can learn "silence = perfect zeros", which never happens live.
    noise_rms = 10.0 ** (params.noise_db / 20.0)
    y = y + rng.normal(0.0, noise_rms, n_samples)

    # Fade out the last few ms so the hard cut at `duration` doesn't click.
    n_fade = min(n_samples, int(FADE_OUT_SECONDS * sample_rate))
    if n_fade > 0:
        y[-n_fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, n_fade)))

    # Output gain: louder picks -> louder notes, always with headroom.
    peak_dbfs = PEAK_DBFS_MAX - PEAK_DBFS_RANGE * (
        (PARAM_RANGES["pick_strength"][1] - params.pick_strength)
        / (PARAM_RANGES["pick_strength"][1] - PARAM_RANGES["pick_strength"][0])
    )
    y *= 10.0 ** (peak_dbfs / 20.0) / (np.max(np.abs(y)) + 1e-12)

    phys["comb_delay"] = comb_delay
    phys["peak_dbfs"] = peak_dbfs
    return y, phys


def to_int16(audio: np.ndarray) -> np.ndarray:
    """Quantize float [-1, 1] audio to 16-bit PCM. We do it ourselves (rather
    than letting the WAV library do it) so the rounding is explicit and
    identical everywhere."""
    return np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), to_int16(audio), sample_rate, subtype="PCM_16", format="WAV")


# ---------------------------------------------------------------------------
# Reproducibility: seeds and the dataset layout
# ---------------------------------------------------------------------------
# Each sample gets its own random stream derived from (global seed, index).
# Whichever worker process renders sample #17, it always gets the same stream,
# so `--workers 8` produces byte-identical files to `--workers 1`.
# Stream 0 = per-sample audio randomness, stream 1 = dataset layout.
def sample_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0, index)))


def layout_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))


def plan_positions(count: int, max_fret: int, seed: int) -> list[tuple[int, int]]:
    """Choose a (string, fret) for every sample, as evenly as possible.

    Why not just pick randomly per sample? Random picks are uniform only on
    average; with 200 samples over 120 positions some would appear 0 times and
    others 4+. A model then quietly learns the dataset's accidental bias.
    Instead we tile all positions `count // n` times, add a random subset for
    the remainder (so every position appears k or k+1 times), and shuffle.
    """
    positions = tuning.all_positions(max_fret)
    rng = layout_rng(seed)
    reps, remainder = divmod(count, len(positions))
    plan = positions * reps
    extra = rng.choice(len(positions), size=remainder, replace=False)
    plan += [positions[i] for i in sorted(extra)]
    return [plan[i] for i in rng.permutation(count)]


LABEL_COLUMNS = [
    "file",
    "string",
    "fret",
    "midi",
    "note",
    "freq_hz",
    # randomized parameters
    "pick_strength",
    "pluck_position",
    "decay_scale",
    "detune_cents",
    "noise_db",
    # derived values (handy for analysis / debugging)
    "actual_freq_hz",
    "t60_s",
    "loop_lowpass",
    "loop_gain",
    "delay_int",
    "allpass_coef",
    "comb_delay",
    "peak_dbfs",
]


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def label_row(file: str, params: NoteParams, phys: dict) -> dict:
    midi = tuning.string_fret_to_midi(params.string, params.fret)
    row = {
        "file": file,
        "string": params.string,
        "fret": params.fret,
        "midi": midi,
        "note": tuning.midi_to_name(midi),
        "freq_hz": phys["nominal_hz"],
        "actual_freq_hz": phys["actual_hz"],
    }
    row.update({k: v for k, v in asdict(params).items() if k not in ("string", "fret")})
    row.update({k: phys[k] for k in LABEL_COLUMNS if k in phys and k not in row})
    return {k: _fmt(row[k]) for k in LABEL_COLUMNS}


def _render_job(job: tuple) -> dict:
    """Worker entry point: render + write one sample, return its label row.
    Top-level function so it can be sent to worker processes."""
    index, string, fret, seed, sample_rate, duration, audio_dir = job
    rng = sample_rng(seed, index)
    params = random_params(rng, string, fret)
    audio, phys = render_note(params, sample_rate, duration, rng)
    name = f"{index:06d}.wav"
    write_wav(Path(audio_dir) / name, audio, sample_rate)
    return label_row(f"audio/{name}", params, phys)


def write_labels(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    count: int,
    out_dir: Path,
    seed: int,
    sample_rate: int = 22050,
    duration: float = 2.0,
    max_fret: int = 19,
    workers: int = 1,
    progress: bool = True,
    cli_args: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_positions(count, max_fret, seed)
    jobs = [
        (i + 1, s, f, seed, sample_rate, duration, str(audio_dir))
        for i, (s, f) in enumerate(plan)
    ]

    start = time.perf_counter()
    bar = dict(total=count, desc="rendering", unit="note", disable=not progress)
    if workers <= 1:
        rows = [_render_job(job) for job in tqdm(jobs, **bar)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map() returns results in submission order, so labels.csv rows
            # are in index order no matter which worker finished first.
            rows = list(tqdm(pool.map(_render_job, jobs, chunksize=8), **bar))
    elapsed = time.perf_counter() - start

    write_labels(out_dir / "labels.csv", rows)

    metadata = {
        "synth": "electro_synth",
        "synth_version": SYNTH_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cli_args": cli_args
        or {
            "count": count,
            "out": str(out_dir),
            "seed": seed,
            "sample_rate": sample_rate,
            "duration": duration,
            "max_fret": max_fret,
            "workers": workers,
        },
        "seed": seed,
        "count": count,
        "sample_rate": sample_rate,
        "duration_s": duration,
        "max_fret": max_fret,
        "audio_format": "WAV, mono, PCM 16-bit",
        "string_numbering": "1 = high E (E4) ... 6 = low E (E2)",
        "open_string_midi": tuning.OPEN_STRING_MIDI,
        "param_ranges": PARAM_RANGES,
        "string_timbre": {s: asdict(t) for s, t in STRING_TIMBRE.items()},
        "generation_seconds": round(elapsed, 3),
        "numba": NUMBA_AVAILABLE,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def generate_preview(
    out_dir: Path, sample_rate: int, duration: float, max_fret: int, seed: int
) -> list[Path]:
    """Render a small listening set: every open string plus frets 5, 12 and
    max_fret, with mid-range (non-random) playing parameters."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frets = sorted({0, min(5, max_fret), min(12, max_fret), max_fret})
    rows, paths = [], []
    for i, (string, fret) in enumerate(
        [(s, f) for s in sorted(tuning.STRINGS, reverse=True) for f in frets], start=1
    ):
        params = default_params(string, fret)
        audio, phys = render_note(params, sample_rate, duration, sample_rng(seed, i))
        midi = tuning.string_fret_to_midi(string, fret)
        name = f"s{string}_f{fret:02d}_{tuning.midi_to_name(midi).replace('#', 's')}.wav"
        write_wav(out_dir / name, audio, sample_rate)
        rows.append(label_row(name, params, phys))
        paths.append(out_dir / name)
    write_labels(out_dir / "labels.csv", rows)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Electro Synth: generate labeled single-note guitar audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--count", type=int, help="number of notes to generate")
    p.add_argument("--out", type=str, help="output dataset folder, e.g. data/notes_v1")
    p.add_argument("--seed", type=int, default=42, help="global random seed")
    p.add_argument("--sample-rate", type=int, default=22050, help="Hz")
    p.add_argument("--duration", type=float, default=2.0, help="seconds per note")
    p.add_argument("--max-fret", type=int, default=19, help="highest fret to sample")
    p.add_argument(
        "--workers", type=int, default=1, help="parallel processes (output is identical)"
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help=f"write open strings + frets 5/12/max to {DEFAULT_PREVIEW_DIR.relative_to(PROJECT_ROOT)}/ "
        "(or --out) for listening",
    )
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.max_fret <= 24:
        parser.error("--max-fret must be between 0 and 24")
    if args.sample_rate < 8000:
        parser.error("--sample-rate must be at least 8000")
    if args.duration <= 0:
        parser.error("--duration must be positive")

    if args.preview:
        out = Path(args.out) if args.out else DEFAULT_PREVIEW_DIR
        paths = generate_preview(out, args.sample_rate, args.duration, args.max_fret, args.seed)
        print(f"Wrote {len(paths)} preview notes to {out}")
        return 0

    if args.count is None or args.out is None:
        parser.error("--count and --out are required (unless --preview)")
    if args.count <= 0:
        parser.error("--count must be positive")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    meta = generate_dataset(
        count=args.count,
        out_dir=Path(args.out),
        seed=args.seed,
        sample_rate=args.sample_rate,
        duration=args.duration,
        max_fret=args.max_fret,
        workers=workers,
        progress=not args.quiet,
        cli_args=vars(args),
    )
    secs = meta["generation_seconds"]
    print(
        f"Wrote {args.count} notes to {args.out} in {secs:.2f}s "
        f"({1000 * secs / args.count:.1f} ms/note, workers={workers}, numba={NUMBA_AVAILABLE})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

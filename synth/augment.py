"""Recording-chain augmentation: make clean synth notes sound "recorded".

Used by `electro_synth.py --augment`. Every parameter is drawn at random
per note and written to labels.csv (columns starting with `aug_`).

Why augment?
------------
A real note reaches our model through a chain: string -> guitar body ->
room -> microphone -> phone's automatic gain control -> lossy codec (AAC).
Every link changes the sound, and the changes differ from phone to phone and
room to room. Measured on our first real recording (tools/compare_real.py),
the chain boosted 150-400 Hz by ~15 dB, scooped ~800 Hz by ~15 dB, and
removed most energy above 5 kHz, which made the fundamental the strongest
partial in only 63% of notes. A model trained on clean synth notes has never
seen anything like that.

We can't model *the* recording chain (it's different every time), so we
randomize over a wide range of plausible chains. The model then can't rely
on any specific coloration and must use what survives all of them. This is
called "domain randomization".

The chain (applied in physical order)
-------------------------------------
1. room reverb:   the note's reflections off walls, from a synthetic impulse
                  response (IR): a few early echoes + an exponentially
                  decaying noise tail. RT60 = time for the tail to fade 60 dB.
2. noise:         room tone / mic self-noise (colored noise), sometimes mains
                  hum (50/60 Hz + harmonics), sometimes handling clicks.
3. mic EQ:        low cut (small mics lose lows), high cut (small mics and
                  codecs lose highs), a broad low-shelf tilt (+-12 dB), and 3
                  random bell filters (resonances of the mic, the phone's
                  body, the player's position).
4. compression:   phones apply automatic gain control: loud parts are turned
                  down, so notes look like they decay slower and noise gets
                  pulled up.
5. codec:         AAC encode + decode at a random bitrate (what a phone's voice
                  memo app does), via macOS's `afconvert`. Elsewhere, MP3
                  through soundfile stands in for it.
Finally the note's original peak level is restored, so augmentation doesn't
change loudness statistics.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

AFCONVERT = shutil.which("afconvert")
CODEC_BITRATES = (32, 48, 64, 96, 128)  # kbit/s

P_REVERB = 0.7
P_HUM = 0.2
P_CLICKS = 0.3
P_COMPRESSION = 0.8
P_CODEC = 0.8


def augment_rng(seed: int, index: int) -> np.random.Generator:
    """Stream 5: separate from the note's own randomness, so the same note can
    be rendered with or without augmentation."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(5, index)))


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Uniform on a log scale: as likely to be in 30-60 Hz as in 120-240 Hz.
    The natural choice for frequencies, which we hear logarithmically."""
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def draw_params(rng: np.random.Generator) -> dict:
    """All random choices, drawn up front in a fixed order (reproducible)."""
    p = {
        "aug_reverb": bool(rng.random() < P_REVERB),
        "aug_rt60_s": float(rng.uniform(0.15, 0.9)),
        "aug_reverb_wet_db": float(rng.uniform(-24.0, -6.0)),
        "aug_noise_db": float(rng.uniform(-65.0, -35.0)),
        "aug_noise_color": float(rng.uniform(0.0, 0.95)),  # 0 = white hiss, ->1 = rumble
        "aug_hum_db": float(rng.uniform(-60.0, -40.0)) if rng.random() < P_HUM else None,
        "aug_hum_hz": float(rng.choice([50.0, 60.0])),
        "aug_clicks": int(rng.integers(1, 4)) if rng.random() < P_CLICKS else 0,
        "aug_lowcut_hz": log_uniform(rng, 30.0, 250.0),
        "aug_highcut_hz": log_uniform(rng, 3000.0, 11000.0),
        # broad tilt: everything below ~f boosted or cut (small mics close to
        # a guitar often add a big low-mid "boom"; others sound thin)
        "aug_shelf_hz": log_uniform(rng, 150.0, 800.0),
        "aug_shelf_db": float(rng.uniform(-12.0, 12.0)),
    }
    for i in (1, 2, 3):
        p[f"aug_eq{i}_hz"] = log_uniform(rng, 100.0, 6000.0)
        p[f"aug_eq{i}_db"] = float(rng.uniform(-15.0, 15.0))
        p[f"aug_eq{i}_q"] = float(rng.uniform(0.4, 2.5))
    p.update(
        {
            "aug_compression": bool(rng.random() < P_COMPRESSION),
            "aug_comp_threshold_db": float(rng.uniform(-30.0, -6.0)),
            "aug_comp_ratio": float(rng.uniform(1.5, 8.0)),
            "aug_comp_ms": float(rng.uniform(5.0, 150.0)),
            "aug_codec_kbps": int(rng.choice(CODEC_BITRATES)) if rng.random() < P_CODEC else 0,
        }
    )
    return p


# --- 1. reverb ------------------------------------------------------------------
def room_ir(rng: np.random.Generator, rt60: float, sr: int) -> np.ndarray:
    n = int(min(1.2 * rt60, 1.0) * sr)
    t = np.arange(n) / sr
    tail = rng.normal(0.0, 1.0, n) * np.exp(-6.9 * t / rt60)  # -60 dB (e^-6.9) at t = rt60
    tail = lfilter([0.5], [1.0, -0.5], tail)  # walls absorb highs: darker tail
    ir = np.zeros(n)  # the direct sound is added by the caller ("dry" signal)
    for _ in range(4):  # early reflections: first bounces off nearby surfaces
        ir[int(rng.uniform(0.003, 0.025) * sr)] += rng.uniform(0.3, 0.7) * rng.choice([-1.0, 1.0])
    ir += 0.3 * tail / (np.sqrt(np.sum(tail**2)) + 1e-12)
    return ir / (np.sqrt(np.sum(ir**2)) + 1e-12)


# --- 3. EQ ----------------------------------------------------------------------
def low_shelf(y: np.ndarray, sr: int, f: float, gain_db: float) -> np.ndarray:
    """Shelf EQ (Audio EQ Cookbook): everything below ~f raised or lowered by gain_db."""
    A = 10.0 ** (gain_db / 40.0)
    w = 2.0 * math.pi * f / sr
    alpha = math.sin(w) / 2.0 * math.sqrt(2.0)  # shelf slope S = 1
    cw, sa = math.cos(w), 2.0 * math.sqrt(A) * alpha
    b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
    a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    return lfilter(b, a, y)


def peaking(y: np.ndarray, sr: int, f: float, gain_db: float, q: float) -> np.ndarray:
    """Bell EQ (Audio EQ Cookbook peaking filter)."""
    f = min(f, 0.45 * sr)
    A = 10.0 ** (gain_db / 40.0)
    w = 2.0 * math.pi * f / sr
    alpha = math.sin(w) / (2.0 * q)
    b = [1 + alpha * A, -2 * math.cos(w), 1 - alpha * A]
    a = [1 + alpha / A, -2 * math.cos(w), 1 - alpha / A]
    return lfilter(b, a, y)


# --- 4. compression ---------------------------------------------------------------
def compress(y: np.ndarray, sr: int, threshold_db: float, ratio: float, time_ms: float) -> np.ndarray:
    """Turn down whatever is louder than the threshold.

    1. Follow the signal's level: smooth y^2 with a one-pole lowpass whose
       time constant is time_ms (fast = grabs the attack, slow = gentle "AGC").
    2. Above the threshold, every 1 dB of input above it gives only 1/ratio dB
       of output above it: e.g. ratio 4 turns +12 dB into +3 dB.
    The threshold is relative to the note's peak level.
    """
    coef = math.exp(-1.0 / (time_ms * 1e-3 * sr))
    power = lfilter([1.0 - coef], [1.0, -coef], y * y)
    level_db = 10.0 * np.log10(power + 1e-20)
    threshold = level_db.max() + threshold_db
    over = np.maximum(level_db - threshold, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    return y * 10.0 ** (gain_db / 20.0)


# --- 5. codec ---------------------------------------------------------------------
def codec_roundtrip(y: np.ndarray, sr: int, kbps: int) -> tuple[np.ndarray, str]:
    """Encode to a lossy format and decode back. Returns (audio, codec name).

    Lossy codecs save bits by throwing away what a psychoacoustic model says
    you won't hear: quiet partials next to loud ones, high frequencies at low
    bitrates. To a model those holes and smears are very visible.
    """
    # Phones record at 44.1/48 kHz, and AAC behaves differently at different
    # rates (at 22.05 kHz mono it doesn't even accept more than 64 kbit/s), so
    # we go up to 44.1 kHz, encode/decode there, and come back down.
    codec_sr = 44100
    up = resample_poly(y, codec_sr, sr)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        sf.write(str(src), up.astype(np.float32), codec_sr, subtype="FLOAT")
        if AFCONVERT:
            enc, dec = Path(tmp) / "enc.m4a", Path(tmp) / "dec.wav"
            subprocess.run([AFCONVERT, "-f", "m4af", "-d", "aac", "-b", str(kbps * 1000), str(src), str(enc)],
                           check=True, capture_output=True)
            subprocess.run([AFCONVERT, "-f", "WAVE", "-d", "LEF32", str(enc), str(dec)],
                           check=True, capture_output=True)
            name = f"aac{kbps}"
        else:  # portable stand-in
            enc, dec = Path(tmp) / "enc.mp3", None
            sf.write(str(enc), up.astype(np.float32), codec_sr, format="MP3", subtype="MPEG_LAYER_III",
                     compression_level=1.0 - (kbps - 32) / 128)
            name = f"mp3~{kbps}"
        out, _ = sf.read(str(dec if dec else enc), dtype="float64")
    if out.ndim > 1:
        out = out.mean(axis=1)
    out = resample_poly(out, sr, codec_sr)
    fixed = np.zeros_like(y)  # same length as the input (decoders may pad)
    fixed[: min(len(y), len(out))] = out[: len(y)]
    return fixed, name


# --- the whole chain -------------------------------------------------------------
def augment_note(audio: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Pass one note through a random recording chain. Returns (audio, logged params)."""
    p = draw_params(rng)
    n = len(audio)
    peak_in = np.max(np.abs(audio)) + 1e-12
    y = audio.copy()

    if p["aug_reverb"]:
        wet = fftconvolve(y, room_ir(rng, p["aug_rt60_s"], sr))[:n]
        y = y + 10.0 ** (p["aug_reverb_wet_db"] / 20.0) * wet / (np.max(np.abs(wet)) + 1e-12) * peak_in

    # noise: room tone, hum, clicks (levels relative to the note's peak)
    noise = lfilter([1.0 - p["aug_noise_color"]], [1.0, -p["aug_noise_color"]], rng.normal(0.0, 1.0, n))
    noise *= 10.0 ** (p["aug_noise_db"] / 20.0) * peak_in / (np.sqrt(np.mean(noise**2)) + 1e-12)
    y = y + noise
    if p["aug_hum_db"] is not None:
        t = np.arange(n) / sr
        hum = sum(np.sin(2 * math.pi * k * p["aug_hum_hz"] * t + rng.uniform(0, 2 * math.pi)) / k for k in (1, 2, 3))
        y = y + 10.0 ** (p["aug_hum_db"] / 20.0) * peak_in * hum / np.max(np.abs(hum))
    for _ in range(p["aug_clicks"]):
        start = int(rng.uniform(0.05, 0.95) * n)
        length = int(rng.uniform(0.002, 0.010) * sr)
        burst = rng.normal(0.0, 1.0, length) * np.hanning(length)
        y[start : start + length] += 10.0 ** (rng.uniform(-30.0, -10.0) / 20.0) * peak_in * burst[: n - start]

    # mic / phone EQ
    y = sosfilt(butter(2, p["aug_lowcut_hz"], "highpass", fs=sr, output="sos"), y)
    if p["aug_highcut_hz"] < 0.45 * sr:
        y = sosfilt(butter(2, p["aug_highcut_hz"], "lowpass", fs=sr, output="sos"), y)
    y = low_shelf(y, sr, p["aug_shelf_hz"], p["aug_shelf_db"])
    for i in (1, 2, 3):
        y = peaking(y, sr, p[f"aug_eq{i}_hz"], p[f"aug_eq{i}_db"], p[f"aug_eq{i}_q"])

    if p["aug_compression"]:
        y = compress(y, sr, p["aug_comp_threshold_db"], p["aug_comp_ratio"], p["aug_comp_ms"])

    codec = "none"
    if p["aug_codec_kbps"]:
        y = y * 0.9 / (np.max(np.abs(y)) + 1e-12)  # encode at a healthy level
        y, codec = codec_roundtrip(y, sr, p["aug_codec_kbps"])

    y *= peak_in / (np.max(np.abs(y)) + 1e-12)  # restore the original peak level
    p["aug_codec"] = codec
    p["aug_hum_db"] = "" if p["aug_hum_db"] is None else p["aug_hum_db"]
    return y, p

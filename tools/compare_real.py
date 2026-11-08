"""Diagnose the sim-to-real gap: how do real notes differ from synth notes?

    python tools/compare_real.py --real data/real_session1 --out runs/real_vs_synth

For every real note, renders a synth "twin" at the same (string, fret) with
the current synth (mid-range playing parameters, optionally --augment), and
measures both the same way. Writes to --out:

  side_by_side.png        CQT of real (top) vs synth (bottom), one position per string
  harmonic_profiles.png   average spectrum per string, pitch-aligned: x = semitones above
                          the fundamental, so harmonic k always sits at 12*log2(k)
  coloration.png          average spectrum vs absolute frequency (Hz), real minus synth:
                          the combined effect of guitar body + room + mic + codec
  report.md               the numbers per string, real vs synth

Why compare per string, pitch-aligned?
--------------------------------------
Averaging raw spectra of notes with different pitches just smears their
harmonics together. Shifting each note so its fundamental sits at 0 lines up
the harmonics: the average then shows the *shape* of a string's sound (how
strong the fundamental is, how fast harmonics fade), which is exactly what the
model must use to tell strings apart.

Leakage warning
---------------
real_session1 is the exam. Looking at it to set a few *global* synth
properties (e.g. "phone mics cut the lows") is a mild form of fitting to the
test set: the exam score afterwards is a little optimistic. Keep tuning coarse
(ranges, not exact fitted values), and use a second recording as a dev set
for anything finer.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "synth"))
import augment as aug  # noqa: E402
import electro_synth as es  # noqa: E402
import spectrogram  # noqa: E402
import tuning  # noqa: E402

SR = 22050
ANALYSIS = (0.05, 0.55)  # seconds: after the pluck transient, while partials are strong
N_HARMONICS = 15
FFT_SIZE = 1 << 16


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def spectrum_db(audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    seg = audio[int(ANALYSIS[0] * sr) : int(ANALYSIS[1] * sr)]
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), FFT_SIZE))
    return np.fft.rfftfreq(FFT_SIZE, 1 / sr), 20 * np.log10(mag + 1e-12)


def find_partials(freqs, spec_db, f0_nominal: float, sr: int = SR) -> list[tuple[int, float, float]]:
    """(k, frequency, level dB) of each harmonic k, searched near where it should be.

    Stiff strings push partials sharp, more so for higher k, so we search in a
    window that follows the partials already found: predict partial k from
    the inharmonicity fitted on partials 1..k-1.
    """
    found = []
    f0, B = f0_nominal, 0.0
    for k in range(1, N_HARMONICS + 1):
        predicted = k * f0 * np.sqrt(1 + max(B, 0.0) * k * k)
        if predicted > 0.45 * sr:
            break
        half = 0.35 * f0  # never reach halfway to the neighbouring harmonic
        band = (freqs > predicted - half) & (freqs < predicted + half)
        if not band.any():
            break
        i = np.flatnonzero(band)[np.argmax(spec_db[band])]
        found.append((k, float(freqs[i]), float(spec_db[i])))
        if len(found) >= 3:
            f0, B = fit_inharmonicity(found)
    return found


def fit_inharmonicity(partials) -> tuple[float, float]:
    """Fit f_k = k * f0 * sqrt(1 + B k^2).

    Squaring: (f_k / k)^2 = f0^2 + f0^2 * B * k^2, a straight line in k^2,
    so ordinary least squares gives f0 (intercept) and B (slope / intercept).
    B is the string's inharmonicity: 0 for an ideal flexible string; real
    steel strings are ~1e-5 to 1e-3 (stiffer and shorter = bigger B).
    """
    k = np.array([p[0] for p in partials], float)
    y = (np.array([p[1] for p in partials]) / k) ** 2
    slope, intercept = np.polyfit(k * k, y, 1)
    return float(np.sqrt(max(intercept, 1e-9))), float(slope / max(intercept, 1e-9))


def envelope_db(audio: np.ndarray, sr: int = SR, frame: int = 512) -> np.ndarray:
    n = len(audio) // frame
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-12)


def decay_t60(audio: np.ndarray, sr: int = SR, frame: int = 512) -> float:
    """Time for the level to drop 60 dB, extrapolated from a straight-line fit
    of the dB envelope between -5 and -25 dB below the peak (the part that's
    above the noise floor in both real and synth notes)."""
    env = envelope_db(audio, sr, frame)
    p = int(np.argmax(env))
    after = env[p:] - env[p]
    sel = np.flatnonzero((after < -5) & (after > -25))
    if len(sel) < 4:
        return float("nan")
    t = sel * frame / sr
    slope = np.polyfit(t, after[sel], 1)[0]  # dB per second (negative)
    return float(-60.0 / slope) if slope < 0 else float("nan")


def noise_floor_db(audio: np.ndarray, sr: int = SR, frame: int = 512) -> float:
    """Quietest 10% of non-silent frames, relative to the peak frame."""
    env = envelope_db(audio, sr, frame)
    live = env[env > -150]
    return float(np.percentile(live, 10) - env.max())


def analyze(audio: np.ndarray, midi: int) -> dict:
    freqs, spec = spectrum_db(audio)
    partials = find_partials(freqs, spec, tuning.midi_to_hz(midi))
    levels = {k: lvl for k, _, lvl in partials}
    top = max(levels.values())
    good = [p for p in partials if p[2] > top - 45]  # well above the noise
    B = fit_inharmonicity(good)[1] if len(good) >= 4 else float("nan")
    return {
        "fundamental_vs_strongest_db": levels[1] - top,
        "h8_vs_h1_db": levels.get(8, np.nan) - levels[1],
        "profile_db": [levels.get(k, np.nan) - top for k in range(1, N_HARMONICS + 1)],
        "inharmonicity_B": B,
        "t60_s": decay_t60(audio),
        "noise_floor_db": noise_floor_db(audio),
    }


def third_octave_spectrum(audio: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Average power in 1/3-octave bands, in dB, normalized to total power."""
    seg = audio[: int(1.0 * SR)]
    p = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), FFT_SIZE)) ** 2
    f = np.fft.rfftfreq(FFT_SIZE, 1 / SR)
    out = []
    for c in centers:
        band = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        out.append(p[band].mean())
    out = np.array(out)
    return 10 * np.log10(out / out.sum() + 1e-20)


# ---------------------------------------------------------------------------
# Synth twins
# ---------------------------------------------------------------------------
def render_twin(string: int, fret: int, index: int, augment: bool) -> np.ndarray:
    params = es.default_params(string, fret)
    rng = es.sample_rng(777, index)
    audio, _ = es.render_note(params, SR, 2.0, rng)
    if augment:
        audio, _ = aug.augment_note(audio, SR, aug.augment_rng(777, index))
    return audio * 10 ** (-3 / 20) / (np.abs(audio).max() + 1e-12)  # same level as the slicer


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def nanmed(values) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare real recorded notes with synth twins.")
    p.add_argument("--real", default="data/real_session1")
    p.add_argument("--out", default="runs/real_vs_synth")
    p.add_argument("--augment", action="store_true", help="pass synth twins through the recording-chain augmentation")
    args = p.parse_args(argv)

    real_dir, out = Path(args.real), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(real_dir / "labels.csv")))
    rows = [r for r in rows if not r.get("flag")]  # skip notes whose label is in doubt

    real_audio, synth_audio, real_m, synth_m = [], [], [], []
    for i, r in enumerate(rows):
        s, f, midi = int(r["string"]), int(r["fret"]), int(r["midi"])
        a, _ = sf.read(str(real_dir / r["file"]))
        b = render_twin(s, f, i, args.augment)
        real_audio.append(a)
        synth_audio.append(b)
        real_m.append({"string": s, "fret": f, "midi": midi, **analyze(a, midi)})
        synth_m.append({"string": s, "fret": f, "midi": midi, **analyze(b, midi)})

    # --- 1. side by side CQTs: first real note on each string ---------------
    picks = []
    for s in (6, 5, 4, 3, 2, 1):
        idx = next(i for i, m in enumerate(real_m) if m["string"] == s)
        picks.append(idx)
    real_specs = [spectrogram.cqt_db(real_audio[i], SR) for i in picks]
    synth_specs = [spectrogram.cqt_db(synth_audio[i], SR) for i in picks]
    vmax = max(x.max() for x in real_specs + synth_specs)
    ticks, names = spectrogram.note_ticks()
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), sharey=True)
    for col, i in enumerate(picks):
        m = real_m[i]
        for row, (spec, label) in enumerate([(real_specs[col], "REAL"), (synth_specs[col], "synth")]):
            ax = axes[row, col]
            ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=-90, vmax=vmax,
                      extent=[0, 2.0, -0.5, spec.shape[0] - 0.5], interpolation="nearest")
            ax.axhline(spectrogram.midi_to_bin(m["midi"]), color="cyan", lw=0.6, ls="--", alpha=0.7)
            ax.set_yticks(ticks)
            ax.set_yticklabels(names, fontsize=7)
            ax.set_title(f"{label}: s{m['string']} f{m['fret']} ({tuning.midi_to_name(m['midi'])})", fontsize=9)
            ax.set_xlabel("time (s)", fontsize=8)
    axes[0, 0].set_ylabel("pitch (note)")
    axes[1, 0].set_ylabel("pitch (note)")
    fig.suptitle("Real (top) vs synth twin (bottom), dashed = labeled fundamental", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "side_by_side.png", dpi=100)
    plt.close(fig)

    # --- 2. pitch-aligned average spectrum (harmonic profile) per string ------
    k = np.arange(1, N_HARMONICS + 1)
    x = 12 * np.log2(k)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for ax, s in zip(axes.flat, (6, 5, 4, 3, 2, 1)):
        for ms, style, label in [(real_m, "o-", "real"), (synth_m, "s--", "synth")]:
            prof = np.array([m["profile_db"] for m in ms if m["string"] == s])
            ax.plot(x, np.nanmean(prof, axis=0), style, label=f"{label} (n={len(prof)})")
        ax.set_xticks(x[:10])
        ax.set_xticklabels([f"{kk}" for kk in k[:10]], fontsize=8)
        ax.set_xlabel("harmonic number (spaced like the CQT: semitones above f0)")
        ax.set_title(f"string {s} ({tuning.midi_to_name(tuning.OPEN_STRING_MIDI[s])})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("level vs strongest partial (dB)")
    axes[1, 0].set_ylabel("level vs strongest partial (dB)")
    fig.suptitle("Average harmonic profile per string, 0.05-0.55 s", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "harmonic_profiles.png", dpi=100)
    plt.close(fig)

    # --- 3. coloration: average spectrum in Hz, real - synth -------------------
    centers = 63 * 2 ** (np.arange(0, 22) / 3)  # 63 Hz .. ~8 kHz, 1/3 octaves
    centers = centers[centers < SR / 2 / 2 ** (1 / 6)]
    real_avg = np.mean([third_octave_spectrum(a, centers) for a in real_audio], axis=0)
    synth_avg = np.mean([third_octave_spectrum(b, centers) for b in synth_audio], axis=0)
    diff = real_avg - synth_avg
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.semilogx(centers, real_avg, "o-", label="real")
    ax1.semilogx(centers, synth_avg, "s--", label="synth")
    ax1.set_ylabel("share of power (dB)")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    ax2.semilogx(centers, diff - np.median(diff), "o-", color="C3")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("real - synth (dB, median removed)")
    ax2.set_xlabel("frequency (Hz)")
    ax2.grid(alpha=0.3, which="both")
    ax1.set_title("Average spectrum over all notes (same positions). The difference = body + room + mic + codec")
    fig.tight_layout()
    fig.savefig(out / "coloration.png", dpi=100)
    plt.close(fig)

    # --- 4. report ---------------------------------------------------------------
    keys = [
        ("fundamental_vs_strongest_db", "fundamental vs strongest partial (dB)", "{:+.1f}"),
        ("h8_vs_h1_db", "harmonic 8 vs fundamental (dB)", "{:+.1f}"),
        ("t60_s", "decay T60 (s)", "{:.2f}"),
        ("noise_floor_db", "noise floor vs peak (dB)", "{:+.1f}"),
        ("inharmonicity_B", "inharmonicity B (x1e-4)", "{:.2f}"),
    ]
    lines = [f"# Real vs synth ({real_dir}, {len(rows)} notes, augment={args.augment})", ""]
    lines.append("Medians per string. Real | synth.\n")
    lines.append("| measure | " + " | ".join(f"string {s}" for s in (6, 5, 4, 3, 2, 1)) + " |")
    lines.append("|---|" + "---|" * 6)
    for key, label, fmt in keys:
        cells = []
        for s in (6, 5, 4, 3, 2, 1):
            scale = 1e4 if key == "inharmonicity_B" else 1
            rv = nanmed([m[key] for m in real_m if m["string"] == s]) * scale
            sv = nanmed([m[key] for m in synth_m if m["string"] == s]) * scale
            cells.append(f"{fmt.format(rv)} \\| {fmt.format(sv)}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Fundamental is the strongest partial in "
              f"{np.mean([m['fundamental_vs_strongest_db'] > -0.01 for m in real_m]) * 100:.0f}% of real notes vs "
              f"{np.mean([m['fundamental_vs_strongest_db'] > -0.01 for m in synth_m]) * 100:.0f}% of synth notes.",
              "", "Coloration, real - synth, 1/3 octave bands (dB, median removed):", ""]
    lines.append("| Hz | " + " | ".join(f"{c:.0f}" for c in centers) + " |")
    lines.append("|---|" + "---|" * len(centers))
    lines.append("| dB | " + " | ".join(f"{d - np.median(diff):+.0f}" for d in diff) + " |")
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    print(report)
    print(f"-> {out}/side_by_side.png, harmonic_profiles.png, coloration.png, report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

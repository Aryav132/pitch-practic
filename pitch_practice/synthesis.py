"""Synthetic singing with a known answer: used by the tests and by the
built-in demo (so no copyrighted audio is ever needed)."""

import numpy as np


def cents_to_hz(c: float) -> float:
    return 440.0 * 2 ** ((c - 6900.0) / 1200.0)


def voice_like(notes, sr: int = 24000, seed: int = 0) -> np.ndarray:
    """Melody that is a bit more voice-like than `tone`: 12 harmonics with a
    vowel-ish spectral tilt and formant bumps, 5.5 Hz vibrato (+/-30 c),
    soft attacks, and a little breath noise. Ground truth = the note list."""
    rng = np.random.default_rng(seed)
    out = []
    for c, d in notes:
        n = int(round(d * sr))
        if c is None:
            out.append(np.zeros(n, np.float32))
            continue
        t = np.arange(n) / sr
        f = cents_to_hz(c) * 2 ** (30 / 1200 * np.sin(2 * np.pi * 5.5 * t))
        ph = 2 * np.pi * np.cumsum(f) / sr
        f0 = cents_to_hz(c)
        y = np.zeros(n)
        for k in range(1, 13):
            fk = k * f0
            formant = 1 + 2.5 * np.exp(-((fk - 700) / 250) ** 2) + 1.5 * np.exp(-((fk - 1200) / 300) ** 2)
            y += formant / k * np.sin(k * ph)
        env = np.minimum(1, t / 0.04) * np.minimum(1, (d - t) / 0.06)
        y = y / np.max(np.abs(y)) * env + 0.01 * rng.standard_normal(n) * env
        out.append(y.astype(np.float32))
    return 0.3 * np.concatenate(out)

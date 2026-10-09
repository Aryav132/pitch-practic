"""Synthetic signals with known ground truth, shared by all tests."""

import numpy as np

SR = 24000


def cents_to_hz(c: float) -> float:
    return 440.0 * 2 ** ((c - 6900.0) / 1200.0)


def tone(freq_hz, dur_s: float, sr: int = SR, harmonics: int = 1, amp: float = 0.3) -> np.ndarray:
    """Tone at a fixed or time-varying frequency.

    `freq_hz` may be a scalar or a per-sample array. Phase is the running sum
    of frequency, so frequency changes are click-free. harmonics > 1 gives a
    1/k spectrum, closer to a voice than a pure sine.
    """
    n = int(round(dur_s * sr))
    f = np.broadcast_to(np.asarray(freq_hz, dtype=float), (n,))
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))
    y = y / np.max(np.abs(y))
    return (amp * y).astype(np.float32)


def silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(dur_s * sr)), dtype=np.float32)


def noise(dur_s: float, amp: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(round(dur_s * sr)))).astype(np.float32)


def melody_track(notes, hop_s: float = 0.01):
    """PitchTrack straight from a note list, skipping audio + pYIN.

    notes: [(cents or None for silence, duration_s), ...]. Use it when a test
    is about alignment/scoring logic and must not depend on tracker noise.
    """
    from pitch_practice.pitch import PitchTrack

    cents = np.concatenate([np.full(int(round(d / hop_s)), np.nan if c is None else c)
                            for c, d in notes])
    f0 = 440.0 * 2 ** ((cents - 6900.0) / 1200.0)
    return PitchTrack(times=np.arange(len(cents)) * hop_s, f0_hz=f0,
                      voiced_prob=np.where(np.isnan(cents), 0.0, 1.0))


def melody_audio(notes, sr: int = SR, harmonics: int = 5) -> np.ndarray:
    """Same note list rendered as audio, for end-to-end tests through pYIN."""
    return np.concatenate([silence(d, sr) if c is None else tone(cents_to_hz(c), d, sr, harmonics)
                           for c, d in notes])


def shift(notes, cents: float):
    return [(None if c is None else c + cents, d) for c, d in notes]


# A 10 s phrase-structured melody: no two adjacent notes equal, rests between
# phrases, and no phrase repeated (so the correct lag is unique).
MELODY = [
    (6000, 0.5), (6200, 0.5), (6400, 0.5), (6500, 1.0), (None, 0.4),
    (6700, 0.5), (6500, 0.3), (6400, 0.7), (6200, 0.5), (None, 0.3),
    (6900, 0.6), (7100, 0.4), (6700, 0.8), (None, 0.5),
    (6400, 0.4), (6600, 0.4), (6300, 0.6), (6000, 1.1),
]


def voice_like(notes, sr: int = SR, seed: int = 0) -> np.ndarray:
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


def backing_track(dur_s: float, sr: int = SR, bpm: float = 100, seed: int = 1) -> np.ndarray:
    """Drums (kick + hi-hat), bass and a sustained chord pad, all synthetic."""
    rng = np.random.default_rng(seed)
    n = int(round(dur_s * sr))
    t = np.arange(n) / sr
    beat = 60 / bpm
    y = np.zeros(n)
    for b in np.arange(0, dur_s, beat):                       # kick on every beat
        i = int(b * sr); m = min(n - i, int(0.25 * sr)); tt = np.arange(m) / sr
        y[i:i + m] += 0.9 * np.sin(2 * np.pi * (50 + 60 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
    for b in np.arange(beat / 2, dur_s, beat):                # hi-hat off-beats
        i = int(b * sr); m = min(n - i, int(0.05 * sr))
        y[i:i + m] += 0.15 * rng.standard_normal(m) * np.exp(-np.arange(m) / sr * 80)
    roots = [cents_to_hz(c) for c in (3600, 4100, 3900, 3400)]  # C2 F2 Eb2 Bb1, 2 bars each
    bar = 4 * beat
    root = np.array([roots[int(x // (2 * bar)) % 4] for x in t])
    y += 0.25 * np.sin(2 * np.pi * np.cumsum(root) / sr)                   # bass
    for mult in (2, 2 * 2 ** (4 / 12), 2 * 2 ** (7 / 12)):                 # pad: triad
        ph = 2 * np.pi * np.cumsum(root * mult * 2) / sr
        y += 0.05 * sum(np.sin(k * ph) / k for k in range(1, 6))
    return (y / np.max(np.abs(y)) * 0.5).astype(np.float32)

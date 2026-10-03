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

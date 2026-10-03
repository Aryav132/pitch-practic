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

"""Audio snippets for "listen and compare": the singer, you, and the
reference melody as a clean tone. All cut from data the analysis already has."""

import numpy as np

from .scoring import Report


def clip(y: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    """y[t0:t1] in seconds (clamped), with 10 ms fades so it doesn't click."""
    a, b = max(0, int(t0 * sr)), min(len(y), int(t1 * sr))
    out = np.array(y[a:b], dtype=np.float32)
    fade = min(len(out) // 2, int(0.01 * sr))
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def take_window(r: Report, t0: float, t1: float, pad: float = 0.0) -> tuple[float, float]:
    """Song-time range -> the matching range in the take, via the alignment."""
    f = r.frames
    hop = f.times_s[1] - f.times_s[0]
    i0 = int(np.clip((t0 - r.ref_start_s) / hop, 0, len(f.times_s) - 1))
    i1 = int(np.clip((t1 - r.ref_start_s) / hop, 0, len(f.times_s) - 1))
    tt = f.take_times_s[i0:i1 + 1]
    tt = tt[np.isfinite(tt)]
    if tt.size == 0:   # section outside the aligned part: fall back to the lag
        return t0 - r.ref_start_s + r.lag_s - pad, t1 - r.ref_start_s + r.lag_s + pad
    return float(tt.min()) - pad, float(tt.max()) + pad


def melody_tone(r: Report, t0: float, t1: float, sr: int) -> np.ndarray:
    """The reference's pitch over [t0, t1] song time, as a soft clean tone
    (silence where the singer is silent): "what the right notes sound like"."""
    f = r.frames
    hop = f.times_s[1] - f.times_s[0]
    i0, i1 = int((t0 - r.ref_start_s) / hop), int((t1 - r.ref_start_s) / hop)
    cents = f.ref_cents[max(0, i0):max(0, i1)]
    n_per = int(round(hop * sr))
    hz = np.repeat(440.0 * 2 ** ((cents - 6900.0) / 1200.0), n_per)
    voiced = np.isfinite(hz)
    # Smooth the on/off gate (20 ms) so voiced edges don't click.
    k = max(1, int(0.02 * sr))
    gate = np.convolve(voiced.astype(float), np.ones(k) / k, mode="same")
    phase = 2 * np.pi * np.cumsum(np.where(voiced, hz, 0.0)) / sr
    y = (np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)) * gate
    return (0.25 * y / 1.4).astype(np.float32)

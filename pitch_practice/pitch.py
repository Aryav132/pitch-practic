"""Pitch (f0) extraction.

`PitchTracker` is the interface; `PyinTracker` is the classical
implementation (YIN candidates + HMM decoding, no learned weights).
Phase 3 adds a torchcrepe tracker behind the same interface.
"""

from dataclasses import dataclass
from typing import Protocol

import librosa
import numpy as np

from .config import AnalysisConfig


def hz_to_cents(f_hz: np.ndarray) -> np.ndarray:
    """Hz -> cents on the MIDI scale (A4 = 440 Hz = 6900 c). NaN stays NaN.

    Cents are log-frequency, so "50 c flat" means the same thing at any
    pitch, and a key change is a constant shift (an octave is exactly 1200).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(np.asarray(f_hz, dtype=float) / 440.0) + 6900.0


@dataclass(frozen=True)
class PitchTrack:
    times: np.ndarray        # frame centre times in seconds
    f0_hz: np.ndarray        # NaN where unvoiced
    voiced_prob: np.ndarray  # tracker's voicing confidence, 0..1

    @property
    def voiced(self) -> np.ndarray:
        return ~np.isnan(self.f0_hz)

    @property
    def cents(self) -> np.ndarray:
        return hz_to_cents(self.f0_hz)

    def __len__(self) -> int:
        return len(self.times)


class PitchTracker(Protocol):
    def track(self, y: np.ndarray, sr: int) -> PitchTrack: ...


class PyinTracker:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.cfg = config

    def track(self, y: np.ndarray, sr: int) -> PitchTrack:
        cfg = self.cfg
        if sr != cfg.sr:
            raise ValueError(f"expected {cfg.sr} Hz audio, got {sr} Hz")
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=cfg.fmin_hz, fmax=cfg.fmax_hz, sr=sr,
            frame_length=cfg.frame_length, hop_length=cfg.hop_length,
            resolution=cfg.pyin_resolution,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=cfg.hop_length)

        # Same framing as pYIN (centred), so frame i lines up in both arrays.
        rms = librosa.feature.rms(y=y, frame_length=cfg.frame_length,
                                  hop_length=cfg.hop_length)[0][: len(f0)]
        voiced = (
            voiced_flag
            & (voiced_prob >= cfg.min_voiced_prob)
            & (_db_rel_max(rms) > cfg.energy_gate_db)
        )
        voiced = drop_short_runs(voiced, round(cfg.min_voiced_run_s / cfg.hop_s))

        f0 = np.where(voiced, f0, np.nan)
        return PitchTrack(times=times, f0_hz=f0, voiced_prob=voiced_prob)


def _db_rel_max(rms: np.ndarray) -> np.ndarray:
    peak = rms.max()
    if peak <= 0:
        return np.full_like(rms, -np.inf)
    return 20.0 * np.log10(np.maximum(rms, 1e-12) / peak)


def drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Set True-runs shorter than min_len frames to False."""
    out = mask.copy()
    # Run boundaries: where the padded mask flips 0->1 (start) or 1->0 (end).
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for s, e in zip(starts, ends):
        if e - s < min_len:
            out[s:e] = False
    return out

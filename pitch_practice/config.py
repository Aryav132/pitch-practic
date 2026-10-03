"""All tunable settings in one place. Later steps add alignment/scoring fields here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    # Analysis rate for pitch tracking. Measured: at 16 kHz a C6 period is only
    # ~15 samples and YIN reads it ~11 c sharp; at 24 kHz the bias is ~1 c,
    # at the same speed (runtime is dominated by HMM decoding, not YIN).
    sr: int = 24000
    # 1536 samples = 64 ms window: ~4 periods of C2 (65 Hz), enough for YIN.
    frame_length: int = 1536
    # 240 samples = exactly 10 ms per frame -> 100 frames/s, 6000 frames/min.
    hop_length: int = 240

    # Search range is wider than the singing range (C2-C6 = 65-1047 Hz):
    # trackers fail at the edge of their search range. Measured: with
    # fmax=1050, a C6 came out 1900 c wrong.
    fmin_hz: float = 60.0
    fmax_hz: float = 1200.0
    # pYIN pitch grid in semitones. 0.1 = 10-cent bins (±5 c quantisation).
    # Measured: 0.05 is 2x slower for ~0.2 c gain; 0.02 broke tracking.
    pyin_resolution: float = 0.1

    # pYIN's HMM sometimes labels noise "voiced" with ~0.01 confidence.
    # Measured over 10 noise seeds: max 0.01. On a vibrato melody + noise the
    # median is 0.62 and 0.05 drops 2.4% of frames (mostly note transitions).
    min_voiced_prob: float = 0.05

    # Frames quieter than this (dB relative to the loudest frame) are forced
    # unvoiced. Targets breath noise and low-level separation residue.
    energy_gate_db: float = -40.0
    # Voiced runs shorter than this are treated as tracker noise and dropped.
    min_voiced_run_s: float = 0.05

    max_ref_s: float = 60.0
    max_take_s: float = 90.0

    @property
    def hop_s(self) -> float:
        return self.hop_length / self.sr

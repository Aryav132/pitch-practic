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

    # --- Key handling -------------------------------------------------------
    # "snapped": subtract the offset rounded to whole semitones (forgives
    #            transposition, keeps "40 c flat throughout" as an error)
    # "free":    subtract the exact offset (hides constant flatness)
    # "absolute": subtract nothing
    key_mode: str = "snapped"

    # --- Lag search (where in the take does the reference start?) -----------
    # A lag is only considered if at least this fraction of the reference's
    # voiced frames overlap voiced frames in the take.
    lag_min_coverage: float = 0.5
    # Per-pair deviation cap in the lag cost (see alignment._lag_cost).
    lag_cost_clip_cents: float = 300.0
    # Coarse pass every N frames (50 ms), then refine +/- N at full resolution.
    lag_coarse_step: int = 5
    # If the best lag at least 1 s away scores within this many cents of the
    # winner, the lag is reported as ambiguous (e.g. a repeated chorus).
    lag_ambiguity_margin_cents: float = 15.0

    # --- DTW ----------------------------------------------------------------
    # Path may stray this far from the diagonal implied by the lag.
    dtw_band_s: float = 1.5
    # Per-frame pitch cost is capped so one octave error can't drag the path.
    dtw_cost_cap_cents: float = 600.0
    # Cost of pairing a voiced frame with an unvoiced one.
    dtw_voicing_penalty: float = 300.0
    # Extra cost for each non-diagonal step. Stops DTW "chasing" small
    # vibrato/tuning wiggles with micro-warps that flatter a bad take.
    dtw_step_penalty: float = 50.0

    @property
    def hop_s(self) -> float:
        return self.hop_length / self.sr

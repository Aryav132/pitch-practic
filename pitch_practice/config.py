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
    # Was 0.05 s; raised after the same ~80 ms room-noise blip appeared in two
    # real recordings and became a fake "note" in scoring. An isolated sung
    # note under 100 ms is rare (fast runs sit inside continuous voicing).
    min_voiced_run_s: float = 0.10

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
    # Was 1.5 s. Real sing-along data (headphones) drifted <= 0.35 s; with
    # 1.5 s, an out-of-tune reference let DTW slide 1.5 s to pair notes by
    # pitch, turning pitch errors into fake timing errors.
    dtw_band_s: float = 0.75
    # Per-frame pitch cost is capped so one octave error can't drag the path.
    dtw_cost_cap_cents: float = 600.0
    # Cost of pairing a voiced frame with an unvoiced one.
    dtw_voicing_penalty: float = 300.0
    # Extra cost for each non-diagonal step. Stops DTW "chasing" small
    # vibrato/tuning wiggles with micro-warps that flatter a bad take.
    dtw_step_penalty: float = 50.0

    # --- Scoring ------------------------------------------------------------
    # A frame is off-pitch if |take - ref| (after key offset) exceeds this.
    pitch_threshold_cents: float = 40.0
    # Pitch is compared against the reference within +/- this many ms, so a
    # slide sung 30 ms late isn't ALSO counted as wrong pitch (timing is
    # scored separately). Measured on a real ornamented song: 78% -> 89%,
    # held-note errors 19% -> 13%. Too short to hide a held wrong note.
    # 0 = strict, exact-instant comparison.
    pitch_time_tolerance_ms: float = 30.0
    # A section is called early/late if its median drift exceeds this.
    timing_threshold_ms: float = 150.0
    # Timing is only measured at reference note starts: a voice entry, or a
    # pitch change larger than this (over 60 ms of median-smoothed pitch).
    # Inside a held note every DTW pairing costs the same, so "timing" there
    # is arbitrary. 80 c is above vibrato's ~30 c and below a semitone step.
    onset_change_cents: float = 80.0
    min_onset_gap_s: float = 0.15
    # A section must have at least this much of its singing off-pitch or
    # missed to be listed as a "worst" section.
    min_worst_badness: float = 0.10
    # Reference phrases are split at silences at least this long...
    phrase_min_gap_s: float = 0.25
    # Phrases longer than max_section_s (e.g. a legato line with no gaps, or
    # no gaps at all) are cut into equal parts of about fallback_window_s.
    # A 7 s "worst section" tells a singer nothing.
    fallback_window_s: float = 2.0
    max_section_s: float = 4.0
    # Sections with less reference singing than this aren't ranked.
    min_section_voiced_s: float = 0.5
    # Deviation within this of +/-1200 c, lasting under max_run, is treated
    # as a tracker octave error ("unsure"), not a singer error.
    octave_error_tolerance_cents: float = 150.0
    octave_error_max_run_s: float = 0.15
    n_worst_sections: int = 3

    @property
    def hop_s(self) -> float:
        return self.hop_length / self.sr

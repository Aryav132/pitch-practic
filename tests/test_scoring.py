import json

import numpy as np
import pytest

from pitch_practice.alignment import align
from pitch_practice.config import AnalysisConfig
from pitch_practice.scoring import (
    _sections, describe_key, score, summary_text, tolerant_deviation)
from tests.synth import MELODY, melody_track, shift

CFG = AnalysisConfig()
REF = melody_track(MELODY)


def run(take_notes, cfg=CFG, ref=REF):
    take = melody_track(take_notes)
    return score(ref, take, align(ref, take, cfg), cfg)


def note_start(k, notes=MELODY):
    return sum(d for _, d in notes[:k])


# ---------------------------------------------------------------- pitch

def test_perfect_take():
    r = run(MELODY)
    assert r.accuracy_pct == 100
    assert r.worst == []
    assert r.mean_abs_drift_ms < 5


def test_fifty_cents_flat_absolute_is_all_wrong():
    r = run(shift(MELODY, -50), AnalysisConfig(key_mode="absolute"))
    assert r.accuracy_pct == pytest.approx(0, abs=1)
    assert r.mean_dev_cents == pytest.approx(-50, abs=1)


def test_constant_flatness_survives_snapped_mode():
    r = run(shift(MELODY, -45))           # snapped mode (default)
    assert r.applied_offset_cents == 0
    assert r.accuracy_pct == pytest.approx(0, abs=1)
    # No key change was removed, so the flatness shows in accuracy, not as a
    # "residual after the key change".
    assert r.mean_dev_cents == pytest.approx(-45, abs=1)


def test_residual_reported_only_after_a_real_key_change():
    r = run(shift(MELODY, 1200 - 30))
    assert "30 cents flat overall" in r.key_description


def test_free_mode_hides_constant_flatness():
    # Documented behaviour, not a bug: "free" forgives any constant offset.
    r = run(shift(MELODY, -50), AnalysisConfig(key_mode="free"))
    assert r.raw_offset_cents == pytest.approx(-50, abs=1)
    assert r.accuracy_pct == 100


def test_octave_up_is_forgiven_and_reported():
    r = run([(None, 1.0)] + shift(MELODY, +1200))
    assert r.accuracy_pct >= 95
    assert r.key_description.startswith("You sang about 1 octave above")


def test_octave_up_in_absolute_mode_counts():
    r = run(shift(MELODY, +1200), AnalysisConfig(key_mode="absolute"))
    assert r.accuracy_pct < 5
    # Long octave runs are real errors, not tracker blips; only a few frames
    # at note changes can briefly look like short octave runs.
    assert r.unsure_s < 0.1


def test_held_wrong_note_is_the_worst_section_with_correct_time():
    # 0.5 s of note 7 (in phrase 2) sung 300 c sharp.
    wrong = list(MELODY)
    wrong[7] = (MELODY[7][0] + 300, 0.5)
    wrong.insert(8, (MELODY[7][0], 0.2))
    r = run(wrong)
    w = r.worst[0]
    assert w.index == 1                               # phrase 2
    t_wrong = note_start(7)
    bad_t = r.frames.times_s[r.frames.off_pitch]
    assert bad_t.min() == pytest.approx(t_wrong, abs=0.1)
    assert bad_t.max() == pytest.approx(t_wrong + 0.5, abs=0.1)
    # Not diluted by the good notes. (~294, not 300: with the +/-30 ms slack
    # the first 3 frames of the wrong note may match the previous note.)
    assert w.off_dev_cents == pytest.approx(300, abs=10)
    assert w.off_direction == "sharp"
    assert "cents (sharp) when off" in summary_text(r)


def test_short_octave_blip_is_unsure_not_wrong():
    blip = list(MELODY)
    blip[3] = (MELODY[3][0], 0.45)
    blip.insert(4, (MELODY[3][0] + 1200, 0.1))   # 100 ms tracker octave jump
    blip.insert(5, (MELODY[3][0], 0.45))
    r = run(blip)
    assert r.accuracy_pct == 100
    assert r.unsure_s == pytest.approx(0.1, abs=0.03)


def test_slack_forgives_a_glide_sung_30ms_late():
    ref = np.linspace(6000, 6600, 60)               # 600 c slide over 0.6 s
    take = np.concatenate([np.full(3, 6000.0), ref[:-3]])   # same slide, 30 ms late
    strict = tolerant_deviation(take, ref, 0)
    slack = tolerant_deviation(take, ref, 3)
    assert np.max(np.abs(strict)) == pytest.approx(30, abs=1)   # 3 frames x 10 c
    assert np.max(np.abs(slack)) < 1


def test_slack_keeps_sign_and_handles_unvoiced_neighbours():
    ref = np.array([np.nan, 6000, 6000, 6000, np.nan])
    take = np.array([np.nan, 5900, 5900, 5900, np.nan])
    assert tolerant_deviation(take, ref, 3)[1:4].tolist() == [-100, -100, -100]


def test_slack_does_not_hide_a_held_wrong_note():
    # 0.5 s held 100 c flat must still be flagged with the default +/-30 ms.
    wrong = list(MELODY)
    wrong[7] = (MELODY[7][0] - 100, 0.5)
    wrong.insert(8, (MELODY[7][0], 0.2))
    r = run(wrong)
    held = (r.frames.times_s > note_start(7) + 0.05) & (r.frames.times_s < note_start(7) + 0.45)
    assert r.frames.off_pitch[held].all()
    assert r.worst[0].off_direction == "flat"


# ---------------------------------------------------------------- not sung / extra

def test_extra_sound_after_the_end_is_not_a_pitch_error():
    # Your real recording: low creak/noise (~G2) after the last note.
    ref = melody_track(MELODY + [(None, 1.0)])
    r = run(MELODY + [(4300, 0.5), (None, 0.5)], ref=ref)
    assert r.accuracy_pct == 100
    assert r.extra_sound_s == pytest.approx(0.5, abs=0.05)


def test_skipped_phrase_is_reported_as_not_sung():
    skipped = [(None, d) if 10 <= k <= 12 else (c, d) for k, (c, d) in enumerate(MELODY)]
    r = run(skipped)
    assert r.worst[0].index == 2
    assert r.worst[0].missed_frac > 0.9
    assert r.accuracy_pct == 100      # what you did sing was in tune
    assert r.sung_pct < 85


# ---------------------------------------------------------------- timing

def test_whole_take_delayed_has_no_drift():
    r = run([(None, 0.2)] + MELODY)
    assert r.lag_s == pytest.approx(0.2, abs=0.011)
    assert r.mean_abs_drift_ms < 5
    assert r.timing_flags == []


def test_one_phrase_late_is_flagged():
    # Phrase 3 (notes 10-12) comes in 200 ms late, rest on time.
    late = MELODY[:9] + [(None, 0.3 + 0.2)] + MELODY[10:13] + [(None, 0.5 - 0.2)] + MELODY[14:]
    r = run(late)
    assert [s.index for s in r.timing_flags] == [2]
    assert r.timing_flags[0].drift_ms == pytest.approx(200, abs=20)


def test_reference_silence_has_no_timing():
    # take3 case: reference has 1.2 s of trailing silence, take stops early.
    ref = melody_track(MELODY + [(None, 1.2)])
    r = run(MELODY + [(None, 0.3)], ref=ref)
    assert np.nanmax(np.abs(r.frames.drift_ms)) < 20


# take1 case: reference's last 3 notes squeezed (50-80 c steps instead of
# whole tones) while the take sings a clean scale.
CLEAN = [(4800 + 200 * k, 1.0) for k in range(8)]
SQUEEZED = CLEAN[:5] + [(5650, 1.0), (5700, 1.0), (5780, 1.0)]


def _squeezed(band_s):
    """Report plus how far the DTW path slid (ms), measured on the path itself."""
    ref = melody_track([(None, 0.3)] + SQUEEZED + [(None, 0.3)])
    take = melody_track([(None, 0.3)] + CLEAN + [(None, 0.3)])
    cfg = AnalysisConfig(dtw_band_s=band_s)
    a = align(ref, take, cfg)
    slide_ms = np.max(np.abs(a.path_take - a.path_ref - a.lag.lag_frames)) * cfg.hop_s * 1000
    return score(ref, take, a, cfg), slide_ms


def test_out_of_tune_reference_is_capped_flagged_and_still_a_pitch_error():
    # Limitation, documented: DTW matches by pitch, so it still slides toward
    # same-pitch notes. The band caps the slide, the edge warning fires, and
    # the notes are still reported off-pitch.
    r, slide_ms = _squeezed(0.75)
    assert slide_ms <= 760
    assert r.band_edge_frames > 0
    assert "timing differed more than the aligner allows" in summary_text(r)
    assert r.frames.off_pitch[r.frames.times_s > 5.3].mean() > 0.8


def test_wide_band_lets_the_slide_double():
    # The old 1.5 s band: same input slides twice as far.
    assert _squeezed(1.5)[1] > 1000


def test_timing_is_only_measured_at_note_starts():
    r = run([(None, 0.2)] + MELODY)
    timed = np.flatnonzero(np.isfinite(r.frames.drift_ms))
    n_notes = sum(1 for c, _ in MELODY if c is not None)
    assert len(timed) == n_notes          # one timing value per sung note


# ---------------------------------------------------------------- sections / output

def test_mixed_misses_are_not_averaged_away():
    # Same phrase: one note 60 c sharp, another 60 c flat.
    mixed = list(MELODY)
    mixed[5] = (MELODY[5][0] + 60, MELODY[5][1])
    mixed[7] = (MELODY[7][0] - 60, MELODY[7][1])
    w = run(mixed).worst[0]
    assert w.off_dev_cents == pytest.approx(60, abs=1)
    assert w.off_direction == "both ways"


def test_phrases_split_at_rests():
    assert len(_sections(REF.voiced, CFG)) == 4


def test_long_phrase_is_cut_into_about_2s_parts():
    legato = melody_track([(6000 + 100 * (k % 5), 0.5) for k in range(14)])  # 7 s, no gaps
    secs = _sections(legato.voiced, CFG)
    assert [round((e - s) * CFG.hop_s, 2) for s, e in secs] == [1.75, 1.75, 1.75, 1.75]


def test_noise_blip_does_not_stop_long_phrase_splitting():
    # Your real reference: a legato scale plus a short noise blip after it.
    secs = _sections(melody_track([(6000, 7.0), (None, 0.8), (4300, 0.1)]).voiced, CFG)
    assert len(secs) == 5       # 4 parts of the scale + the blip


def test_take_ending_early_is_not_a_band_edge():
    # Reference continues 2 s after the take stops: coverage, not timing.
    # (A different pitch from the last note: a same-pitch note would let DTW
    # legitimately slide back to it - the pitch-matching limitation again.)
    ref = melody_track(MELODY + [(7000, 2.0)])
    r = run(MELODY + [(None, 0.2)], ref=ref)
    assert r.band_edge_frames == 0


@pytest.mark.parametrize("raw,applied,mode,expected", [
    (1180, 1200, "snapped", "1 octave above"),
    (-710, -700, "snapped", "7 semitones below"),
    (1900, 1900, "free", "1 octave and 7 semitones above"),
    (12, 0, "snapped", "Same key"),
])
def test_describe_key(raw, applied, mode, expected):
    assert expected in describe_key(raw, applied, mode)


def test_report_is_json_serialisable_and_summary_renders():
    r = run([(None, 1.0)] + shift(MELODY, 1200))
    json.dumps(r.to_dict(include_frames=True))
    text = summary_text(r)
    assert "Pitch accuracy" in text and "octave above" in text

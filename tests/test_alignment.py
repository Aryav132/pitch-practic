import librosa
import numpy as np
import pytest

from pitch_practice.alignment import (
    AlignmentError, align, applied_offset, dtw_banded, estimate_lag, _local_cost)
from pitch_practice.config import AnalysisConfig
from pitch_practice.pitch import PyinTracker
from tests.synth import MELODY, SR, melody_audio, melody_track, shift

CFG = AnalysisConfig()
REF = melody_track(MELODY)


def lag_s(est):
    return est.lag_frames * CFG.hop_s


# ---------------------------------------------------------------- lag search

def test_take_starts_7s_before_the_song():
    take = melody_track([(None, 7.0)] + MELODY)
    assert lag_s(estimate_lag(REF, take, CFG)) == pytest.approx(7.0, abs=0.011)


def test_take_starts_after_the_song_started():
    # Pressed record 2 s late: the take is missing the reference's first 2 s.
    take = melody_track(MELODY)
    take = take.__class__(take.times[200:], take.f0_hz[200:], take.voiced_prob[200:])
    assert lag_s(estimate_lag(REF, take, CFG)) == pytest.approx(-2.0, abs=0.011)


def test_lag_search_covers_the_whole_take():
    # 90 s take with the song placed at the very end.
    take = melody_track([(None, 80.0)] + MELODY)
    assert lag_s(estimate_lag(REF, take, CFG)) == pytest.approx(80.0, abs=0.011)


def test_200ms_delay():
    take = melody_track([(None, 0.2)] + MELODY)
    est = estimate_lag(REF, take, CFG)
    assert lag_s(est) == pytest.approx(0.2, abs=0.011)
    assert est.offset_cents == pytest.approx(0.0, abs=1)


def test_octave_up_offset_is_1200():
    take = melody_track([(None, 1.0)] + shift(MELODY, +1200))
    est = estimate_lag(REF, take, CFG)
    assert lag_s(est) == pytest.approx(1.0, abs=0.011)
    assert est.offset_cents == pytest.approx(1200, abs=1)
    assert not est.ambiguous


def test_repeated_phrase_is_flagged_ambiguous():
    phrase = MELODY[:5]
    ref = melody_track(phrase * 4)
    take = melody_track([(None, 1.0)] + phrase * 6)
    assert estimate_lag(ref, take, CFG).ambiguous


def test_periodic_song_prefers_full_coverage():
    # 60 s reference that repeats every 10 s; 90 s take, song starts at 20 s,
    # sung a fifth up. Lag 0 also matches perfectly on the 2/3 that overlap;
    # the full-coverage lag (20 s) must win. Also the realistic-size case.
    ref = melody_track(MELODY * 6)
    take = melody_track([(None, 20.0)] + shift(MELODY * 6, 700) + [(None, 10.0)])
    est = estimate_lag(ref, take, CFG)
    assert lag_s(est) == pytest.approx(20.0, abs=0.011)
    assert est.offset_cents == pytest.approx(700, abs=1)


def test_no_overlap_raises():
    with pytest.raises(AlignmentError):
        estimate_lag(REF, melody_track([(None, 10.0)]), CFG)


# ---------------------------------------------------------------- key modes

@pytest.mark.parametrize("raw,mode,expected", [
    (1200, "snapped", 1200), (-40, "snapped", 0), (-60, "snapped", -100),
    (-40, "free", -40), (1200, "absolute", 0),
])
def test_applied_offset(raw, mode, expected):
    assert applied_offset(raw, mode) == expected


def test_snapped_keeps_constant_flatness_free_hides_it():
    take = melody_track(shift(MELODY, -40))
    snapped = align(REF, take, CFG)
    free = align(REF, take, AnalysisConfig(key_mode="free"))
    assert snapped.raw_offset_cents == pytest.approx(-40, abs=1)
    assert snapped.applied_offset_cents == 0     # -40 c stays in the comparison
    assert free.applied_offset_cents == pytest.approx(-40, abs=1)  # ...or is hidden


# ---------------------------------------------------------------- DTW

def test_identical_inputs_give_diagonal_path():
    a = align(REF, REF, CFG)
    assert np.array_equal(a.path_ref, a.path_take)
    assert not a.hit_band_edge


def test_matches_librosa_dtw_oracle():
    # With a band wider than the inputs, our DP must equal librosa's subsequence
    # DTW using the same local costs and the same step penalties.
    rng = np.random.default_rng(1)
    ref = rng.choice([np.nan, 6000, 6200, 6400], size=40)
    take = rng.choice([np.nan, 6000, 6200, 6400], size=55)
    C = np.array([[_local_cost(r, t, CFG.dtw_cost_cap_cents, CFG.dtw_voicing_penalty)
                   for t in take] for r in ref])
    p = CFG.dtw_step_penalty
    D = librosa.sequence.dtw(C=C, subseq=True, backtrack=False,
                             step_sizes_sigma=np.array([[1, 1], [1, 0], [0, 1]]),
                             weights_add=np.array([0, p, p]), weights_mul=np.ones(3))
    *_, cost = dtw_banded(ref, take, lag=0, band=100, cfg=CFG)
    assert cost == pytest.approx(D[-1].min())


def test_recovers_a_phrase_sung_20_percent_slower():
    # Phrase 2 (notes 5-8) stretched by 20%; everything else on time.
    slow = [(c, d * 1.2) if 5 <= k <= 8 else (c, d) for k, (c, d) in enumerate(MELODY)]
    take = melody_track(slow)
    a = align(REF, take, CFG)
    # Where reference note 9 starts, the take should be 20% of phrase 2 later.
    ref_start = int(round(sum(d for _, d in MELODY[:9]) / CFG.hop_s))
    take_start = int(round(sum(d for _, d in slow[:9]) / CFG.hop_s))
    matched = a.path_take[a.path_ref == ref_start]
    assert np.min(np.abs(matched - take_start)) <= 3


def test_wrong_lag_hits_band_edge():
    # Force the DTW band to sit 3 s away from the truth.
    take = melody_track([(None, 3.0)] + MELODY)
    i, j, edge, _ = dtw_banded(REF.cents, take.cents, lag=0,
                               band=int(1.5 / CFG.hop_s), cfg=CFG)
    assert edge > 0


def test_dtw_does_not_hide_a_held_wrong_note():
    # 0.5 s of note 3 sung 300 c sharp. No nearby reference note has that
    # pitch, so the only way to "hide" it would be a long warp - the step
    # penalty must make that more expensive than admitting the error.
    wrong = list(MELODY)
    wrong[3] = (MELODY[3][0] + 300, 0.5)
    wrong.insert(4, (MELODY[3][0], 0.5))
    a = align(REF, melody_track(wrong), CFG)
    drift = a.path_take - a.path_ref - a.lag.lag_frames
    assert np.max(np.abs(drift)) <= 5


# ---------------------------------------------------------------- end to end

pyin = PyinTracker(CFG)


@pytest.mark.parametrize("delay_s,cents", [(0.2, 0), (3.0, 1200), (1.0, -50)])
def test_end_to_end_through_pyin(delay_s, cents):
    ref = pyin.track(melody_audio(MELODY), SR)
    take = pyin.track(melody_audio([(None, delay_s)] + shift(MELODY, cents)), SR)
    a = align(ref, take, AnalysisConfig(key_mode="free"))
    assert lag_s(a.lag) == pytest.approx(delay_s, abs=0.02)
    assert a.raw_offset_cents == pytest.approx(cents, abs=5)
    assert not a.hit_band_edge

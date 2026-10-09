import numpy as np
import pytest

from pitch_practice.config import AnalysisConfig
from pitch_practice.pitch import PyinTracker, drop_short_runs, hz_to_cents
from tests.synth import SR, cents_to_hz, noise, silence, tone

tracker = PyinTracker(AnalysisConfig())


def median_cents(y):
    tr = tracker.track(y, SR)
    return np.nanmedian(tr.cents), tr.voiced.mean()


def test_cents_scale():
    assert hz_to_cents(440.0) == pytest.approx(6900.0)
    assert hz_to_cents(880.0) == pytest.approx(8100.0)  # octave = +1200
    assert np.isnan(hz_to_cents(np.nan))


def test_sine_a4():
    c, voiced = median_cents(tone(440.0, 2.0))
    assert c == pytest.approx(6900, abs=5)
    assert voiced > 0.9


def test_fifty_cents_flat_is_resolved():
    c, _ = median_cents(tone(cents_to_hz(6850), 2.0))
    assert c == pytest.approx(6850, abs=5)


@pytest.mark.parametrize("name,cents", [("C2", 3600), ("A3", 5700), ("C6", 8400)])
def test_range_edges_harmonic_tone(name, cents):
    c, voiced = median_cents(tone(cents_to_hz(cents), 2.0, harmonics=5))
    assert c == pytest.approx(cents, abs=10), name
    assert voiced > 0.9, name


def test_silence_is_unvoiced():
    tr = tracker.track(silence(1.0), SR)
    assert not tr.voiced.any()


@pytest.mark.parametrize("seed", range(10))
def test_white_noise_is_unvoiced(seed):
    # Several seeds: with one seed this passed by luck before the
    # voiced_prob gate existed (other seeds gave up to 26% voiced).
    tr = tracker.track(noise(2.0, amp=0.3, seed=seed), SR)
    assert tr.voiced.mean() < 0.02


def test_note_boundaries_within_half_window():
    # A centred 64 ms frame "sees" a note up to 32 ms before it starts, so
    # boundary precision is ~half a window. The same bias hits reference and
    # take alike, so it cancels in relative timing.
    y = np.concatenate([silence(1.0), tone(440.0, 1.0, harmonics=5), silence(1.0)])
    tr = tracker.track(y, SR)
    t = tr.times[tr.voiced]
    assert t[0] == pytest.approx(1.0, abs=0.035)
    assert t[-1] == pytest.approx(2.0, abs=0.035)


def test_gap_between_phrases_is_unvoiced():
    y = np.concatenate([tone(440.0, 1.0, harmonics=5), silence(0.3), tone(494.0, 1.0, harmonics=5)])
    tr = tracker.track(y, SR)
    gap = (tr.times > 1.05) & (tr.times < 1.25)
    assert not tr.voiced[gap].any()


def test_energy_gate_removes_faint_bleed():
    # A loud note, then the same pitch 50 dB quieter: e.g. instrument residue
    # left in a separated vocal stem. pYIN alone would call it voiced.
    loud = tone(440.0, 1.0, harmonics=5, amp=0.3)
    faint = tone(330.0, 1.0, harmonics=5, amp=0.3 * 10 ** (-50 / 20))
    tr = tracker.track(np.concatenate([loud, faint]), SR)
    assert tr.voiced[tr.times < 0.95].mean() > 0.9
    assert not tr.voiced[tr.times > 1.1].any()


def testdrop_short_runs():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
    out = drop_short_runs(m, min_len=3)
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]


def test_rejects_wrong_sample_rate():
    with pytest.raises(ValueError):
        tracker.track(tone(440.0, 1.0, sr=16000), 16000)

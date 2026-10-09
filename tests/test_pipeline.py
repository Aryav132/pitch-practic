import numpy as np
import pytest
import soundfile as sf

from pitch_practice.config import AnalysisConfig
from pitch_practice.pipeline import InputError, analyze_files
from pitch_practice.plotting import make_figure
from tests.synth import MELODY, SR, melody_audio, shift, silence


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    # Real audio through every stage: reference has 5 s of intro before the
    # section we analyse; the take starts recording 2 s early, is sung an
    # octave down, and holds note 7 300 c sharp for 0.5 s.
    d = tmp_path_factory.mktemp("audio")
    sf.write(d / "ref.wav", np.concatenate([melody_audio(MELODY[::-1]), silence(0.0),
                                           melody_audio(MELODY)]), SR)
    wrong = list(MELODY)
    wrong[7] = (MELODY[7][0] + 300, 0.5)
    wrong.insert(8, (MELODY[7][0], 0.2))
    sf.write(d / "take.wav", melody_audio([(None, 2.0)] + shift(wrong, -1200)), SR)
    return analyze_files(d / "ref.wav", d / "take.wav", ref_start_s=10.0)


def test_end_to_end_numbers(report):
    assert report.lag_s == pytest.approx(2.0, abs=0.03)
    assert report.raw_offset_cents == pytest.approx(-1200, abs=10)
    assert "1 octave below" in report.key_description
    assert report.worst[0].index == 1
    assert report.worst[0].off_dev_cents == pytest.approx(300, abs=15)
    assert 85 <= report.accuracy_pct < 100


def test_worst_section_is_reported_in_song_time(report):
    # Song time = window start (10 s) + time inside the window.
    off_t = report.ref_start_s + report.frames.times_s[report.frames.off_pitch]
    t_wrong = 10.0 + sum(d for _, d in MELODY[:7])
    assert np.median(off_t) == pytest.approx(t_wrong + 0.25, abs=0.1)


def test_figure_shows_every_off_pitch_frame(report):
    fig = make_figure(report)
    off_trace = next(t for t in fig.data if t.name.startswith("Off-pitch"))
    assert len(off_trace.x) == report.frames.off_pitch.sum()
    assert sum(1 for a in fig.layout.annotations if a.text.startswith("worst #")) == len(report.worst)


def test_too_long_take_is_rejected(tmp_path):
    sf.write(tmp_path / "ref.wav", melody_audio(MELODY), SR)
    sf.write(tmp_path / "take.wav", np.zeros(int(SR * 2.0), dtype=np.float32), SR)
    with pytest.raises(InputError, match="longer than"):
        analyze_files(tmp_path / "ref.wav", tmp_path / "take.wav",
                      cfg=AnalysisConfig(max_take_s=1.0))


def test_silent_take_is_rejected(tmp_path):
    sf.write(tmp_path / "ref.wav", melody_audio(MELODY), SR)
    sf.write(tmp_path / "take.wav", silence(5.0), SR)
    with pytest.raises(InputError, match="no singing found in your take"):
        analyze_files(tmp_path / "ref.wav", tmp_path / "take.wav")

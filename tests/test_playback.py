import numpy as np
import pytest

from pitch_practice.playback import clip, hear_the_difference
from tests.synth import SR, tone


def peak_hz(y):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(spec)]


def test_clip_is_the_requested_window():
    y = np.arange(SR * 3, dtype=np.float32)
    c = clip(y, SR, 1.0, 2.0)
    assert len(c) == SR and c[SR // 2] == pytest.approx(1.5 * SR)


def test_difference_is_singer_you_singer_at_equal_loudness():
    singer, you = tone(440.0, 1.0, amp=0.5), tone(415.3, 1.0, amp=0.05)   # you: quiet, lower
    out = hear_the_difference(singer, you, SR)
    assert len(out) == pytest.approx(SR * 4, abs=2)      # 1 + 0.5 + 1 + 0.5 + 1 s
    first, middle = out[:SR], out[int(1.5 * SR):int(2.5 * SR)]
    assert peak_hz(first) == pytest.approx(440.0, abs=2)
    assert peak_hz(middle) == pytest.approx(415.3, abs=2)
    assert np.max(np.abs(first)) == pytest.approx(np.max(np.abs(middle)), rel=0.01)


def test_slow_version_is_longer_but_same_pitch():
    singer, you = tone(440.0, 1.0), tone(415.3, 1.0)
    slow = hear_the_difference(singer, you, SR, slow=True)
    assert len(slow) == pytest.approx(SR * 4 / 0.75, rel=0.02)
    assert peak_hz(slow[: int(SR / 0.75)]) == pytest.approx(440.0, abs=3)

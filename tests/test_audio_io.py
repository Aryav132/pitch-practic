import subprocess

import numpy as np
import pytest
import soundfile as sf

from pitch_practice.audio_io import AudioDecodeError, file_sha256, load_audio
from tests.synth import silence, tone


def peak_hz(y: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(spec)]


def test_resamples_and_preserves_frequency(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, tone(440.0, 2.0, sr=44100), 44100)
    y = load_audio(p, sr=16000)
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(len(y) - 32000) <= 2
    assert abs(peak_hz(y, 16000) - 440.0) < 1.0


def test_window_seeks_to_start(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.concatenate([silence(1.0, sr=16000), tone(440.0, 1.0, sr=16000)]), 16000)
    y = load_audio(p, sr=16000, start_s=1.1, duration_s=0.5)
    assert abs(len(y) - 8000) <= 2
    assert np.sqrt(np.mean(y**2)) > 0.1  # got the tone, not the leading silence


def test_stereo_shape(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, tone(440.0, 1.0, sr=44100), 44100)
    y = load_audio(p, sr=44100, channels=2)
    assert y.shape == (2, 44100)


def test_decodes_m4a_like_phone_voice_memo(tmp_path):
    wav, m4a = tmp_path / "a.wav", tmp_path / "a.m4a"
    sf.write(wav, tone(330.0, 3.0, sr=16000), 16000)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", "aac", str(m4a)], check=True)
    y = load_audio(m4a, sr=16000)
    assert abs(len(y) / 16000 - 3.0) < 0.05  # AAC encoder padding is trimmed
    assert abs(peak_hz(y, 16000) - 330.0) < 1.0


def test_bad_inputs_raise(tmp_path):
    with pytest.raises(AudioDecodeError):
        load_audio(tmp_path / "missing.mp3", sr=16000)
    junk = tmp_path / "junk.mp3"
    junk.write_text("not audio")
    with pytest.raises(AudioDecodeError):
        load_audio(junk, sr=16000)


def test_hash_depends_on_content(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, tone(440.0, 0.5), 24000)
    sf.write(b, tone(441.0, 0.5), 24000)
    assert file_sha256(a) == file_sha256(a) != file_sha256(b)

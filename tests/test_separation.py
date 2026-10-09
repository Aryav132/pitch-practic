import numpy as np
import pytest
import soundfile as sf

from pitch_practice.audio_io import load_audio
from pitch_practice.config import AnalysisConfig
from pitch_practice.pitch import PyinTracker
from pitch_practice.separation import DemucsSeparator, NoSeparator
from tests.synth import MELODY, backing_track, shift, voice_like

MODEL_SR = 44100


class FakeModel:
    """Stands in for Demucs: 'vocals' = left channel. Lets us test windowing,
    padding and caching in milliseconds, without torch."""
    samplerate, audio_channels = MODEL_SR, 2
    sources = ["drums", "bass", "other", "vocals"]


class FakeDemucs(DemucsSeparator):
    def __init__(self, cache_dir):
        super().__init__(cache_dir=cache_dir)
        self.calls = []

    def _load_model(self):
        return FakeModel()

    def _run_model(self, model, mix):
        self.calls.append(mix.shape[1] / MODEL_SR)
        return np.stack([mix[0], mix[0]])


@pytest.fixture
def song(tmp_path):
    # Left channel: a timestamp ramp (value = time in seconds / 100), so any
    # windowing error shows up as a wrong value. Right channel: silent.
    t = np.arange(20 * MODEL_SR) / MODEL_SR
    stereo = np.stack([t / 100, np.zeros_like(t)], 1)
    p = tmp_path / "song.wav"
    sf.write(p, stereo.astype(np.float32), MODEL_SR, subtype="FLOAT")
    return p


def test_window_is_exact_after_padding(song, tmp_path):
    sep = FakeDemucs(tmp_path / "cache")
    y = sep.load_vocals(song, start_s=5.0, duration_s=4.0, sr=MODEL_SR)
    assert sep.calls == [pytest.approx(8.0, abs=0.01)]      # 2 s pad each side
    assert len(y) == pytest.approx(4.0 * MODEL_SR, abs=2)
    assert y[0] * 100 == pytest.approx(5.0, abs=0.01)        # starts at 5 s, not 3 s
    assert y[-1] * 100 == pytest.approx(9.0, abs=0.01)


def test_window_at_song_start_pads_only_after(song, tmp_path):
    sep = FakeDemucs(tmp_path / "cache")
    y = sep.load_vocals(song, start_s=0.5, duration_s=3.0, sr=MODEL_SR)
    assert sep.calls == [pytest.approx(0.5 + 3.0 + 2.0, abs=0.01)]
    assert y[0] * 100 == pytest.approx(0.5, abs=0.01)


def test_second_call_hits_cache(song, tmp_path):
    sep = FakeDemucs(tmp_path / "cache")
    a = sep.load_vocals(song, 5.0, 4.0, MODEL_SR)
    assert sep.last_was_cached is False
    b = sep.load_vocals(song, 5.0, 4.0, MODEL_SR)
    assert sep.last_was_cached is True and len(sep.calls) == 1
    assert np.array_equal(a, b)


def test_cache_key_depends_on_content_and_window(song, tmp_path):
    sep = FakeDemucs(tmp_path / "cache")
    k = sep.cache_path(song, 5.0, 4.0)
    assert sep.cache_path(song, 6.0, 4.0) != k
    assert sep.cache_path(song, 5.0, 3.0) != k
    renamed = song.with_name("renamed.wav")
    renamed.write_bytes(song.read_bytes())
    assert sep.cache_path(renamed, 5.0, 4.0) == k            # same content, new name


def test_no_separator_is_a_plain_windowed_load(song):
    y = NoSeparator().load_vocals(song, 5.0, 4.0, MODEL_SR)
    assert y[0] * 100 == pytest.approx(5.0 / 2, abs=0.01)    # mono = mean of L and R (R = 0)


@pytest.mark.slow
def test_demucs_recovers_the_melody_from_a_mix(tmp_path):
    # Real model (downloads weights once). A synthetic voice over synthetic
    # drums, bass and chords. Caveat: Demucs was trained on real music; a
    # synthetic voice is an easier case than a real recording.
    cfg = AnalysisConfig()
    notes = MELODY + shift(MELODY, -500)
    vox = voice_like(notes, MODEL_SR)
    mix = 0.8 * vox + backing_track(len(vox) / MODEL_SR, MODEL_SR)
    sf.write(tmp_path / "mix.wav", np.stack([mix, mix], 1), MODEL_SR)
    sf.write(tmp_path / "vox.wav", vox, MODEL_SR)

    tr = PyinTracker(cfg)
    clean = tr.track(load_audio(tmp_path / "vox.wav", cfg.sr), cfg.sr)
    raw = tr.track(load_audio(tmp_path / "mix.wav", cfg.sr), cfg.sr)
    stem = tr.track(DemucsSeparator(cache_dir=tmp_path / "c").load_vocals(
        tmp_path / "mix.wav", 0.0, 20.0, cfg.sr), cfg.sr)

    def within_50c(t):
        both = clean.voiced & t.voiced
        return np.sum(np.abs(t.cents[both] - clean.cents[both]) < 50) / clean.voiced.sum()

    assert within_50c(raw) < 0.5        # the mix alone is not usable...
    assert within_50c(stem) > 0.95      # ...the separated vocals are
    assert np.mean(stem.voiced & ~clean.voiced) < 0.02

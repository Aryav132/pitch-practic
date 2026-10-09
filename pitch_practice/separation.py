"""Get the reference VOCALS for a time window, from either a clean recording
(NoSeparator) or a full song mix (DemucsSeparator).

Demucs (HTDemucs, Meta) is a pretrained deep neural network - the one
machine-learning component in the pipeline. Everything after it (pitch
tracking, alignment, scoring) is classical signal processing.

Sample rates: Demucs runs at its native rate (model.samplerate, 44.1 kHz
stereo). Only the separated vocals are resampled, to the analysis rate,
when they are loaded back through audio_io.load_audio (ffmpeg).
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from .audio_io import file_sha256, load_audio

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "stems"


class Separator(Protocol):
    def load_vocals(self, path: str | Path, start_s: float, duration_s: float, sr: int) -> np.ndarray:
        """Mono vocals for [start_s, start_s + duration_s) at rate sr."""
        ...


class NoSeparator:
    """The reference is already a clean vocal (e.g. you recorded it)."""

    def load_vocals(self, path, start_s, duration_s, sr):
        return load_audio(path, sr, start_s=start_s, duration_s=duration_s)


class DemucsSeparator:
    # Demucs quality drops near the edges of the audio it's given, so we
    # separate a little extra on each side and trim it off afterwards.
    PAD_S = 2.0

    def __init__(self, model_name: str = "htdemucs", device: str = "auto",
                 cache_dir: str | Path = DEFAULT_CACHE):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self._model = None
        self.last_was_cached: bool | None = None  # for the UI and tests

    # -- cache ---------------------------------------------------------------

    def cache_path(self, path, start_s: float, duration_s: float) -> Path:
        # Content hash, not file name: re-uploading the same song under a new
        # name still hits the cache; a different file with the same name doesn't.
        key = f"{file_sha256(path)}|{self.model_name}|{start_s:.3f}|{duration_s:.3f}|{self.PAD_S}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.wav"

    def load_vocals(self, path, start_s, duration_s, sr):
        stem = self.cache_path(path, start_s, duration_s)
        self.last_was_cached = stem.exists()
        if not self.last_was_cached:
            vocals, model_sr = self._separate(path, start_s, duration_s)
            self._write_atomic(stem, vocals, model_sr)
        # Resample the stem to the analysis rate with the same ffmpeg path as
        # every other input.
        return load_audio(stem, sr)

    def _write_atomic(self, stem: Path, vocals: np.ndarray, sr: int) -> None:
        # Write to a temp file and rename, so an interrupted run never leaves
        # a half-written stem that later counts as a cache hit.
        stem.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=stem.parent)
        os.close(fd)
        sf.write(tmp, vocals.T, sr, subtype="FLOAT")
        os.replace(tmp, stem)

    # -- model ---------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from demucs.pretrained import get_model  # torch only when needed
            from huggingface_hub.utils import logging as hf_logging

            # Weights come from the Hugging Face Hub (cached after the first
            # download); its "unauthenticated requests" notice is just noise.
            hf_logging.set_verbosity_error()

            self._model = get_model(self.model_name)
            self._model.eval()
        return self._model

    def _separate(self, path, start_s: float, duration_s: float) -> tuple[np.ndarray, int]:
        """Return (vocals of shape (channels, n), model sample rate)."""
        model = self._load_model()
        sr, ch = model.samplerate, model.audio_channels
        s0 = max(0.0, start_s - self.PAD_S)
        lead = start_s - s0
        mix = load_audio(path, sr, channels=ch, start_s=s0,
                         duration_s=lead + duration_s + self.PAD_S)
        vocals = self._run_model(model, mix)
        i0 = int(round(lead * sr))
        return vocals[:, i0:i0 + int(round(duration_s * sr))], sr

    def resolved_device(self) -> str:
        """'auto' = Apple GPU (MPS) when available, else CPU. Measured on a
        60 s clip: CPU 119 s, MPS 32 s, identical pitch results downstream
        (stems differ by ~32 dB SNR - floating-point, not quality)."""
        if self.device != "auto":
            return self.device
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _run_model(self, model, mix: np.ndarray) -> np.ndarray:
        import torch
        from demucs.apply import apply_model

        wav = torch.from_numpy(mix)
        # Same normalisation as Demucs' own CLI: zero-mean, unit-std on the
        # mono mix, undone on the output.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(model, ((wav - mean) / std)[None], device=self.resolved_device(),
                              split=True, overlap=0.25, progress=False)[0]
        vocals = out[model.sources.index("vocals")] * std + mean
        return vocals.cpu().numpy().astype(np.float32)

"""Decode any audio file to a float32 numpy array via ffmpeg.

Why ffmpeg for everything: phone takes are usually .m4a (AAC), which
soundfile/libsndfile cannot read, and librosa's fallback decoders differ
across machines. One subprocess call gives one code path for MP3/M4A/WAV,
with resampling and downmixing done by the same tool.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np


class AudioDecodeError(RuntimeError):
    pass


def load_audio(
    path: str | Path,
    sr: int,
    channels: int = 1,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> np.ndarray:
    """Return audio resampled to `sr`.

    Shape is (n_samples,) for mono, (channels, n_samples) otherwise.
    Separation (step 4) calls this with the model's rate and channels=2;
    pitch tracking calls it with 16 kHz mono.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("ffmpeg not found on PATH (brew install ffmpeg)")
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"no such file: {path}")

    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    # -ss/-t before -i seek in the input, so we never decode the whole song
    # just to keep 60 seconds of it.
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-i", str(path), "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(channels), "-ar", str(sr), "-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg failed on {path.name}: {msg}")

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise AudioDecodeError(f"{path.name}: no audio decoded (start past end of file?)")
    if channels > 1:
        # ffmpeg emits interleaved samples: L R L R ...
        y = y.reshape(-1, channels).T
    return np.ascontiguousarray(y)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash used as the separation cache key (step 4)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

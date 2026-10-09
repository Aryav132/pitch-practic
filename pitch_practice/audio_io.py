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
    # For mono we decode at the file's own channel count and average
    # ourselves. ffmpeg's -ac remixing uses a -3 dB pan law both ways:
    # stereo->mono gives (L + R) * 0.707 and mono->stereo gives 0.707 * x,
    # so levels would depend on how the file happened to be stored.
    # (Surround files are first folded to stereo by ffmpeg.)
    out_ch = min(_probe_channels(path), 2) if channels == 1 else channels
    cmd += ["-i", str(path), "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(out_ch), "-ar", str(sr), "-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg failed on {path.name}: {msg}")

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise AudioDecodeError(f"{path.name}: no audio decoded (start past end of file?)")
    # ffmpeg emits interleaved samples: L R L R ...
    y = y.reshape(-1, out_ch).T
    if channels == 1:
        y = y.mean(axis=0)
    return np.ascontiguousarray(y)


def _probe_channels(path: Path) -> int:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return max(1, int(proc.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        raise AudioDecodeError(f"{path.name}: no audio stream found") from None


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash used as the separation cache key (step 4)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

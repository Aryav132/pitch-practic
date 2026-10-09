"""The one entry point the UI and CLI call: files in, Report out.

The reference goes through a Separator (Demucs for a full song, nothing for
a clean vocal); the take is always your own voice and is used as is.
"""

from pathlib import Path

from .alignment import align
from .audio_io import load_audio
from .config import AnalysisConfig
from .pitch import PitchTrack, PyinTracker, PitchTracker
from .scoring import Report, score
from .separation import NoSeparator, Separator


class InputError(ValueError):
    """Problem with the user's files, worded for the user."""


def analyze_tracks(ref: PitchTrack, take: PitchTrack, cfg: AnalysisConfig = AnalysisConfig(),
                   ref_start_s: float = 0.0) -> Report:
    return score(ref, take, align(ref, take, cfg), cfg, ref_start_s)


def analyze_files(ref_path: str | Path, take_path: str | Path, ref_start_s: float = 0.0,
                  cfg: AnalysisConfig = AnalysisConfig(),
                  tracker: PitchTracker | None = None,
                  separator: Separator | None = None) -> Report:
    tracker = tracker or PyinTracker(cfg)
    separator = separator or NoSeparator()
    ref_y = separator.load_vocals(ref_path, ref_start_s, cfg.max_ref_s, cfg.sr)
    # Read one extra second so "too long" can be detected rather than silently cut.
    take_y = load_audio(take_path, cfg.sr, duration_s=cfg.max_take_s + 1.0)
    if len(take_y) / cfg.sr > cfg.max_take_s:
        raise InputError(f"Your take is longer than {cfg.max_take_s:.0f} s. Record just the "
                         f"section you're practising, plus a few seconds either side.")
    ref = tracker.track(ref_y, cfg.sr)
    take = tracker.track(take_y, cfg.sr)
    if ref.voiced.mean() < 0.1:
        raise InputError("Almost no singing found in the reference at this start time "
                         "(an instrumental section?). Try a different start time.")
    if take.voiced.mean() < 0.1:
        raise InputError("Almost no singing found in your take. Is it too quiet or noisy?")
    return analyze_tracks(ref, take, cfg, ref_start_s)

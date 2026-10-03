"""Step-1 manual check: track pitch in one file and print what was found.

    python scripts/show_pitch.py my_take.m4a [--start 30] [--duration 20] [--step 0.1]
"""

import argparse
import time

import librosa
import numpy as np

from pitch_practice.audio_io import load_audio
from pitch_practice.config import AnalysisConfig
from pitch_practice.pitch import PyinTracker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--step", type=float, default=1.0, help="seconds per printed row")
    args = ap.parse_args()

    cfg = AnalysisConfig()
    y = load_audio(args.path, sr=cfg.sr, start_s=args.start, duration_s=args.duration)
    t0 = time.time()
    track = PyinTracker(cfg).track(y, cfg.sr)
    elapsed = time.time() - t0

    cents = track.cents[track.voiced]
    print(f"{len(y) / cfg.sr:.1f} s of audio, tracked in {elapsed:.1f} s")
    print(f"voiced: {track.voiced.mean():.0%} of {len(track)} frames")
    if cents.size == 0:
        print("no pitched sound found")
        return
    lo, med, hi = np.percentile(cents, [5, 50, 95]) / 100
    print(f"range (5th-95th pct): {librosa.midi_to_note(lo, cents=True)} .. "
          f"{librosa.midi_to_note(hi, cents=True)}, median {librosa.midi_to_note(med, cents=True)}")
    # Each row: median pitch of the voiced frames in [t, t + step).
    # Times are printed in the original file's timeline (offset by --start).
    print("\n    time    note")
    for t in np.arange(0.0, track.times[-1] + 1e-9, args.step):
        in_row = (track.times >= t) & (track.times < t + args.step) & track.voiced
        note = librosa.midi_to_note(np.median(track.cents[in_row]) / 100, cents=True) if in_row.any() else "-"
        print(f"  {args.start + t:6.2f}s  {note}")


if __name__ == "__main__":
    main()

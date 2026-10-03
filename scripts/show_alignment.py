"""Step-2 manual check: align a take to a reference and print what was found.

    python scripts/show_alignment.py reference.m4a take.m4a [--mode snapped|free|absolute]
"""

import argparse
from dataclasses import replace

import numpy as np

from pitch_practice.alignment import align
from pitch_practice.audio_io import load_audio
from pitch_practice.config import AnalysisConfig
from pitch_practice.pitch import PyinTracker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("take")
    ap.add_argument("--mode", default="snapped", choices=["snapped", "free", "absolute"])
    args = ap.parse_args()

    cfg = replace(AnalysisConfig(), key_mode=args.mode)
    tracker = PyinTracker(cfg)
    ref = tracker.track(load_audio(args.reference, cfg.sr, duration_s=cfg.max_ref_s), cfg.sr)
    take = tracker.track(load_audio(args.take, cfg.sr, duration_s=cfg.max_take_s), cfg.sr)
    a = align(ref, take, cfg)

    lag = a.lag
    print(f"reference starts {lag.lag_frames * cfg.hop_s:+.2f} s into the take")
    if lag.ambiguous:
        alt = lag.runner_up_lag_frames * cfg.hop_s
        print(f"  WARNING: ambiguous - a lag of {alt:+.2f} s fits almost as well "
              f"(cost {lag.runner_up_cost_cents:.0f} vs {lag.cost_cents:.0f})")
    print(f"key offset: {a.raw_offset_cents:+.0f} cents "
          f"({a.raw_offset_cents / 100:+.1f} semitones); mode={a.key_mode}, "
          f"subtracted {a.applied_offset_cents:+.0f}")
    print(f"band +/-{a.band_frames * cfg.hop_s:.1f} s, path touched the edge on "
          f"{a.band_edge_frames} frames" + ("  <- timing differs more than the band allows"
                                            if a.hit_band_edge else ""))

    # Drift: how far the take is from where the lag alone predicts.
    # Positive = singer later than their own average placement.
    drift_ms = (a.path_take - a.path_ref - lag.lag_frames) * cfg.hop_s * 1000
    t_ref = a.path_ref * cfg.hop_s
    print("\n  ref time   drift (ms, + = late)")
    for sec in range(int(t_ref[-1]) + 1):
        in_sec = (t_ref >= sec) & (t_ref < sec + 1)
        print(f"    {sec:4d}s   {np.median(drift_ms[in_sec]):+5.0f}")


if __name__ == "__main__":
    main()

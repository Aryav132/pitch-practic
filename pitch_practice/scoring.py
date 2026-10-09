"""Turn an alignment into feedback: per-frame deviations, timing drift,
phrase sections, the worst sections and a summary.

Everything is indexed on the REFERENCE timeline: frame i of the reference
gets one matched take pitch and one matched take time. Every number the
user (or a Phase 3 coach) sees is computed here, not invented downstream.
"""

from dataclasses import asdict, dataclass

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .alignment import Alignment
from .config import AnalysisConfig
from .pitch import PitchTrack, drop_short_runs

# Per-frame status codes
SILENT, BOTH, MISSED, EXTRA = 0, 1, 2, 3  # ref/take voiced: no/no, y/y, y/n, n/y


@dataclass(frozen=True)
class FrameScores:
    """Arrays of length n_ref_frames (reference timeline)."""
    times_s: np.ndarray      # reference frame times (within the window)
    ref_cents: np.ndarray
    take_cents: np.ndarray   # matched take pitch, key offset already removed
    deviation: np.ndarray    # take - ref in cents; NaN unless scored
    status: np.ndarray       # SILENT / BOTH / MISSED / EXTRA
    unsure: np.ndarray       # bool: probable tracker octave error, not scored
    off_pitch: np.ndarray    # bool: |deviation| > threshold
    drift_ms: np.ndarray     # timing vs the singer's own median, at reference
                             # note starts only; NaN elsewhere
    take_times_s: np.ndarray  # matched time in the take (NaN outside path)


@dataclass(frozen=True)
class Section:
    index: int
    start_s: float           # reference-window time
    end_s: float
    take_start_s: float      # where this section was sung in the take
    ref_voiced_s: float
    scored_s: float          # both voiced and not "unsure"
    off_pitch_frac: float    # of scored frames
    missed_frac: float       # of reference-voiced frames
    mean_abs_dev_cents: float
    mean_dev_cents: float    # signed: < 0 flat, > 0 sharp
    off_dev_cents: float     # mean |deviation| of the off-pitch frames only:
                             # "when you missed, by how much" (not diluted by good notes)
    off_direction: str       # "sharp", "flat" or "both ways" (>= 70% one sign)
    drift_ms: float          # median note-start timing vs singer's own average
    badness: float           # (off-pitch + missed) / reference-voiced


@dataclass(frozen=True)
class Report:
    ref_start_s: float       # window start inside the reference file
    accuracy_pct: float      # in-tune scored frames / scored frames
    mean_abs_dev_cents: float
    mean_dev_cents: float
    sung_pct: float          # reference-voiced frames where the singer sang
    mean_abs_drift_ms: float
    raw_offset_cents: float
    applied_offset_cents: float
    key_mode: str
    key_description: str
    lag_s: float
    lag_ambiguous: bool
    band_edge_frames: int
    unsure_s: float
    extra_sound_s: float
    threshold_cents: float
    pitch_time_tolerance_ms: float
    timing_threshold_ms: float
    sections: list[Section]
    worst: list[Section]
    timing_flags: list[Section]  # sections with |drift| > timing threshold
    frames: FrameScores

    def to_dict(self, include_frames: bool = False) -> dict:
        """JSON-safe dict. This is the only input a Phase 3 coach gets."""
        d = {k: v for k, v in asdict(self).items() if k != "frames"}
        if include_frames:
            d["frames"] = {k: np.where(np.isnan(v), None, v).tolist()
                           if v.dtype.kind == "f" else v.tolist()
                           for k, v in asdict(self.frames).items()}
        return _json_safe(d)


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def describe_key(raw_offset_cents: float, applied_cents: float, key_mode: str) -> str:
    def interval(c):
        semis = int(round(abs(c) / 100))
        octaves, rest = divmod(semis, 12)
        parts = []
        if octaves:
            parts.append(f"{octaves} octave" + ("s" if octaves > 1 else ""))
        if rest:
            parts.append(f"{rest} semitone" + ("s" if rest > 1 else ""))
        return " and ".join(parts)

    if abs(raw_offset_cents) < 50:
        text = "Same key as the reference"
    else:
        where = "above" if raw_offset_cents > 0 else "below"
        text = f"You sang about {interval(raw_offset_cents)} {where} the reference"
    text += f" (measured {raw_offset_cents:+.0f} cents)."
    if key_mode == "absolute" and abs(raw_offset_cents) >= 50:
        text += " Absolute mode: this difference counts as error."
    elif key_mode == "snapped" and applied_cents != 0:
        residual = raw_offset_cents - applied_cents
        if abs(residual) >= 10:
            text += (f" After allowing for the key change you were {abs(residual):.0f} cents "
                     f"{'sharp' if residual > 0 else 'flat'} overall; that still counts.")
        if 40 <= abs(residual) <= 50:
            text += " (Close to half a semitone: the key estimate could be one semitone off.)"
    return text


def _per_ref_frame(alignment: Alignment, take_cents_adj: np.ndarray, n_ref: int):
    """Collapse the warp path to one take pitch and one take index per ref frame."""
    take_c = np.full(n_ref, np.nan)
    take_j = np.full(n_ref, np.nan)
    ref_idx, starts = np.unique(alignment.path_ref, return_index=True)
    for i, group in zip(ref_idx, np.split(alignment.path_take, starts[1:])):
        vals = take_cents_adj[group]
        if not np.isnan(vals).all():
            take_c[i] = np.nanmedian(vals)
        take_j[i] = group.mean()
    return take_c, take_j


def tolerant_deviation(take_c: np.ndarray, ref_c: np.ndarray, k: int) -> np.ndarray:
    """take - ref, using whichever voiced reference frame within +/-k frames
    is closest in pitch. Signed, so sharp/flat survives. k = 0 is exact."""
    n = len(ref_c)
    best = take_c - ref_c
    for shift in range(-k, k + 1):
        if shift == 0:
            continue
        ref_shifted = np.full(n, np.nan)          # ref_shifted[i] = ref_c[i + shift]
        if shift > 0:
            ref_shifted[:n - shift] = ref_c[shift:]
        else:
            ref_shifted[-shift:] = ref_c[:n + shift]
        d = take_c - ref_shifted
        closer = np.isfinite(d) & ~(np.abs(best) <= np.abs(d))   # NaN best counts as worse
        best = np.where(closer, d, best)
    return best


def _note_starts(ref_cents: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """Reference frames where a note starts: voice entries and pitch changes."""
    n = len(ref_cents)
    voiced = ~np.isnan(ref_cents)
    entry = voiced & ~np.concatenate(([False], voiced[:-1]))
    # Median-smoothed pitch (5 frames) so single-frame tracker wobble and
    # vibrato don't count; then the change across +/-3 frames (60 ms).
    pad = np.pad(ref_cents, 2, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows
        smooth = np.nanmedian(sliding_window_view(pad, 5), axis=1)
    change = np.zeros(n, bool)
    change[3:-3] = np.abs(smooth[6:] - smooth[:-6]) > cfg.onset_change_cents
    candidates = np.flatnonzero(entry | (change & voiced))
    gap = round(cfg.min_onset_gap_s / cfg.hop_s)
    starts, last = [], -gap
    for i in candidates:
        if i - last >= gap:
            starts.append(i)
        last = i                 # a run of candidates is one note change
    return np.array(starts, dtype=int)


def _direction(devs: np.ndarray) -> str:
    """Averaging +45 and -60 gives -7, which says nothing; report the mix."""
    if devs.size == 0:
        return ""
    sharp = np.mean(devs > 0)
    return "sharp" if sharp >= 0.7 else "flat" if sharp <= 0.3 else "both ways"


def _sections(ref_voiced: np.ndarray, cfg: AnalysisConfig) -> list[tuple[int, int]]:
    """[start, end) reference-frame ranges: phrases split at rests, with long
    phrases cut into roughly fallback_window_s parts."""
    edges = np.diff(np.concatenate(([0], ref_voiced.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return []
    gap = round(cfg.phrase_min_gap_s / cfg.hop_s)
    phrases = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - phrases[-1][1] < gap:
            phrases[-1][1] = e      # short gap (a breath, a consonant): same phrase
        else:
            phrases.append([s, e])

    out = []
    max_len = round(cfg.max_section_s / cfg.hop_s)
    for lo, hi in phrases:
        if hi - lo <= max_len:
            out.append((lo, hi))
            continue
        parts = int(round((hi - lo) * cfg.hop_s / cfg.fallback_window_s))
        cuts = np.linspace(lo, hi, parts + 1).round().astype(int)
        out.extend(zip(cuts[:-1], cuts[1:]))
    return [(int(a), int(b)) for a, b in out]


def score(ref: PitchTrack, take: PitchTrack, alignment: Alignment,
          cfg: AnalysisConfig = AnalysisConfig(), ref_start_s: float = 0.0) -> Report:
    hop, n = cfg.hop_s, len(ref)
    ref_c = ref.cents
    take_c, take_j = _per_ref_frame(alignment, take.cents - alignment.applied_offset_cents, n)

    ref_v, take_v = ~np.isnan(ref_c), ~np.isnan(take_c)
    status = np.select([ref_v & take_v, ref_v, take_v], [BOTH, MISSED, EXTRA], SILENT)
    both = status == BOTH

    k = round(cfg.pitch_time_tolerance_ms / 1000 / hop)
    dev = np.where(both, tolerant_deviation(take_c, ref_c, k), np.nan)
    near_octave = both & (np.abs(np.abs(dev) - 1200) < cfg.octave_error_tolerance_cents)
    max_run = round(cfg.octave_error_max_run_s / hop)
    # Short octave jumps = tracker error; long ones = really sung an octave off.
    unsure = near_octave & ~drop_short_runs(near_octave, max_run + 1)
    scored = both & ~unsure
    dev = np.where(scored, dev, np.nan)
    off = scored & (np.abs(dev) > cfg.pitch_threshold_cents)

    # Timing: only at reference note starts that you actually sang (checked
    # 50 ms in, once the note is established), relative to your own median.
    # A constant offset is the recording start, not an error.
    raw_drift = (take_j - np.arange(n) - alignment.lag.lag_frames) * hop * 1000
    starts = _note_starts(ref_c, cfg)
    settle = round(0.05 / hop)
    starts = starts[(starts + settle < n)]
    starts = starts[both[starts + settle] & np.isfinite(raw_drift[starts])]
    drift = np.full(n, np.nan)
    drift[starts] = raw_drift[starts]
    if starts.size:
        drift = drift - np.nanmedian(drift)
    timed = ~np.isnan(drift)

    take_times = take_j * hop
    sections = []
    for k, (s, e) in enumerate(_sections(ref_v, cfg)):
        sl = slice(s, e)
        n_ref_v, n_scored = ref_v[sl].sum(), scored[sl].sum()
        n_off, n_missed = off[sl].sum(), (status[sl] == MISSED).sum()
        d = dev[sl][scored[sl]]
        sections.append(Section(
            index=k, start_s=s * hop, end_s=e * hop,
            take_start_s=float(np.nanmin(take_times[sl])) if np.isfinite(take_times[sl]).any() else np.nan,
            ref_voiced_s=n_ref_v * hop, scored_s=n_scored * hop,
            off_pitch_frac=n_off / n_scored if n_scored else np.nan,
            missed_frac=n_missed / n_ref_v if n_ref_v else np.nan,
            mean_abs_dev_cents=float(np.mean(np.abs(d))) if d.size else np.nan,
            mean_dev_cents=float(np.mean(d)) if d.size else np.nan,
            off_dev_cents=float(np.mean(np.abs(dev[sl][off[sl]]))) if n_off else np.nan,
            off_direction=_direction(dev[sl][off[sl]]),
            drift_ms=float(np.nanmedian(drift[sl])) if timed[sl].any() else np.nan,
            badness=(n_off + n_missed) / n_ref_v if n_ref_v else 0.0,
        ))

    min_v = cfg.min_section_voiced_s
    ranked = sorted((s for s in sections
                     if s.ref_voiced_s >= min_v and s.badness >= cfg.min_worst_badness),
                    key=lambda s: (s.badness, np.nan_to_num(s.mean_abs_dev_cents)), reverse=True)
    timing_flags = [s for s in sections
                    if s.ref_voiced_s >= min_v and abs(np.nan_to_num(s.drift_ms)) > cfg.timing_threshold_ms]

    n_scored = scored.sum()
    return Report(
        ref_start_s=ref_start_s,
        accuracy_pct=100.0 * (n_scored - off.sum()) / n_scored if n_scored else np.nan,
        mean_abs_dev_cents=float(np.nanmean(np.abs(dev))) if n_scored else np.nan,
        mean_dev_cents=float(np.nanmean(dev)) if n_scored else np.nan,
        sung_pct=100.0 * both.sum() / ref_v.sum() if ref_v.any() else np.nan,
        mean_abs_drift_ms=float(np.nanmean(np.abs(drift))) if timed.any() else np.nan,
        raw_offset_cents=alignment.raw_offset_cents,
        applied_offset_cents=alignment.applied_offset_cents,
        key_mode=alignment.key_mode,
        key_description=describe_key(alignment.raw_offset_cents,
                                     alignment.applied_offset_cents, alignment.key_mode),
        lag_s=alignment.lag.lag_frames * hop,
        lag_ambiguous=alignment.lag.ambiguous,
        band_edge_frames=alignment.band_edge_frames,
        unsure_s=unsure.sum() * hop,
        extra_sound_s=(status == EXTRA).sum() * hop,
        threshold_cents=cfg.pitch_threshold_cents,
        pitch_time_tolerance_ms=cfg.pitch_time_tolerance_ms,
        timing_threshold_ms=cfg.timing_threshold_ms,
        sections=sections,
        worst=ranked[: cfg.n_worst_sections],
        timing_flags=timing_flags,
        frames=FrameScores(
            times_s=ref.times, ref_cents=ref_c, take_cents=take_c, deviation=dev,
            status=status, unsure=unsure, off_pitch=off, drift_ms=drift,
            take_times_s=take_times,
        ),
    )


def _clock(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def summary_text(r: Report) -> str:
    """Plain-language summary. All numbers come from the Report."""
    lines = [
        f"Pitch accuracy: {r.accuracy_pct:.0f}% of your sung notes were within "
        f"{r.threshold_cents:.0f} cents of the reference"
        + (f", compared within ±{r.pitch_time_tolerance_ms:.0f} ms " if r.pitch_time_tolerance_ms else " ")
        + f"(average miss {r.mean_abs_dev_cents:.0f} cents, "
        f"{'flat' if r.mean_dev_cents < 0 else 'sharp'} tendency {r.mean_dev_cents:+.0f} cents).",
        r.key_description,
        f"Timing: your note starts were on average {r.mean_abs_drift_ms:.0f} ms away from "
        f"your own typical timing.",
        f"You sang during {r.sung_pct:.0f}% of the reference's singing.",
    ]
    if r.worst:
        lines.append("Worst sections (song time):")
        for k, s in enumerate(r.worst, 1):
            t0, t1 = r.ref_start_s + s.start_s, r.ref_start_s + s.end_s
            if not s.scored_s:
                how = "not sung"
            elif np.isnan(s.off_dev_cents):
                how = "in tune"
            else:
                how = (f"{s.off_pitch_frac:.0%} off-pitch, by {s.off_dev_cents:.0f} cents "
                       f"({s.off_direction}) when off")
            if s.missed_frac >= 0.25:
                how += f", {s.missed_frac:.0%} not sung"
            lines.append(f"  {k}. {_clock(t0)}-{_clock(t1)}: {how}")
    else:
        lines.append("No off-pitch sections found.")
    for s in r.timing_flags:
        lines.append(f"Timing: phrase at {_clock(r.ref_start_s + s.start_s)} was "
                     f"{abs(s.drift_ms):.0f} ms {'later' if s.drift_ms > 0 else 'earlier'} "
                     f"than your usual timing.")
    if r.lag_ambiguous:
        lines.append("Warning: the start point was ambiguous (repetitive melody); "
                     "check the alignment in the graph.")
    if r.band_edge_frames:
        lines.append("Warning: timing differed more than the aligner allows; "
                     "results near those parts may be unreliable.")
    if r.unsure_s >= 0.1:
        lines.append(f"{r.unsure_s:.1f} s skipped where the pitch tracker was unsure (octave jumps).")
    return "\n".join(lines)

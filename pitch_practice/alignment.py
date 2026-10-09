"""Find where the reference sits in the take, remove the key difference,
then align frame-by-frame with banded Dynamic Time Warping.

Conventions used throughout:
  i = reference frame index, j = take frame index.
  lag: take frame j = i + lag plays the same moment as reference frame i.
       lag > 0 means the phone started recording before the song started.
  Offsets are take - reference, in cents (+1200 = take an octave higher).
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .config import AnalysisConfig
from .pitch import PitchTrack

KEY_MODES = ("snapped", "free", "absolute")


class AlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class LagEstimate:
    lag_frames: int
    offset_cents: float        # median(take - ref) at this lag, unsnapped
    cost_cents: float          # see _lag_cost; ~0 for a perfect match
    runner_up_lag_frames: int | None
    runner_up_cost_cents: float
    ambiguous: bool


@dataclass(frozen=True)
class Alignment:
    path_ref: np.ndarray       # reference frame indices along the warp path
    path_take: np.ndarray      # matching take frame indices
    lag: LagEstimate
    raw_offset_cents: float    # refined on the aligned pairs, unsnapped
    applied_offset_cents: float  # what was subtracted from the take
    key_mode: str
    band_frames: int
    band_edge_frames: int      # path cells touching the band limit
    ref_covered: tuple[int, int]  # [first, last] reference frame inside the take

    @property
    def hit_band_edge(self) -> bool:
        return self.band_edge_frames > 0


# --------------------------------------------------------------------------
# Lag + key offset
# --------------------------------------------------------------------------

def _lag_cost(ref: np.ndarray, take: np.ndarray, lag: int, n_ref_voiced: int,
              min_pairs: int, clip: float):
    """Return (cost, median) of d = take - ref at `lag`.

    cost = average over ALL voiced reference frames of min(|d - median(d)|, clip),
    where a reference frame with no voiced take frame opposite it costs `clip`.

    Why not the MAD: it is 0 whenever over half the pairs agree, so a lag
    0.3 s off (long notes still overlapping themselves) ties with the truth.
    Why divide by all reference frames, not just the overlap: otherwise a lag
    that explains 2/3 of the song perfectly beats one that explains all of it
    almost perfectly - exactly what a repeated chorus produces.
    """
    i0, i1 = max(0, -lag), min(len(ref), len(take) - lag)
    if i1 - i0 < min_pairs:
        return np.inf, np.nan
    d = take[i0 + lag:i1 + lag] - ref[i0:i1]
    d = d[~np.isnan(d)]  # NaN if either side is unvoiced
    if d.size < min_pairs:
        return np.inf, np.nan
    med = np.median(d)
    dev = np.minimum(np.abs(d - med), clip).sum()
    unexplained = (n_ref_voiced - d.size) * clip
    return float((dev + unexplained) / n_ref_voiced), float(med)


def estimate_lag(ref: PitchTrack, take: PitchTrack, cfg: AnalysisConfig) -> LagEstimate:
    """Search every lag where the reference overlaps the take.

    Key-invariant by construction: at the right lag, take - ref is roughly a
    constant (the key offset), so its spread around its median is small
    whatever that constant is. We never need to know the key to find the lag.
    """
    r, t = ref.cents, take.cents
    n, m = len(r), len(t)
    n_ref_voiced = int(np.count_nonzero(ref.voiced))
    min_pairs = max(1, int(cfg.lag_min_coverage * n_ref_voiced))

    # Range: from "reference starts before the take" (-(n-1)) to
    # "reference starts at the take's last frame" (m-1). Covers the full take.
    coarse = np.arange(-(n - 1), m, cfg.lag_coarse_step)
    clip = cfg.lag_cost_clip_cents
    costs = np.array([_lag_cost(r, t, int(L), n_ref_voiced, min_pairs, clip)[0] for L in coarse])
    if not np.isfinite(costs).any():
        raise AlignmentError(
            "could not find the reference melody in the take "
            "(too little voiced overlap at every lag)")

    best_coarse = int(coarse[np.argmin(costs)])
    fine = range(best_coarse - cfg.lag_coarse_step, best_coarse + cfg.lag_coarse_step + 1)
    lag, (cost, offset) = min(((L, _lag_cost(r, t, L, n_ref_voiced, min_pairs, clip)) for L in fine),
                              key=lambda x: x[1][0])

    # Runner-up: best coarse lag at least 1 s away from the winner.
    far = np.abs(coarse - lag) * cfg.hop_s >= 1.0
    if np.isfinite(costs[far]).any():
        k = np.flatnonzero(far)[np.argmin(costs[far])]
        runner_lag, runner_cost = int(coarse[k]), float(costs[k])
    else:
        runner_lag, runner_cost = None, np.inf

    return LagEstimate(
        lag_frames=lag, offset_cents=offset, cost_cents=cost,
        runner_up_lag_frames=runner_lag, runner_up_cost_cents=runner_cost,
        ambiguous=runner_cost - cost < cfg.lag_ambiguity_margin_cents,
    )


def applied_offset(raw_offset_cents: float, key_mode: str) -> float:
    """How much to subtract from the take before comparing pitches."""
    if key_mode == "snapped":
        return 100.0 * round(raw_offset_cents / 100.0)
    if key_mode == "free":
        return raw_offset_cents
    if key_mode == "absolute":
        return 0.0
    raise ValueError(f"key_mode must be one of {KEY_MODES}, got {key_mode!r}")


# --------------------------------------------------------------------------
# Banded DTW
# --------------------------------------------------------------------------

DIAG, REF_STEP, TAKE_STEP, START = 0, 1, 2, -1


@njit(cache=True)
def _local_cost(r, t, cap, voicing_pen):
    r_voiced, t_voiced = not np.isnan(r), not np.isnan(t)
    if r_voiced and t_voiced:
        return min(abs(r - t), cap)
    if r_voiced or t_voiced:
        return voicing_pen
    return 0.0


@njit(cache=True)
def _dtw_band(ref, take, lag, band, cap, voicing_pen, step_pen):
    """Accumulated cost over a diagonal band; returns (D, steps).

    Row i stores only columns j = i + lag - band + k, k in [0, 2*band].
    Moving from row i-1 to row i shifts the band right by one, so:
      diagonal  (i-1, j-1) is column k   in row i-1
      ref step  (i-1, j)   is column k+1 in row i-1  (take holds, ref moves)
      take step (i, j-1)   is column k-1 in row i    (ref holds, take moves)
    Row 0 may start at any j: the take is longer, so the start is free.
    """
    n, m = len(ref), len(take)
    w = 2 * band + 1
    D = np.full((n, w), np.inf)
    steps = np.full((n, w), -1, dtype=np.int8)
    for i in range(n):
        j0 = i + lag - band
        for k in range(w):
            j = j0 + k
            if j < 0 or j >= m:
                continue
            c = _local_cost(ref[i], take[j], cap, voicing_pen)
            if i == 0:
                D[0, k] = c
                continue
            best, arg = D[i - 1, k], 0
            if k + 1 < w and D[i - 1, k + 1] + step_pen < best:
                best, arg = D[i - 1, k + 1] + step_pen, 1
            if k >= 1 and D[i, k - 1] + step_pen < best:
                best, arg = D[i, k - 1] + step_pen, 2
            D[i, k] = best + c
            steps[i, k] = arg
    return D, steps


def _backtrack(D, steps, lag, band):
    i, k = D.shape[0] - 1, int(np.argmin(D[-1]))  # free end in the take
    path = []
    while True:
        path.append((i, i + lag - band + k, k))
        s = steps[i, k]
        if s == START:
            break
        if s == DIAG:
            i -= 1
        elif s == REF_STEP:
            i, k = i - 1, k + 1
        else:
            k -= 1
    path.reverse()
    return np.array(path)


def dtw_banded(ref_cents, take_cents, lag, band, cfg: AnalysisConfig):
    """Align two cents sequences. Returns (path_ref, path_take, edge_count, cost)."""
    ref_cents = np.ascontiguousarray(ref_cents, dtype=np.float64)
    take_cents = np.ascontiguousarray(take_cents, dtype=np.float64)
    D, steps = _dtw_band(ref_cents, take_cents, lag, band, cfg.dtw_cost_cap_cents,
                         cfg.dtw_voicing_penalty, cfg.dtw_step_penalty)
    if not np.isfinite(D[-1]).any():
        raise AlignmentError("no valid alignment path inside the band")
    path = _backtrack(D, steps, lag, band)
    i, j, k = path.T
    # A band-edge cell only counts if the edge is the band's limit, not the
    # start/end of the take itself - including when the reference "should" be
    # past the end of the take (the take ended: a coverage issue, not timing).
    m, w = len(take_cents), 2 * band + 1
    centre = i + lag
    inside = (centre >= 0) & (centre <= m - 1)
    at_edge = inside & (((k == 0) & (j > 0)) | ((k == w - 1) & (j < m - 1)))
    return i, j, int(np.count_nonzero(at_edge)), float(D[-1].min())


# --------------------------------------------------------------------------
# Full alignment
# --------------------------------------------------------------------------

def align(ref: PitchTrack, take: PitchTrack, cfg: AnalysisConfig = AnalysisConfig()) -> Alignment:
    if cfg.key_mode not in KEY_MODES:
        raise ValueError(f"key_mode must be one of {KEY_MODES}, got {cfg.key_mode!r}")
    lag = estimate_lag(ref, take, cfg)
    band = int(round(cfg.dtw_band_s / cfg.hop_s))
    r, t = ref.cents, take.cents

    # Only reference frames whose band reaches into the take can be aligned.
    i_lo = max(0, -lag.lag_frames - band)
    i_hi = min(len(r) - 1, len(t) - 1 - lag.lag_frames + band)
    r_cov = r[i_lo:i_hi + 1]
    lag_cov = lag.lag_frames + i_lo

    def run(offset):
        pi, pj, edge, _ = dtw_banded(r_cov, t - offset, lag_cov, band, cfg)
        d = t[pj] - r_cov[pi]
        d = d[~np.isnan(d)]
        refined = float(np.median(d)) if d.size else offset
        return pi + i_lo, pj, edge, refined

    applied = applied_offset(lag.offset_cents, cfg.key_mode)
    pi, pj, edge, raw = run(applied)
    # The refined offset (on aligned pairs) is more accurate than the lag
    # search's. If it snaps to a different semitone, align once more.
    if applied_offset(raw, cfg.key_mode) != applied:
        applied = applied_offset(raw, cfg.key_mode)
        pi, pj, edge, raw = run(applied)

    return Alignment(
        path_ref=pi, path_take=pj, lag=lag,
        raw_offset_cents=raw, applied_offset_cents=applied, key_mode=cfg.key_mode,
        band_frames=band, band_edge_frames=edge, ref_covered=(i_lo, i_hi),
    )

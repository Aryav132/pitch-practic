"""Turn a Report into a few plain-language tips.

Each detector looks for one singing habit in the per-note measurements. All
numbers in a tip come from the Report; only the wording and the practice
suggestion are fixed templates. (A Phase 3 LLM coach may rephrase these,
but never invent the numbers.)

Detectors are heuristics. Each one is tested to fire on a synthetic take
that has the habit, and to stay silent on a perfect take.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig
from .scoring import MISSED, Report, note_spans

# A habit must show up on at least this many notes, and this share of the
# notes that could show it, before we mention it.
MIN_NOTES = 3
MIN_SHARE = 0.25

HEAD_S = 0.12        # "start of the note" window
SETTLE_S = 0.15      # body of the note starts after this
SCOOP_CENTS = 40     # start this far off, then settle within BODY_OK_CENTS
BODY_OK_CENTS = 30
LONG_NOTE_S = 0.6
DRIFT_CENTS = 30     # last third vs first third of a long note
HIGH_GAP_CENTS = 25  # high notes this much flatter/sharper than the others
ENTRY_LATE_MS = 100  # median lateness of phrase entries
OVERALL_CENTS = 20


@dataclass(frozen=True)
class Tip:
    kind: str
    title: str
    detail: str          # what we measured, in plain words
    try_this: str        # practice suggestion (generic, not a measurement)
    where: list[tuple[float, float]] = field(default_factory=list)  # song times
    impact: float = 0.0  # for ranking: roughly "seconds of singing affected"


def describe_cents(c: float) -> str:
    """Cents in words. 100 cents = one semitone = one piano key."""
    c = abs(c)
    if c < 20:
        return "very slightly"
    if c < 35:
        return "a little (about a quarter of a piano key)"
    if c < 65:
        return "noticeably (about half a piano key)"
    if c < 130:
        return "by about a whole piano key (a semitone)"
    return f"by about {round(c / 100)} piano keys"


def low_high(c: float) -> str:
    return "too low (flat)" if c < 0 else "too high (sharp)"


def _clock(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


@dataclass
class _Note:
    start: int
    end: int
    dev: np.ndarray      # scored deviations within the note (NaN = not scored)
    ref_pitch: float


def _notes(r: Report, cfg: AnalysisConfig) -> list[_Note]:
    f = r.frames
    out = []
    for s, e in note_spans(f.ref_cents, cfg):
        dev = f.deviation[s:e]
        if (e - s) * cfg.hop_s >= 0.25 and np.isfinite(dev).mean() >= 0.6:
            out.append(_Note(s, e, dev, float(np.nanmedian(f.ref_cents[s:e]))))
    return out


def _span(r: Report, cfg: AnalysisConfig, n: _Note) -> tuple[float, float]:
    return (r.ref_start_s + n.start * cfg.hop_s, r.ref_start_s + n.end * cfg.hop_s)


def _where(r, cfg, notes, key=None, k=3):
    """Song-time ranges of the k most affected notes, in time order."""
    chosen = sorted(notes, key=key, reverse=True)[:k] if key else notes[:k]
    return sorted(_span(r, cfg, n) for n in chosen)


def _scooping(r, cfg, notes):
    head, settle = round(HEAD_S / cfg.hop_s), round(SETTLE_S / cfg.hop_s)
    flat_in, sharp_in, eligible = [], [], 0
    for n in notes:
        h, b = n.dev[:head], n.dev[settle:]
        if np.isfinite(h).sum() < head // 2 or np.isfinite(b).sum() < 5:
            continue
        eligible += 1
        mh, mb = np.nanmedian(h), np.nanmedian(b)
        if abs(mb) < BODY_OK_CENTS and abs(mh) >= SCOOP_CENTS:
            (flat_in if mh < 0 else sharp_in).append((n, mh))
    tips = []
    for hits, word, fix in [
        (flat_in, "low", "Hear the note in your head before you sing it and aim for its centre "
                         "from the very first moment. Humming the first note of a phrase "
                         "before singing it helps."),
        (sharp_in, "high", "Aim slightly lower as each note begins and land on it gently, "
                           "rather than pushing into it."),
    ]:
        if len(hits) >= MIN_NOTES and len(hits) / max(eligible, 1) >= MIN_SHARE:
            size = float(np.median([abs(m) for _, m in hits]))
            tips.append(Tip(
                kind=f"scoop_{word}",
                title=f"You start notes too {word}, then slide onto them",
                detail=(f"On {len(hits)} of {eligible} notes, the first moment was "
                        f"{describe_cents(size)} too {word} before you settled on the right pitch."),
                try_this=fix,
                where=_where(r, cfg, [n for n, _ in hits], key=lambda n: abs(np.nanmedian(n.dev[:head]))),
                impact=len(hits) * HEAD_S * size / 100 * 2,
            ))
    return tips


def _long_note_drift(r, cfg, notes):
    long = [n for n in notes if (n.end - n.start) * cfg.hop_s >= LONG_NOTE_S]
    drops = []
    for n in long:
        third = len(n.dev) // 3
        a, b = np.nanmedian(n.dev[:third]), np.nanmedian(n.dev[-third:])
        if np.isfinite(a) and np.isfinite(b):
            drops.append((n, b - a))
    tips = []
    for sign, word, fix in [(-1, "flat", "Keep your breath support steady to the end of long "
                                         "notes; the pitch sags when the air pressure drops."),
                            (+1, "sharp", "Relax as you hold long notes; pushing more air or "
                                          "tension toward the end raises the pitch.")]:
        hits = [(n, d) for n, d in drops if sign * d >= DRIFT_CENTS]
        if len(hits) >= 2 and len(hits) / max(len(drops), 1) >= MIN_SHARE:
            size = float(np.median([abs(d) for _, d in hits]))
            tips.append(Tip(
                kind=f"drift_{word}",
                title=f"You go {word} while holding long notes",
                detail=(f"On {len(hits)} of {len(drops)} long notes, the end was "
                        f"{describe_cents(size)} {'lower' if sign < 0 else 'higher'} than the start."),
                try_this=fix,
                where=_where(r, cfg, [n for n, _ in hits], key=lambda n: n.end - n.start),
                impact=sum((n.end - n.start) * cfg.hop_s / 3 for n, _ in hits) * size / 100,
            ))
    return tips


def _high_notes(r, cfg, notes):
    rated = [(n, np.nanmedian(n.dev[round(SETTLE_S / cfg.hop_s):])) for n in notes]
    rated = [(n, d) for n, d in rated if np.isfinite(d)]
    if len(rated) < 8:
        return []
    cut = np.percentile([n.ref_pitch for n, _ in rated], 75)
    high = [(n, d) for n, d in rated if n.ref_pitch >= cut]
    rest = [d for n, d in rated if n.ref_pitch < cut]
    if len(high) < MIN_NOTES:
        return []
    h, o = float(np.median([d for _, d in high])), float(np.median(rest))
    gap = h - o
    if abs(gap) < HIGH_GAP_CENTS or abs(h) < BODY_OK_CENTS:
        return []
    word = "flat" if gap < 0 else "sharp"
    return [Tip(
        kind=f"high_{word}",
        title=f"Your high notes are {word}",
        detail=(f"Your {len(high)} highest notes were {describe_cents(h)} {low_high(h)}, "
                f"while your other notes were {describe_cents(o)} off."),
        try_this=("Support high notes with more breath and think of the pitch slightly higher "
                  "as you climb." if word == "flat" else
                  "Don't reach for high notes; think of them slightly lower and keep your "
                  "throat relaxed."),
        where=_where(r, cfg, [n for n, _ in high], key=lambda n: n.ref_pitch),
        impact=sum((n.end - n.start) * cfg.hop_s for n, _ in high) * abs(gap) / 100,
    )]


def _phrase_entries(r, cfg):
    f = r.frames
    gap = round(cfg.phrase_min_gap_s / cfg.hop_s)
    voiced = ~np.isnan(f.ref_cents)
    entries = [i for i in np.flatnonzero(np.isfinite(f.drift_ms))
               if i >= gap and not voiced[i - gap:i].any()]
    if len(entries) < MIN_NOTES:
        return []
    med = float(np.median(f.drift_ms[entries]))
    if abs(med) < ENTRY_LATE_MS:
        return []
    late = med > 0
    worst = sorted(entries, key=lambda i: abs(f.drift_ms[i]), reverse=True)[:3]
    return [Tip(
        kind="entries_late" if late else "entries_early",
        title=f"You come in {'late' if late else 'early'} after pauses",
        detail=(f"When a new phrase starts, you typically come in about {abs(med):.0f} ms "
                f"{'after' if late else 'before'} the singer (compared with your timing elsewhere)."),
        try_this=("Breathe in during the rest and be ready before the singer's next phrase; "
                  "keep counting the beat through pauses." if late else
                  "Wait for the singer's breath; count the rest fully before coming in."),
        where=sorted((r.ref_start_s + i * cfg.hop_s - 0.3, r.ref_start_s + i * cfg.hop_s + 1.0)
                     for i in worst),
        impact=len(entries) * abs(med) / 1000 * 2,
    )]


def _overall(r, covered: set[str]):
    c = r.mean_dev_cents
    if not np.isfinite(c) or abs(c) < OVERALL_CENTS or covered & {"high_flat", "high_sharp"}:
        return []
    word = "flat" if c < 0 else "sharp"
    return [Tip(
        kind=f"overall_{word}",
        title=f"Overall you sing slightly {word}",
        detail=f"On average your notes were {describe_cents(c)} {low_high(c)}.",
        try_this=("Before singing, play the first note and hum it until it feels locked in. "
                  "Listen for the singer's pitch, not just the melody's shape."),
        impact=abs(c) / 100 * 2,
    )]


def _skipped(r):
    tips = []
    for s in r.sections:
        if s.ref_voiced_s >= 0.5 and np.nan_to_num(s.missed_frac) >= 0.5:
            tips.append(Tip(
                kind="skipped",
                title=f"You didn't sing at {_clock(r.ref_start_s + s.start_s)}",
                detail=f"The singer sings here, but we heard little or nothing from you "
                       f"({s.missed_frac:.0%} of this part).",
                try_this="Listen to this part a few times, then sing just this line on its own.",
                where=[(r.ref_start_s + s.start_s, r.ref_start_s + s.end_s)],
                impact=s.ref_voiced_s * s.missed_frac,
            ))
    return tips


def _worst_section(r):
    if not r.worst:
        return []
    s = r.worst[0]
    if not s.scored_s or not s.off_direction:
        return []
    direction = {"flat": "mostly too low", "sharp": "mostly too high"}.get(s.off_direction,
                                                                         "both too high and too low")
    return [Tip(
        kind="worst_section",
        title=f"Practise {_clock(r.ref_start_s + s.start_s)}–{_clock(r.ref_start_s + s.end_s)}",
        detail=(f"Your hardest part: {s.off_pitch_frac:.0%} of it was off, {direction}, "
                f"by {describe_cents(s.off_dev_cents).replace('by ', '')}."),
        try_this="Loop this part: listen to the singer, sing it slowly on 'aah', then with the words.",
        where=[(r.ref_start_s + s.start_s, r.ref_start_s + s.end_s)],
        impact=s.scored_s * s.off_pitch_frac * s.off_dev_cents / 100,
    )]


def coach(r: Report, cfg: AnalysisConfig = AnalysisConfig(), max_tips: int = 3) -> list[Tip]:
    """The few most useful tips, most important first."""
    notes = _notes(r, cfg)
    habits = (_scooping(r, cfg, notes) + _long_note_drift(r, cfg, notes)
              + _high_notes(r, cfg, notes) + _phrase_entries(r, cfg))
    habits += _overall(r, {t.kind for t in habits})
    tips = sorted(habits + _skipped(r), key=lambda t: t.impact, reverse=True)
    # A habit tells you what to change; the worst section tells you where to
    # practise. Always keep one slot for the latter if there is one.
    worst = _worst_section(r)
    return tips[: max_tips - len(worst)] + worst if worst else tips[:max_tips]

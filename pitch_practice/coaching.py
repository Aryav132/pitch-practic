"""Turn a Report into a few tips, in plain words or in music terms.

Each detector looks for one singing habit in the per-note measurements. All
numbers in a tip come from the Report; only the wording, the practice advice
and the beginner exercise are fixed templates. The same measurements are
written two ways: `expert=False` (no music vocabulary: "too low", "a small
step") and `expert=True` ("flat", "semitone", cents, ms). A Phase 3 LLM coach
may rephrase these, but never invent the numbers.

Detectors are heuristics. Each one is tested to fire on a synthetic take that
has the habit, and to stay silent on a perfect take.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig
from .scoring import Report, note_spans

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
    detail: str          # what we measured, in words for the chosen audience
    try_this: str        # practice advice (generic, not a measurement)
    where: list[tuple[float, float]] = field(default_factory=list)  # song times
    impact: float = 0.0  # for ranking: roughly "seconds of singing affected"


# ---------------------------------------------------------------- wording

def describe_cents(c: float) -> str:
    """Cents in music terms. 100 cents = one semitone = one piano key."""
    c = abs(c)
    if c < 20:
        return "very slightly"
    if c < 35:
        return "a little (about a quarter of a semitone)"
    if c < 65:
        return f"noticeably (about {c:.0f} cents, half a semitone)"
    if c < 130:
        return f"by about a semitone ({c:.0f} cents)"
    return f"by about {round(c / 100)} semitones"


def plain_size(c: float) -> str:
    """Cents with no music vocabulary."""
    c = abs(c)
    if c < 20:
        return "just a tiny bit"
    if c < 35:
        return "a little"
    if c < 65:
        return "noticeably"
    if c < 130:
        return "a lot (about one full step)"
    return "very far (several steps)"


def plain_seconds(ms: float) -> str:
    s = abs(ms) / 1000
    for limit, words in [(0.15, "a split second"), (0.2, "about a fifth of a second"),
                         (0.35, "about a quarter of a second"), (0.6, "about half a second")]:
        if s < limit:
            return words
    return f"about {s:.1f} seconds"


@dataclass(frozen=True)
class Words:
    expert: bool

    def size(self, c):              # "noticeably" / "noticeably (about 55 cents...)"
        return describe_cents(c) if self.expert else plain_size(c)

    def low_high(self, c):
        if self.expert:
            return "flat (too low)" if c < 0 else "sharp (too high)"
        return "too low" if c < 0 else "too high"

    def late(self, ms):
        return f"about {abs(ms):.0f} ms" if self.expert else plain_seconds(ms)


def describe_key_plain(raw_offset_cents: float, key_mode: str) -> str:
    """The key difference with no music words. 1200 cents = an octave =
    "a whole set of notes" (e.g. a man singing a song recorded by a woman)."""
    c = raw_offset_cents
    if abs(c) < 50:
        return "You sang at the same height as the singer."
    steps = round(abs(c) / 100)
    where = "higher" if c > 0 else "lower"
    if steps % 12 == 0:
        sets = steps // 12
        size = "a whole set of notes" if sets == 1 else f"{sets} whole sets of notes"
        example = (" (common when a man sings a woman's song)" if c < 0 else
                   " (common when a woman sings a man's song)") if sets == 1 else ""
        text = f"You sang everything {size} {where} than the singer{example}."
    else:
        text = f"You sang everything about {steps} small steps {where} than the singer."
    if key_mode == "absolute":
        return text + " Your settings count this as a mistake."
    return text + " That's fine: we compare the tune, not the height."


def _clock(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


# ---------------------------------------------------------------- notes

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


@dataclass(frozen=True)
class NoteVerdict:
    start_s: float       # song time
    end_s: float
    verdict: str         # "ok" | "low" | "high" | "missed" | "unclear"
    dev_cents: float     # median deviation of the settled part (NaN if none)


def note_verdicts(r: Report, cfg: AnalysisConfig = AnalysisConfig()) -> list[NoteVerdict]:
    """One verdict per reference note: the beginner's "tuner strip".
    Judged on the settled part of the note (after SETTLE_S), so a slide into
    the note doesn't decide the verdict on its own."""
    f, settle = r.frames, round(SETTLE_S / cfg.hop_s)
    out = []
    for s, e in note_spans(f.ref_cents, cfg):
        if (e - s) * cfg.hop_s < 0.15:
            continue
        t0, t1 = r.ref_start_s + s * cfg.hop_s, r.ref_start_s + e * cfg.hop_s
        body = f.deviation[s + min(settle, (e - s) // 2):e]
        sang = np.isfinite(f.take_cents[s:e]).mean()
        if sang < 0.3:
            out.append(NoteVerdict(t0, t1, "missed", np.nan))
        elif np.isfinite(body).sum() < 3:
            out.append(NoteVerdict(t0, t1, "unclear", np.nan))
        else:
            d = float(np.nanmedian(body))
            v = "ok" if abs(d) <= r.threshold_cents else ("low" if d < 0 else "high")
            out.append(NoteVerdict(t0, t1, v, d))
    return out


# ---------------------------------------------------------------- detectors

def _scooping(r, cfg, notes, w: Words):
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
    for hits, word, slide, expert_fix, beginner_fix in [
        (flat_in, "low", "up",
         "Hear the note in your head before you sing it and aim for its centre from the very "
         "first moment. Humming the first note of a phrase before singing it helps.",
         "Play the singer's version, then hum just the first note of each word for 2 seconds. "
         "Now sing the line, starting each word right on that hummed note. Repeat 3 times."),
        (sharp_in, "high", "down",
         "Aim slightly lower as each note begins and land on it gently, rather than pushing "
         "into it.",
         "Sing the line softly, as if you're telling a secret. Gentle starts land on the note "
         "instead of jumping above it. Repeat 3 times, then sing it normally."),
    ]:
        if len(hits) >= MIN_NOTES and len(hits) / max(eligible, 1) >= MIN_SHARE:
            size = float(np.median([abs(m) for _, m in hits]))
            tips.append(Tip(
                kind=f"scoop_{word}",
                title=(f"You start notes too {word}, then slide onto them" if w.expert else
                       f"You start notes too {word}, then slide {slide}"),
                detail=(f"On {len(hits)} of {eligible} notes, the first moment was "
                        f"{w.size(size)} too {word} before you settled on the right note."),
                try_this=expert_fix if w.expert else beginner_fix,
                where=_where(r, cfg, [n for n, _ in hits], key=lambda n: abs(np.nanmedian(n.dev[:head]))),
                impact=len(hits) * HEAD_S * size / 100 * 2,
            ))
    return tips


def _long_note_drift(r, cfg, notes, w: Words):
    long = [n for n in notes if (n.end - n.start) * cfg.hop_s >= LONG_NOTE_S]
    drops = []
    for n in long:
        third = len(n.dev) // 3
        a, b = np.nanmedian(n.dev[:third]), np.nanmedian(n.dev[-third:])
        if np.isfinite(a) and np.isfinite(b):
            drops.append((n, b - a))
    tips = []
    for sign, word, plain_title, expert_fix, beginner_fix in [
        (-1, "flat", "Your long notes sink lower as you hold them",
         "Keep your breath support steady to the end of long notes; the pitch sags when the "
         "air pressure drops.",
         "Take a bigger breath before long notes. Hold an 'aah' for 5 seconds while gently "
         "pressing your belly in, keeping the sound exactly the same to the end. Repeat 3 times."),
        (+1, "sharp", "Your long notes creep higher as you hold them",
         "Relax as you hold long notes; pushing more air or tension toward the end raises the "
         "pitch.",
         "Hold an 'aah' for 5 seconds and keep your shoulders and jaw loose. Don't get louder "
         "toward the end. Repeat 3 times."),
    ]:
        hits = [(n, d) for n, d in drops if sign * d >= DRIFT_CENTS]
        if len(hits) >= 2 and len(hits) / max(len(drops), 1) >= MIN_SHARE:
            size = float(np.median([abs(d) for _, d in hits]))
            tips.append(Tip(
                kind=f"drift_{word}",
                title=f"You go {word} while holding long notes" if w.expert else plain_title,
                detail=(f"On {len(hits)} of {len(drops)} long notes, the end was "
                        f"{w.size(size)} {'lower' if sign < 0 else 'higher'} than the start."),
                try_this=expert_fix if w.expert else beginner_fix,
                where=_where(r, cfg, [n for n, _ in hits], key=lambda n: n.end - n.start),
                impact=sum((n.end - n.start) * cfg.hop_s / 3 for n, _ in hits) * size / 100,
            ))
    return tips


def _high_notes(r, cfg, notes, w: Words):
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
    flat = gap < 0
    word = "flat" if flat else "sharp"
    if w.expert:
        title = f"Your high notes are {word}"
        fix = ("Support high notes with more breath and think of the pitch slightly higher as "
               "you climb." if flat else
               "Don't reach for high notes; think of them slightly lower and keep your throat "
               "relaxed.")
    else:
        title = f"Your highest notes come out too {'low' if flat else 'high'}"
        fix = ("Play the singer's version and hum along only on the highest note, matching "
               "it until it sounds the same. Then sing the whole line. Repeat 3 times." if flat
               else "On the highest notes, sing a little softer and don't strain. Hum the "
                    "highest note quietly with the singer, then sing the line. Repeat 3 times.")
    return [Tip(
        kind=f"high_{word}",
        title=title,
        detail=(f"Your {len(high)} highest notes were {w.size(h)} {w.low_high(h)}, "
                f"while your other notes were {w.size(o)} off."),
        try_this=fix,
        where=_where(r, cfg, [n for n, _ in high], key=lambda n: n.ref_pitch),
        impact=sum((n.end - n.start) * cfg.hop_s for n, _ in high) * abs(gap) / 100,
    )]


def _phrase_entries(r, cfg, w: Words):
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
    if w.expert:
        title = f"You come in {'late' if late else 'early'} after pauses"
        fix = ("Breathe in during the rest and be ready before the singer's next phrase; keep "
               "counting the beat through pauses." if late else
               "Wait for the singer's breath; count the rest fully before coming in.")
    else:
        title = f"You start lines {'late' if late else 'too early'} after a pause"
        fix = ("Tap the beat on your leg while the song plays. During each pause, take a "
               "breath on the last two taps so you're ready to sing on time." if late else
               "During each pause, wait until you hear the singer breathe in, then start "
               "together with them.")
    return [Tip(
        kind="entries_late" if late else "entries_early",
        title=title,
        detail=(f"When a new line starts, you typically come in {w.late(med)} "
                f"{'after' if late else 'before'} the singer (compared with your timing elsewhere)."),
        try_this=fix,
        where=sorted((r.ref_start_s + i * cfg.hop_s - 0.3, r.ref_start_s + i * cfg.hop_s + 1.0)
                     for i in worst),
        impact=len(entries) * abs(med) / 1000 * 2,
    )]


def _overall(r, covered: set[str], w: Words):
    c = r.mean_dev_cents
    if not np.isfinite(c) or abs(c) < OVERALL_CENTS or covered & {"high_flat", "high_sharp"}:
        return []
    word = "flat" if c < 0 else "sharp"
    return [Tip(
        kind=f"overall_{word}",
        title=(f"Overall you sing slightly {word}" if w.expert else
               f"Overall you sing a little too {'low' if c < 0 else 'high'}"),
        detail=f"On average your notes were {w.size(c)} {w.low_high(c)}.",
        try_this=("Before singing, play the first note and hum it until it feels locked in. "
                  "Listen for the singer's pitch, not just the melody's shape." if w.expert else
                  "Before each try, play the song's first line and hum along until your hum "
                  "and the singer blend into one sound. Then sing."),
        impact=abs(c) / 100 * 2,
    )]


def _skipped(r, w: Words):
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


def _worst_section(r, w: Words):
    if not r.worst:
        return []
    s = r.worst[0]
    if not s.scored_s or not s.off_direction:
        return []
    direction = {"flat": "mostly too low", "sharp": "mostly too high"}.get(
        s.off_direction, "sometimes too high, sometimes too low")
    if w.expert:
        size = describe_cents(s.off_dev_cents).replace("by ", "")
        fix = "Loop this part: listen to the singer, sing it slowly on 'aah', then with the words."
    else:
        size = plain_size(s.off_dev_cents)
        fix = ("Use 'Hear the difference' below a few times. Then sing this part on 'aah' "
               "(no words) along with the singer, then once more with the words.")
    return [Tip(
        kind="worst_section",
        title=f"Practise {_clock(r.ref_start_s + s.start_s)}–{_clock(r.ref_start_s + s.end_s)}",
        detail=f"Your hardest part: {s.off_pitch_frac:.0%} of it was off, {direction}, {size}.",
        try_this=fix,
        where=[(r.ref_start_s + s.start_s, r.ref_start_s + s.end_s)],
        impact=s.scored_s * s.off_pitch_frac * s.off_dev_cents / 100,
    )]


def coach(r: Report, cfg: AnalysisConfig = AnalysisConfig(), max_tips: int = 3,
          expert: bool = False) -> list[Tip]:
    """The few most useful tips, most important first."""
    w = Words(expert)
    notes = _notes(r, cfg)
    habits = (_scooping(r, cfg, notes, w) + _long_note_drift(r, cfg, notes, w)
              + _high_notes(r, cfg, notes, w) + _phrase_entries(r, cfg, w))
    habits += _overall(r, {t.kind for t in habits}, w)
    tips = sorted(habits + _skipped(r, w), key=lambda t: t.impact, reverse=True)
    # A habit tells you what to change; the worst section tells you where to
    # practise. Always keep one slot for the latter if there is one.
    worst = _worst_section(r, w)
    return tips[: max_tips - len(worst)] + worst if worst else tips[:max_tips]

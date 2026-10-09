import pytest

from pitch_practice.alignment import align
import re

from pitch_practice.coaching import (
    coach, describe_cents, describe_key_plain, note_verdicts, plain_size)
from pitch_practice.config import AnalysisConfig
from pitch_practice.scoring import score
from tests.synth import MELODY, melody_track

CFG = AnalysisConfig()
REF = melody_track(MELODY)


def report_for(take_notes, ref=REF):
    take = melody_track(take_notes)
    return score(ref, take, align(ref, take, CFG), CFG)


def tips_for(take_notes, ref=REF, expert=False):
    return coach(report_for(take_notes, ref), CFG, expert=expert)


def kinds(tips):
    return [t.kind for t in tips]


def each_note(fn):
    out = []
    for c, d in MELODY:
        out += [(None, d)] if c is None else fn(c, d)
    return out


def test_perfect_take_gets_no_tips():
    assert tips_for(MELODY) == []


def test_scooping_up_into_notes():
    take = each_note(lambda c, d: [(c - 80, 0.1), (c, d - 0.1)])
    tips = tips_for(take)
    assert tips[0].kind == "scoop_low"
    assert "too low" in tips[0].title
    assert not {"drift_flat", "high_flat", "overall_flat"} & set(kinds(tips))


def test_going_flat_on_long_notes():
    take = each_note(lambda c, d: [(c, d / 3), (c - 20, d / 3), (c - 50, d / 3)]
                     if d >= 0.6 else [(c, d)])
    assert "drift_flat" in kinds(tips_for(take))


def test_high_notes_flat():
    take = each_note(lambda c, d: [(c - 50 if c >= 6700 else c, d)])
    tips = tips_for(take)
    assert "high_flat" in kinds(tips)
    assert "overall_flat" not in kinds(tips)     # not double-reported


def test_late_phrase_entries():
    take, prev_rest = [], False
    for c, d in MELODY:
        if c is None:
            take.append((None, d + 0.2)); prev_rest = True
        elif prev_rest:
            take.append((c, d - 0.2)); prev_rest = False
        else:
            take.append((c, d))
    plain = next(t for t in tips_for(take) if t.kind == "entries_late")
    expert = next(t for t in tips_for(take, expert=True) if t.kind == "entries_late")
    assert "about a quarter of a second" in plain.detail
    assert "about 200 ms" in expert.detail


def test_skipped_phrase():
    take = [(None, d) if 10 <= k <= 12 else (c, d) for k, (c, d) in enumerate(MELODY)]
    assert "skipped" in kinds(tips_for(take))


def test_one_wrong_note_is_not_called_a_habit():
    wrong = list(MELODY)
    wrong[7] = (MELODY[7][0] + 300, 0.5)
    wrong.insert(8, (MELODY[7][0], 0.2))
    assert kinds(tips_for(wrong)) == ["worst_section"]


def test_tips_point_at_song_time():
    take = each_note(lambda c, d: [(c - 80, 0.1), (c, d - 0.1)])
    for t0, t1 in tips_for(take)[0].where:
        assert 0 <= t0 < t1 <= 10


@pytest.mark.parametrize("cents,expert,plain", [
    (10, "very slightly", "tiny bit"), (25, "quarter", "a little"), (50, "half a semitone", "noticeably"),
    (95, "about a semitone", "one full step"), (240, "2 semitones", "several steps")])
def test_size_words(cents, expert, plain):
    assert expert in describe_cents(cents)
    assert plain in plain_size(cents)


def test_demo_finds_exactly_the_planted_habits(tmp_path):
    # Full audio pipeline (synthetic voice -> pYIN -> DTW -> scoring -> coach).
    from pitch_practice.demo import write_demo
    from pitch_practice.pipeline import analyze_files
    ref, take = write_demo(tmp_path)
    tips = coach(analyze_files(ref, take), CFG)
    assert kinds(tips) == ["entries_late", "high_flat", "worst_section"]
    assert "about a quarter of a second" in tips[0].detail
    t0, t1 = tips[2].where[0]
    assert t0 <= 6.0 <= t1          # the note held a semitone low (5.7-6.7 s)


JARGON = re.compile(r"\b(flat|sharp|semitones?|cents?|ms|pitch|phrase|key|octave)\b", re.I)


@pytest.mark.parametrize("make_take", [
    lambda: [(None, d) if c is None else (c - 50 if c >= 6700 else c, d) for c, d in MELODY],
    lambda: each_note(lambda c, d: [(c - 80, 0.1), (c, d - 0.1)]),
    lambda: each_note(lambda c, d: [(c, d / 3), (c - 20, d / 3), (c - 50, d / 3)] if d >= 0.6 else [(c, d)]),
    lambda: [(c - 45 if c else None, d) for c, d in MELODY],
])
def test_simple_mode_has_no_music_jargon(make_take):
    tips = tips_for(make_take())
    assert tips
    for t in tips:
        text = " ".join([t.title, t.detail, t.try_this])
        assert not JARGON.search(text), (t.kind, JARGON.findall(text))


def test_expert_mode_keeps_music_terms():
    tips = tips_for([(c - 50 if c and c >= 6700 else c, d) for c, d in MELODY], expert=True)
    assert any("flat" in t.title for t in tips)


def test_note_verdicts_tuner_strip():
    wrong = list(MELODY)
    wrong[3] = (MELODY[3][0] - 100, MELODY[3][1])      # 4th note a full step low
    v = note_verdicts(report_for(wrong), CFG)
    verdicts = [x.verdict for x in v]
    assert verdicts.count("low") == 1 and verdicts[3] == "low"
    assert verdicts.count("ok") == len(verdicts) - 1
    assert v[3].start_s == pytest.approx(1.5, abs=0.05)


def test_note_verdicts_marks_missed_notes():
    skipped = [(None, d) if k == 5 else (c, d) for k, (c, d) in enumerate(MELODY)]
    assert "missed" in [x.verdict for x in note_verdicts(report_for(skipped), CFG)]


@pytest.mark.parametrize("cents,mode,expected", [
    (8, "snapped", "same height"),
    (-1195, "snapped", "a whole set of notes lower"),
    (1210, "snapped", "a whole set of notes higher"),
    (-300, "snapped", "about 3 small steps lower"),
    (-1200, "absolute", "count this as a mistake"),
])
def test_key_in_plain_words(cents, mode, expected):
    text = describe_key_plain(cents, mode)
    assert expected in text
    assert not JARGON.search(text), JARGON.findall(text)

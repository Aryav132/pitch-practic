import pytest

from pitch_practice.alignment import align
from pitch_practice.coaching import coach, describe_cents
from pitch_practice.config import AnalysisConfig
from pitch_practice.scoring import score
from tests.synth import MELODY, melody_track

CFG = AnalysisConfig()
REF = melody_track(MELODY)


def tips_for(take_notes, ref=REF):
    take = melody_track(take_notes)
    return coach(score(ref, take, align(ref, take, CFG), CFG), CFG)


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
    tip = next(t for t in tips_for(take) if t.kind == "entries_late")
    assert "about 200 ms" in tip.detail


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


@pytest.mark.parametrize("cents,words", [
    (10, "very slightly"), (25, "quarter"), (50, "half"), (95, "whole piano key"), (240, "2 piano keys")])
def test_describe_cents(cents, words):
    assert words in describe_cents(cents)


def test_demo_finds_exactly_the_planted_habits(tmp_path):
    # Full audio pipeline (synthetic voice -> pYIN -> DTW -> scoring -> coach).
    from pitch_practice.demo import write_demo
    from pitch_practice.pipeline import analyze_files
    ref, take = write_demo(tmp_path)
    tips = coach(analyze_files(ref, take), CFG)
    assert kinds(tips) == ["entries_late", "high_flat", "worst_section"]
    assert "about 250 ms" in tips[0].detail
    t0, t1 = tips[2].where[0]
    assert t0 <= 6.0 <= t1          # the note held a semitone low (5.7-6.7 s)

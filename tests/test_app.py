"""App tests with Streamlit's AppTest (no browser). Uploads can't be
simulated here, but the built-in demo exercises the full results page."""

from streamlit.testing.v1 import AppTest


def app():
    return AppTest.from_file("../app.py", default_timeout=60).run()


def test_app_renders_and_waits_for_both_files():
    at = app()
    assert not at.exception
    assert at.title[0].value == "Sing along. See exactly what to fix."
    analyse = next(b for b in at.button if b.label == "Analyse my singing")
    assert analyse.disabled


def test_demo_shows_plain_language_tips():
    at = app()
    next(b for b in at.button if "See an example" in b.label).click().run()
    assert not at.exception
    assert any("%</div>" in m.value for m in at.markdown)           # score badge
    cards = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert cards == ["#### 1. You start lines late after a pause",
                     "#### 2. Your highest notes come out too low",
                     "#### 3. Practise 0:05.9–0:08.4"]
    # Simple mode: one "hear the difference" player per tip, plus the three
    # separate players folded away in an expander.
    assert len(at.get("audio")) >= 12


def test_picking_a_phrase_zooms_and_explains_it():
    at = app()
    next(b for b in at.button if "See an example" in b.label).click().run()
    pills = next(b for b in at.get("button_group") if b.label == "Phrases")
    pills.set_value(1).run()
    assert not at.exception
    assert any("Some issues" in m.value or "Needs work" in m.value or "Good" in m.value
               for m in at.markdown)


def test_demo_link_opens_on_the_result():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.query_params["demo"] = "1"
    at.run()
    assert not at.exception
    assert any("%</div>" in m.value for m in at.markdown)


def test_detailed_mode_uses_music_terms():
    at = app()
    next(b for b in at.button if "See an example" in b.label).click().run()
    mode = next(b for b in at.get("button_group") if b.label == "How should we explain it?")
    mode.set_value("Detailed").run()
    assert not at.exception
    cards = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert "#### 2. Your high notes are flat" in cards

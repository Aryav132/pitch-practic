"""Smoke test: the Streamlit page builds without uploads and the Analyse
button stays disabled until both files are given."""

from streamlit.testing.v1 import AppTest


def test_app_renders_and_waits_for_both_files():
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "🎤 Pitch Practice"
    assert at.button[0].label == "Analyse my singing"
    assert at.button[0].disabled

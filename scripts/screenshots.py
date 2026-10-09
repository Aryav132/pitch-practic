"""Regenerate the README screenshots from the built-in demo (synthetic audio,
so nothing copyrighted ends up in the repo).

Dev-only: pip install playwright (uses your installed Chrome).
Start the app first:  streamlit run app.py
Then:                 python scripts/screenshots.py [http://127.0.0.1:8501]
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8501") + "/?demo=1"
OUT = Path(__file__).resolve().parents[1] / "docs"


def to_top(locator) -> None:
    locator.first.evaluate("e => e.scrollIntoView({block: 'start'})")
    locator.page.wait_for_timeout(400)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        # Streamlit scrolls inside its own container, so "full page" only ever
        # captures one screen: scroll each part into view and capture that.
        for scheme in ("light", "dark"):
            page = browser.new_page(viewport={"width": 1280, "height": 1000},
                                    color_scheme=scheme, device_scale_factor=2)
            page.goto(URL)
            badge = page.get_by_text("in tune", exact=True)
            badge.wait_for(timeout=60_000)
            page.wait_for_timeout(1500)      # fonts + audio players settle
            if scheme == "light":
                page.get_by_text("Sing along. See exactly what to fix.").scroll_into_view_if_needed()
                page.screenshot(path=OUT / "home.png")
            to_top(page.get_by_text("This is the built-in demo", exact=False))
            page.wait_for_timeout(500)
            page.screenshot(path=OUT / f"results_{scheme}.png")
            if scheme == "light":
                page.get_by_role("tab", name="Phrases").click()
                pills = page.get_by_role("radiogroup", name="Phrases")
                pills.get_by_text("0:05.9").click()
                page.locator(".js-plotly-plot").first.wait_for()
                page.wait_for_timeout(2000)
                to_top(pills)
                page.screenshot(path=OUT / "phrase.png")
            page.close()
        browser.close()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()

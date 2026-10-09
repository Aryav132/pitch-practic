"""Command line: analyse a take against a reference, print the summary,
write an interactive HTML graph and (optionally) the JSON report.

    python -m pitch_practice.cli reference.m4a take.m4a [--start 30]
        [--mode snapped|free|absolute] [--threshold 40] [--html out.html] [--json out.json]
"""

import argparse
import json
import webbrowser
from dataclasses import replace
from pathlib import Path

from .config import AnalysisConfig
from .pipeline import analyze_files
from .plotting import make_figure
from .scoring import summary_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("take")
    ap.add_argument("--start", type=float, default=0.0, help="start time in the reference (s)")
    ap.add_argument("--mode", default="snapped", choices=["snapped", "free", "absolute"])
    ap.add_argument("--threshold", type=float, default=AnalysisConfig.pitch_threshold_cents,
                    help="off-pitch threshold in cents")
    ap.add_argument("--html", default="report.html")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    cfg = replace(AnalysisConfig(), key_mode=args.mode, pitch_threshold_cents=args.threshold)
    report = analyze_files(args.reference, args.take, args.start, cfg)
    print(summary_text(report))

    make_figure(report).write_html(args.html, include_plotlyjs="cdn")
    print(f"\nGraph: {Path(args.html).resolve()}")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"JSON:  {Path(args.json).resolve()}")
    if not args.no_open:
        webbrowser.open(Path(args.html).resolve().as_uri())


if __name__ == "__main__":
    main()

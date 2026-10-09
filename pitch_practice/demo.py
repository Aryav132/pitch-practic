"""Built-in demo: a synthetic singer and a synthetic take with deliberate
habits, so testers can see what a report looks like before recording.
No copyrighted audio involved."""

from pathlib import Path

import soundfile as sf

from .synthesis import voice_like

SR = 24000

# Reference: four short phrases (cents on the MIDI scale, None = rest).
REFERENCE = [
    (6000, 0.5), (6200, 0.5), (6400, 0.5), (6500, 1.0), (None, 0.5),
    (6700, 0.5), (6900, 0.4), (7100, 0.8), (6700, 0.7), (None, 0.5),
    (6400, 0.5), (6500, 0.4), (6700, 1.0), (6400, 0.6), (None, 0.5),
    (6900, 0.5), (7100, 0.5), (7200, 1.0), (6700, 0.6), (6000, 1.2),
]


def _take_notes():
    """The same melody with three habits: high notes (>= 6900) sung ~55 c
    flat, phrase entries ~250 ms late, and one note held a semitone low."""
    out, prev_rest = [], True
    for k, (c, d) in enumerate(REFERENCE):
        if c is None:
            out.append((None, d + 0.25)); prev_rest = True
            continue
        if prev_rest and k > 0:
            d -= 0.25                       # came in late; rest absorbed it
        prev_rest = False
        if k == 12:
            out.append((c - 100, d))        # held a whole semitone low
        else:
            out.append((c - 55 if c >= 6900 else c, d))
    return [(None, 1.5)] + out + [(None, 1.0)]   # phone started 1.5 s early


def write_demo(folder: str | Path) -> tuple[Path, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ref, take = folder / "demo_reference.wav", folder / "demo_take.wav"
    if not ref.exists():
        sf.write(ref, voice_like(REFERENCE, SR, seed=1), SR)
    if not take.exists():
        sf.write(take, voice_like(_take_notes(), SR, seed=2), SR)
    return ref, take

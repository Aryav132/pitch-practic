"""Streamlit front end. Thin by design: all measurement happens in
pitch_practice.pipeline (and the tips in pitch_practice.coaching); this
file only collects inputs and presents the Report.

    streamlit run app.py
"""

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import streamlit as st

from pitch_practice.alignment import AlignmentError
from pitch_practice.audio_io import AudioDecodeError, load_audio, probe_duration
from pitch_practice.coaching import coach, describe_cents
from pitch_practice.config import AnalysisConfig
from pitch_practice.demo import write_demo
from pitch_practice.pipeline import InputError, analyze_files
from pitch_practice.playback import clip, melody_tone, take_window
from pitch_practice.plotting import make_figure
from pitch_practice.scoring import summary_text
from pitch_practice.separation import DemucsSeparator, NoSeparator

AUDIO_TYPES = ["mp3", "m4a", "wav", "aac", "flac", "ogg"]
WORK_DIR = Path(tempfile.gettempdir()) / "pitch-practice-uploads"

st.set_page_config(page_title="Pitch Practice", page_icon="🎤", layout="wide")


# ---------------------------------------------------------------- helpers

@st.cache_resource
def get_separator() -> DemucsSeparator:
    # One model instance per server process; stems are cached on disk.
    return DemucsSeparator()


def save_upload(upload) -> Path:
    """Write an upload to disk under its content hash (ffmpeg needs a path)."""
    data = upload.getvalue()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = WORK_DIR / f"{hashlib.sha256(data).hexdigest()[:20]}{Path(upload.name).suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return path


def clock(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def run_analysis(ref_path, take_path, start_s, cfg, separator) -> None:
    """Analyse and keep everything the results page needs in the session."""
    report = analyze_files(ref_path, take_path, start_s, cfg, separator=separator)
    st.session_state.update(
        report=report, cfg=cfg,
        # Audio for "listen and compare": the singer's (separated) voice for
        # the analysed window, and the whole take. Both cached/cheap to load.
        ref_audio=separator.load_vocals(ref_path, start_s, cfg.max_ref_s, cfg.sr),
        take_audio=load_audio(take_path, cfg.sr, duration_s=cfg.max_take_s),
    )


def listen_row(t0: float, t1: float) -> None:
    """Three players for the same moment: singer, you, and the right notes."""
    r, sr = st.session_state.report, st.session_state.cfg.sr
    a, b = t0 - 0.3, t1 + 0.3
    singer = clip(st.session_state.ref_audio, sr, a - r.ref_start_s, b - r.ref_start_s)
    ta, tb = take_window(r, t0, t1, pad=0.3)
    you = clip(st.session_state.take_audio, sr, ta, tb)
    cols = st.columns(3)
    cols[0].caption("▶ The singer")
    cols[0].audio(singer, sample_rate=sr)
    cols[1].caption("▶ You")
    cols[1].audio(you, sample_rate=sr)
    cols[2].caption("▶ The right notes (clean tone)")
    cols[2].audio(melody_tone(r, a, b, sr), sample_rate=sr)


def phrase_status(s) -> tuple[str, str]:
    if s.ref_voiced_s < 0.3:
        return "·", "too short to judge"
    if s.badness < 0.10:
        return "✓", "Good"
    if s.badness < 0.30:
        return "⚠", "Some issues"
    return "✗", "Needs work"


def phrase_sentence(s, ref_start: float) -> str:
    icon, word = phrase_status(s)
    parts = [f"**{icon} {word}** · {clock(ref_start + s.start_s)}–{clock(ref_start + s.end_s)}."]
    if s.scored_s and s.off_direction:
        direction = {"flat": "too low", "sharp": "too high"}.get(s.off_direction, "both too high and too low")
        parts.append(f"{s.off_pitch_frac:.0%} of it was off, {direction}, "
                     f"{describe_cents(s.off_dev_cents)}.")
    elif s.scored_s:
        parts.append("In tune throughout.")
    if s.missed_frac and s.missed_frac >= 0.25:
        parts.append(f"You didn't sing {s.missed_frac:.0%} of it.")
    if s.drift_ms == s.drift_ms and abs(s.drift_ms) >= 80:
        parts.append(f"You were about {abs(s.drift_ms):.0f} ms "
                     f"{'later' if s.drift_ms > 0 else 'earlier'} than your usual timing.")
    return " ".join(parts)


# ---------------------------------------------------------------- intro

st.title("🎤 Pitch Practice")
st.write("Sing along to a song, upload your recording, and see **where** your pitch and timing "
         "differed from the singer, and **what to practise**.")

report = st.session_state.get("report")
with st.expander("How it works & how to record", expanded=report is None):
    st.markdown(
        "1. **Pick a song** you want to practise and a section of it (up to 60 s).\n"
        "2. **Record yourself singing along** on your phone:\n"
        "   - 🎧 **Wear headphones**, so your phone hears only you, not the song\n"
        "   - Quiet room, phone about 30 cm from your mouth\n"
        "   - Press record, then start the song at your chosen point, then sing\n"
        "   - A different key or octave is fine: we adjust for it\n"
        "3. **Upload both** below and press *Analyse*.\n\n"
        "Under the hood: we separate the singer's voice from the music with **Demucs** (a "
        "pretrained neural network), then use classical signal processing (pYIN pitch "
        "tracking, dynamic time warping) to compare your singing with theirs. The tips come "
        "from those measurements, not from an AI.")
    if st.button("See an example first (built-in demo, no upload needed)"):
        ref, take = write_demo(WORK_DIR / "demo")
        with st.spinner("Analysing the demo..."):
            run_analysis(ref, take, 0.0, AnalysisConfig(), NoSeparator())
        st.session_state.demo = True
        st.rerun()

# ---------------------------------------------------------------- inputs

col_ref, col_take = st.columns(2)
with col_ref:
    st.subheader("1 · The song")
    ref_upload = st.file_uploader("Song file (or a solo vocal)", type=AUDIO_TYPES)
    clean_ref = st.checkbox("This is already a solo vocal (skip separating it from the music)")
    start_s = 0.0
    if ref_upload:
        ref_path = save_upload(ref_upload)
        try:
            total = probe_duration(ref_path)
        except AudioDecodeError as e:
            st.error(f"Couldn't read that file: {e}")
            st.stop()
        start_s = float(st.slider("Where in the song did you start singing? (seconds)",
                                  0.0, max(0.0, total - 5.0), 0.0, step=0.5))
        st.caption(f"We'll analyse {clock(start_s)} to {clock(min(total, start_s + 60))}. "
                   "Preview from that point:")
        st.audio(ref_upload.getvalue(), start_time=int(start_s))

with col_take:
    st.subheader("2 · Your recording")
    take_upload = st.file_uploader("Your take (up to 90 s)", type=AUDIO_TYPES)
    if take_upload:
        st.audio(take_upload.getvalue())

with st.expander("Settings"):
    c1, c2, c3 = st.columns(3)
    threshold = c1.slider("How strict is 'off-pitch'? (cents)", 15, 100, 40, step=5,
                          help="100 cents = one semitone = one piano key. 40 is about "
                               "half a piano key.")
    key_mode = c2.selectbox(
        "Key handling", ["snapped", "free", "absolute"],
        format_func={"snapped": "Allow a different key (recommended)",
                     "free": "Forgive any constant offset",
                     "absolute": "Must match the singer's key exactly"}.get)
    tolerance = c3.slider("Timing slack for pitch (ms)", 0, 60, 30, step=10,
                          help="Compare your pitch with the singer's within this many ms, so a "
                               "slide sung slightly late isn't also counted as wrong pitch.")

ready = ref_upload is not None and take_upload is not None
if st.button("Analyse my singing", type="primary", disabled=not ready):
    cfg = replace(AnalysisConfig(), pitch_threshold_cents=threshold, key_mode=key_mode,
                  pitch_time_tolerance_ms=tolerance)
    separator = NoSeparator() if clean_ref else get_separator()
    with st.status("Analysing...", expanded=True) as status:
        if not clean_ref and not separator.cache_path(ref_path, start_s, cfg.max_ref_s).exists():
            st.write("Separating the singer's voice from the music (first time for this "
                     "section: about 30 s on a Mac, longer on a server)...")
        st.write("Finding your notes, lining your take up with the song, comparing...")
        try:
            run_analysis(ref_path, save_upload(take_upload), start_s, cfg, separator)
        except (InputError, AlignmentError, AudioDecodeError) as e:
            status.update(label="Couldn't analyse this take", state="error")
            st.error(str(e))
            st.stop()
        status.update(label="Done", state="complete", expanded=False)
    st.session_state.demo = False
    st.rerun()

# ---------------------------------------------------------------- results

report = st.session_state.get("report")
if report:
    cfg = st.session_state.cfg
    dark = st.context.theme.type == "dark"
    st.divider()
    if st.session_state.get("demo"):
        st.info("This is the **built-in demo**: a synthetic singer, and a take with three "
                "deliberate habits. Upload your own files above to analyse yourself.")
    st.header(f"{report.accuracy_pct:.0f}% of your singing was in tune")
    timing_words = ("very steady" if report.mean_abs_drift_ms < 60 else
                    "mostly steady" if report.mean_abs_drift_ms < 120 else "uneven")
    st.caption(f"{report.key_description} Your timing was {timing_words} "
               f"(note starts within about {report.mean_abs_drift_ms:.0f} ms of your usual).")

    tab_fix, tab_phrases, tab_numbers = st.tabs(
        ["What to work on", "Phrase by phrase", "All the numbers"])

    with tab_fix:
        tips = coach(report, cfg)
        if not tips:
            st.success("No clear problems found in this take. Try a harder section, or make "
                       "the off-pitch setting stricter.")
        for k, tip in enumerate(tips, 1):
            with st.container(border=True):
                st.markdown(f"#### {k}. {tip.title}")
                st.write(tip.detail)
                st.markdown(f"**Try this:** {tip.try_this}")
                if tip.where:
                    t0, t1 = tip.where[0]
                    st.caption(f"Listen at {clock(t0)}–{clock(t1)}:")
                    listen_row(t0, t1)
        st.caption("Tips come from your measurements. The practice suggestions are general "
                   "singing advice.")

    with tab_phrases:
        st.write("Each button is one phrase of the song. Pick one to zoom in and listen.")
        sections = [s for s in report.sections if s.ref_voiced_s >= 0.3]
        pick = st.pills(
            "Phrases", list(range(len(sections))), label_visibility="collapsed",
            format_func=lambda i: f"{phrase_status(sections[i])[0]} "
                                  f"{clock(report.ref_start_s + sections[i].start_s)}")
        st.caption("✓ good · ⚠ some issues · ✗ needs work. On the graph: grey band = the "
                   "singer, blue line = you, red dots = more than "
                   f"{report.threshold_cents:.0f} cents off "
                   f"(about {report.threshold_cents / 100:.1f} of a piano key).")
        x_range = None
        if pick is not None:
            s = sections[pick]
            st.markdown(phrase_sentence(s, report.ref_start_s))
            x_range = (report.ref_start_s + s.start_s - 0.5, report.ref_start_s + s.end_s + 0.5)
            listen_row(report.ref_start_s + s.start_s, report.ref_start_s + s.end_s)
        st.plotly_chart(make_figure(report, dark=dark, x_range=x_range), theme=None)

    with tab_numbers:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pitch accuracy", f"{report.accuracy_pct:.0f}%",
                  help=f"Share of your singing within {report.threshold_cents:.0f} cents.")
        m2.metric("Sharp / flat tendency", f"{report.mean_dev_cents:+.0f} cents")
        m3.metric("Timing (note starts)", f"±{report.mean_abs_drift_ms:.0f} ms",
                  help="Compared with your own average timing.")
        m4.metric("Sang along", f"{report.sung_pct:.0f}%",
                  help="Share of the singer's singing during which you were also singing.")
        st.text(summary_text(report))
        if report.worst:
            st.dataframe([{
                "rank": k,
                "song time": f"{clock(report.ref_start_s + s.start_s)} – "
                             f"{clock(report.ref_start_s + s.end_s)}",
                "off-pitch": f"{s.off_pitch_frac:.0%}" if s.scored_s else "not sung",
                "by (when off)": f"{s.off_dev_cents:.0f} c ({s.off_direction})"
                                 if s.off_direction else "",
                "timing": f"{s.drift_ms:+.0f} ms" if s.drift_ms == s.drift_ms else "",
            } for k, s in enumerate(report.worst, 1)], hide_index=True)
        st.download_button("Download measurements (JSON)",
                           json.dumps(report.to_dict(), indent=2), "pitch_report.json",
                           "application/json")

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
from pitch_practice.coaching import (
    coach, describe_cents, describe_key_plain, note_verdicts, plain_seconds, plain_size)
from pitch_practice.config import AnalysisConfig
from pitch_practice.demo import write_demo
from pitch_practice.pipeline import InputError, analyze_files
from pitch_practice.playback import clip, hear_the_difference, melody_tone, take_window
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


def listen(t0: float, t1: float, key: str, expert: bool) -> None:
    """Simple: one "hear the difference" player (singer, you, singer), with a
    slower option. Detailed: the three separate players. Both offer the other
    one folded away."""
    r, sr = st.session_state.report, st.session_state.cfg.sr
    a, b = t0 - 0.3, t1 + 0.3
    singer = clip(st.session_state.ref_audio, sr, a - r.ref_start_s, b - r.ref_start_s)
    ta, tb = take_window(r, t0, t1, pad=0.3)
    you = clip(st.session_state.take_audio, sr, ta, tb)

    def difference():
        slow = st.toggle("Slower", key=f"slow_{key}",
                         help="75% speed. The notes stay at exactly the same pitch.")
        st.caption(f"🔊 Hear the difference ({clock(t0)}–{clock(t1)}): the singer, then you, "
                   "then the singer again.")
        st.audio(hear_the_difference(singer, you, sr, slow=slow), sample_rate=sr)

    def separate():
        cols = st.columns(3)
        for col, label, y in [(cols[0], "▶ The singer", singer), (cols[1], "▶ You", you),
                              (cols[2], "▶ The right notes (clean tone)", melody_tone(r, a, b, sr))]:
            col.caption(label)
            col.audio(y, sample_rate=sr)

    if expert:
        separate()
        with st.expander("Hear the difference (back to back)"):
            difference()
    else:
        difference()
        with st.expander("Listen to each one separately"):
            separate()


TUNER = {  # verdict -> (icon, simple words, detailed words)
    "ok": ("✅", "right", "in tune"),
    "low": ("⬆️", "too low: sing higher", "flat"),
    "high": ("⬇️", "too high: sing lower", "sharp"),
    "missed": ("⏸", "not sung", "not sung"),
    "unclear": ("·", "unclear", "unclear"),
}


def tuner_strip(verdicts, expert: bool) -> None:
    """One small card per note, like a tuner: right / sing higher / sing lower."""
    if not verdicts:
        st.caption("No clear notes in this part.")
        return
    with st.container(horizontal=True, gap="small"):
        for v in verdicts:
            icon, simple, detailed = TUNER[v.verdict]
            words = detailed + (f" {v.dev_cents:+.0f} c" if expert and v.verdict in ("low", "high")
                                else "") if expert else simple
            with st.container(border=True, width=150 if not expert else 120):
                st.markdown(f"<div style='font-size:1.4rem;line-height:1.2'>{icon}</div>"
                            f"<div style='font-size:.85rem'>{words}</div>"
                            f"<div style='font-size:.75rem;opacity:.6'>{clock(v.start_s)}</div>",
                            unsafe_allow_html=True)


def phrase_status(s) -> tuple[str, str]:
    if s.ref_voiced_s < 0.3:
        return "·", "too short to judge"
    if s.badness < 0.10:
        return "✓", "Good"
    if s.badness < 0.30:
        return "⚠", "Some issues"
    return "✗", "Needs work"


def phrase_sentence(s, ref_start: float, expert: bool) -> str:
    icon, word = phrase_status(s)
    parts = [f"**{icon} {word}** · {clock(ref_start + s.start_s)}–{clock(ref_start + s.end_s)}."]
    if s.scored_s and s.off_direction:
        if expert:
            direction = {"flat": "flat", "sharp": "sharp"}.get(s.off_direction, "both flat and sharp")
            parts.append(f"{s.off_pitch_frac:.0%} off-pitch, {direction}, "
                         f"{describe_cents(s.off_dev_cents)}.")
        else:
            direction = {"flat": "too low", "sharp": "too high"}.get(
                s.off_direction, "sometimes too high, sometimes too low")
            parts.append(f"{s.off_pitch_frac:.0%} of it was off, {direction}, "
                         f"{plain_size(s.off_dev_cents)}.")
    elif s.scored_s:
        parts.append("Right on throughout." if not expert else "In tune throughout.")
    if s.missed_frac and s.missed_frac >= 0.25:
        parts.append(f"You didn't sing {s.missed_frac:.0%} of it.")
    if s.drift_ms == s.drift_ms and abs(s.drift_ms) >= 80:
        when = f"about {abs(s.drift_ms):.0f} ms" if expert else plain_seconds(s.drift_ms)
        parts.append(f"You were {when} {'later' if s.drift_ms > 0 else 'earlier'} "
                     "than your usual timing.")
    return " ".join(parts)


# ---------------------------------------------------------------- intro

st.caption("🎤 PITCH PRACTICE")
st.title("Sing along. See exactly what to fix.")
st.write("Upload a song and a recording of yourself singing it. We line the two up and show "
         "**where** your pitch and timing drifted from the singer, and **what to practise**.")

for col, (icon, head, body) in zip(st.columns(3), [
        ("🎧", "Record along", "Headphones on, sing with the song while your phone records you."),
        ("📤", "Upload both", "The song (any MP3) and your recording, up to 60 s of singing."),
        ("🎯", "See what to fix", "Plain-language tips, and the singer vs you, side by side.")]):
    with col.container(border=True):
        st.markdown(f"### {icon}\n**{head}**  \n{body}")

demo_col, _ = st.columns([1, 2])
# "?demo=1" in the address opens straight onto the demo result: a link you
# can send people, and how the README screenshots are taken.
auto_demo = st.query_params.get("demo") == "1" and "report" not in st.session_state
if demo_col.button("▶  See an example first", help="A built-in demo; no upload needed.") or auto_demo:
    ref, take = write_demo(WORK_DIR / "demo")
    with st.spinner("Analysing the demo..."):
        run_analysis(ref, take, 0.0, AnalysisConfig(), NoSeparator())
    st.session_state.demo = True
    st.rerun()

report = st.session_state.get("report")
with st.expander("Recording checklist"):
    st.markdown(
        "- 🎧 **Wear headphones**, so your phone hears only you, not the song\n"
        "- Quiet room, phone about 30 cm from your mouth\n"
        "- Press record first, then start the song at your chosen point, then sing\n"
        "- A different key or octave is fine: we adjust for it\n\n"
        "*Under the hood:* the singer's voice is separated from the music with **Demucs** "
        "(a pretrained neural network); pitch tracking (pYIN), alignment (dynamic time "
        "warping) and scoring are classical signal processing. Tips come from those "
        "measurements, not from an AI.")

# ---------------------------------------------------------------- inputs

st.write("")
col_ref, col_take = st.columns(2)
ref_box, take_box = col_ref.container(border=True), col_take.container(border=True)
with ref_box:
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

with take_box:
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

    mode = st.segmented_control(
        "How should we explain it?", ["Simple", "Detailed"], default="Simple", key="mode",
        help="Simple: everyday words, no music knowledge needed. "
             "Detailed: music terms (flat, sharp, semitones) and exact numbers.")
    expert = mode == "Detailed"

    acc = report.accuracy_pct
    verdict = ("Great take." if acc >= 90 else
               "Solid take, with a few spots to fix." if acc >= 75 else
               "Plenty to work on. Start with the first tip below.")
    timing_words = ("very steady" if report.mean_abs_drift_ms < 60 else
                    "mostly steady" if report.mean_abs_drift_ms < 120 else "uneven")
    with st.container(border=True):
        badge, words = st.columns([1, 3], vertical_alignment="center")
        badge.markdown(
            f"<div style='font-size:3.4rem;font-weight:600;line-height:1'>{acc:.0f}%</div>"
            f"<div style='opacity:.7'>{'in tune' if expert else 'of your notes were right'}</div>",
            unsafe_allow_html=True)
        words.subheader(verdict)
        if expert:
            words.write(f"{report.key_description} Note-start timing within about "
                        f"{report.mean_abs_drift_ms:.0f} ms of your usual ({timing_words}).")
        else:
            words.write(f"{describe_key_plain(report.raw_offset_cents, report.key_mode)} "
                        f"Your timing was **{timing_words}**.")

    tab_fix, tab_lines, tab_numbers = st.tabs(
        ["What to fix", "Phrases" if expert else "Line by line", "Numbers"])

    with tab_fix:
        tips = coach(report, cfg, expert=expert)
        if not tips:
            st.success("No clear problems found in this take. Try a harder part of the song, "
                       "or make the off-pitch setting stricter.")
        for k, tip in enumerate(tips, 1):
            with st.container(border=True):
                st.markdown(f"#### {k}. {tip.title}")
                st.write(tip.detail)
                st.markdown(f"**{'Try this' if expert else 'Practise like this'}:** {tip.try_this}")
                if tip.where:
                    listen(*tip.where[0], key=f"tip{k}", expert=expert)
        st.caption("Tips come from your measurements. The practice suggestions are general "
                   "singing advice.")

    with tab_lines:
        sections = [s for s in report.sections if s.ref_voiced_s >= 0.3]
        unit = "phrase" if expert else "line"
        st.write(f"Each button is one {unit} of the song. Pick one to see every note and listen.")
        worst_idx = next((i for i, s in enumerate(sections)
                          if report.worst and s.index == report.worst[0].index), None)
        pick = st.pills(
            "Phrases", list(range(len(sections))), label_visibility="collapsed",
            default=worst_idx, key="pick",
            format_func=lambda i: f"{phrase_status(sections[i])[0]} "
                                  f"{clock(report.ref_start_s + sections[i].start_s)}")
        st.caption("✓ good · ⚠ some issues · ✗ needs work")
        x_range = None
        if pick is not None:
            s = sections[pick]
            t0, t1 = report.ref_start_s + s.start_s, report.ref_start_s + s.end_s
            st.markdown(phrase_sentence(s, report.ref_start_s, expert))
            st.caption("Every note in this " + unit + ":")
            tuner_strip([v for v in note_verdicts(report, cfg) if t0 - 0.05 <= v.start_s < t1],
                        expert)
            listen(t0, t1, key=f"line{pick}", expert=expert)
            x_range = (t0 - 0.5, t1 + 0.5)
        st.caption("On the graph: grey band = the singer, blue line = you, red dots = "
                   + (f"more than {report.threshold_cents:.0f} cents off." if expert else
                      "too high or too low. Higher on the graph = higher note."))
        st.plotly_chart(make_figure(report, dark=dark, x_range=x_range, show_title=False,
                                    plain=not expert),
                        theme=None, config={"displayModeBar": False})

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

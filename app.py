"""Streamlit front end. Thin by design: all measurement happens in
pitch_practice.pipeline; this file only collects inputs and shows the Report.

    streamlit run app.py
"""

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import streamlit as st

from pitch_practice.alignment import AlignmentError
from pitch_practice.audio_io import AudioDecodeError, probe_duration
from pitch_practice.config import AnalysisConfig
from pitch_practice.pipeline import InputError, analyze_files
from pitch_practice.plotting import make_figure
from pitch_practice.scoring import summary_text
from pitch_practice.separation import DemucsSeparator, NoSeparator

AUDIO_TYPES = ["mp3", "m4a", "wav", "aac", "flac", "ogg"]
UPLOAD_DIR = Path(tempfile.gettempdir()) / "pitch-practice-uploads"

st.set_page_config(page_title="Pitch Practice", page_icon="🎤", layout="wide")


@st.cache_resource
def get_separator() -> DemucsSeparator:
    # One model instance per server process; stems are cached on disk.
    return DemucsSeparator()


def save_upload(upload) -> Path:
    """Write an upload to disk under its content hash (ffmpeg needs a path)."""
    data = upload.getvalue()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{hashlib.sha256(data).hexdigest()[:20]}{Path(upload.name).suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return path


def clock(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


st.title("🎤 Pitch Practice")
st.caption("Sing along to a song, upload your recording, and see exactly where your pitch "
           "and timing differed from the singer. Pitch tracking, alignment and scoring are "
           "classical signal processing; separating the singer's voice from the music uses "
           "Demucs, a pretrained neural network.")

col_ref, col_take = st.columns(2)

with col_ref:
    st.subheader("1 · Reference")
    ref_upload = st.file_uploader("The song (or a solo vocal)", type=AUDIO_TYPES)
    clean_ref = st.checkbox("This is already a solo vocal (skip separating it from the music)")
    start_s = 0.0
    if ref_upload:
        ref_path = save_upload(ref_upload)
        try:
            total = probe_duration(ref_path)
        except AudioDecodeError as e:
            st.error(f"Couldn't read that file: {e}")
            st.stop()
        start_s = float(st.slider("Start of the section you sang (seconds into the song)",
                                  0.0, max(0.0, total - 5.0), 0.0, step=0.5))
        st.caption(f"Analysing {clock(start_s)} to {clock(min(total, start_s + 60))} "
                   f"(up to 60 s). Preview from the start point:")
        st.audio(ref_upload.getvalue(), start_time=int(start_s))

with col_take:
    st.subheader("2 · Your take")
    take_upload = st.file_uploader("Your recording (max 90 s)", type=AUDIO_TYPES)
    st.markdown(
        "**How to record:** wear **headphones** (so your phone only hears you), press "
        "record, then play the song from the start point above and sing along. "
        "Same key or a different octave are both fine.")
    if take_upload:
        st.audio(take_upload.getvalue())

with st.expander("Settings"):
    c1, c2, c3 = st.columns(3)
    threshold = c1.slider("Off-pitch threshold (cents)", 15, 100, 40, step=5,
                          help="100 cents = one semitone (one piano key).")
    key_mode = c2.selectbox(
        "Key handling", ["snapped", "free", "absolute"],
        help="snapped: a different key/octave is fine, but being slightly flat throughout "
             "still counts. free: forgive any constant offset. absolute: must match exactly.")
    tolerance = c3.slider("Timing slack for pitch (ms)", 0, 60, 30, step=10,
                          help="Compare your pitch with the reference within this many ms, so a "
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
        st.write("Tracking pitch, finding where the song starts in your take, aligning...")
        try:
            report = analyze_files(ref_path, save_upload(take_upload), start_s, cfg,
                                   separator=separator)
        except (InputError, AlignmentError, AudioDecodeError) as e:
            status.update(label="Couldn't analyse this take", state="error")
            st.error(str(e))
            st.stop()
        status.update(label="Done", state="complete", expanded=False)
    st.session_state["report"] = report

report = st.session_state.get("report")
if report:
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pitch accuracy", f"{report.accuracy_pct:.0f}%",
              help=f"Share of your sung notes within {report.threshold_cents:.0f} cents.")
    m2.metric("Sharp / flat tendency", f"{report.mean_dev_cents:+.0f} cents")
    m3.metric("Timing (note starts)", f"±{report.mean_abs_drift_ms:.0f} ms",
              help="Compared with your own average timing.")
    m4.metric("Sang along", f"{report.sung_pct:.0f}%",
              help="Share of the singer's singing during which you were also singing.")

    st.text(summary_text(report))
    st.plotly_chart(make_figure(report), theme=None)

    if report.worst:
        st.subheader("Worst sections")
        st.dataframe([{
            "rank": k,
            "song time": f"{clock(report.ref_start_s + s.start_s)} – {clock(report.ref_start_s + s.end_s)}",
            "off-pitch": f"{s.off_pitch_frac:.0%}" if s.scored_s else "not sung",
            "by (when off)": f"{s.off_dev_cents:.0f} c ({s.off_direction})" if s.off_direction else "",
            "timing": f"{s.drift_ms:+.0f} ms" if s.drift_ms == s.drift_ms else "",
        } for k, s in enumerate(report.worst, 1)], hide_index=True)

    st.download_button("Download measurements (JSON)",
                       json.dumps(report.to_dict(), indent=2), "pitch_report.json",
                       "application/json")

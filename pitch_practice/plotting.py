"""Interactive Plotly figure built only from a Report (no recomputation).

Top panel: reference vs your pitch on a note-name axis, off-pitch frames
marked, worst sections shaded. Bottom panel: timing drift vs your own
average. Two panels, not two y-axes on one plot: pitch (notes) and timing
(ms) are different units.

Colour roles (validated with the dataviz palette checker): reference is a
wide pale neutral band, you are a thin blue line, off-pitch is large red
markers. Width/shape separate them too, so colour is never the only cue
(red vs gray is close for red-blind viewers).
"""

import librosa
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .scoring import Report

# Light and dark tokens from the reference palette (dataviz skill). Dark is
# its own set of steps for the dark surface, not an inverted light theme.
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e1e0d9",
                  axis="#c3c2b7", ref="#c3c2b7", you="#2a78d6",
                  timing_band="rgba(195,194,183,0.18)"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", grid="#2c2c2a",
                 axis="#383835", ref="#5f5e59", you="#3987e5",
                 timing_band="rgba(95,94,89,0.25)"),
}
OFF_COLOR = "#d03b3b"           # status: critical (same in both modes)
WORST_FILL_STRONG = "#fab219"   # status: warning (always with a text label)


def _note_axis(cents: np.ndarray):
    lo, hi = np.nanmin(cents) / 100, np.nanmax(cents) / 100
    midis = np.arange(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 2)
    step = 1 if len(midis) <= 16 else 2   # avoid crowded labels on wide ranges
    midis = midis[::step]
    return midis * 100, [librosa.midi_to_note(m) for m in midis]


def make_figure(r: Report, dark: bool = False, x_range: tuple[float, float] | None = None
                ) -> go.Figure:
    th = THEMES["dark" if dark else "light"]
    SURFACE, TEXT, TEXT_2, GRID, AXIS = (th[k] for k in ("surface", "text", "text2", "grid", "axis"))
    REF_COLOR, YOU_COLOR = th["ref"], th["you"]
    f = r.frames
    t = r.ref_start_s + f.times_s   # song time

    # No title on the pitch panel: the legend names its lines, and the space
    # above it holds the worst-section brackets (which used to sit on the melody).
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        row_heights=[0.72, 0.28],
                        subplot_titles=("", "Note-start timing vs your own average"))

    def note(c):
        return np.array([librosa.midi_to_note(x / 100, cents=True) if np.isfinite(x) else "-"
                         for x in c])

    ref_notes, you_notes = note(f.ref_cents), note(f.take_cents)
    dev_txt = np.where(np.isfinite(f.deviation), np.char.add(
        np.char.mod("%+.0f", np.nan_to_num(f.deviation)), " c"), "not scored")
    hover = np.stack([ref_notes, you_notes, dev_txt], axis=-1)

    fig.add_trace(go.Scatter(
        x=t, y=f.ref_cents, name="Reference", mode="lines",
        line=dict(color=REF_COLOR, width=7), customdata=hover,
        hovertemplate="%{x:.2f}s  ref %{customdata[0]}<br>you %{customdata[1]}"
                      "  (%{customdata[2]})<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=f.take_cents, name="You (key offset removed)", mode="lines",
        line=dict(color=YOU_COLOR, width=2), hoverinfo="skip",
    ), row=1, col=1)
    off = f.off_pitch
    fig.add_trace(go.Scatter(
        x=t[off], y=f.take_cents[off], name=f"Off-pitch (> {r.threshold_cents:.0f} c"
        + (f", ±{r.pitch_time_tolerance_ms:.0f} ms)" if r.pitch_time_tolerance_ms else ")"),
        mode="markers", marker=dict(color=OFF_COLOR, size=8, symbol="circle",
                                    line=dict(color=SURFACE, width=1)),
        customdata=hover[off],
        hovertemplate="%{x:.2f}s  off by %{customdata[2]}<extra></extra>",
    ), row=1, col=1)

    # Worst sections: a bracket + rank label just ABOVE the pitch panel, so
    # they never cover the melody.
    for k, s in enumerate(r.worst, 1):
        x0, x1 = r.ref_start_s + s.start_s, r.ref_start_s + s.end_s
        inset = min(0.05, 0.05 * (x1 - x0))  # visible gap when sections touch
        fig.add_shape(type="rect", x0=x0 + inset, x1=x1 - inset, y0=1.01, y1=1.035,
                      yref="y domain", xref="x", fillcolor=WORST_FILL_STRONG, line_width=0,
                      row=1, col=1)
        fig.add_annotation(x=(x0 + x1) / 2, y=1.04, yref="y domain", text=f"worst #{k}",
                           showarrow=False, yanchor="bottom",
                           font=dict(size=11, color=TEXT_2), row=1, col=1)

    tickvals, ticktext = _note_axis(np.concatenate([f.ref_cents, f.take_cents]))
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, row=1, col=1)

    # Timing panel: drift line, a 0 baseline and the "early/late" threshold band.
    thr = r.timing_threshold_ms
    fig.add_hrect(y0=-thr, y1=thr, fillcolor=th["timing_band"], line_width=0,
                  layer="below", row=2, col=1)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=2, col=1)
    timed = np.isfinite(f.drift_ms)
    fig.add_trace(go.Bar(
        x=t[timed], y=f.drift_ms[timed], name="Note-start timing", showlegend=False,
        width=0.06, marker=dict(color=YOU_COLOR, line_width=0),
        customdata=ref_notes[timed],
        hovertemplate="%{x:.2f}s  %{customdata}: %{y:+.0f} ms "
                      "(+ = later than your usual)<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="ms (+ late)", zeroline=False, row=2, col=1)
    fig.update_xaxes(title_text="song time (s)", row=2, col=1)

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif", color=TEXT, size=13),
        height=640, margin=dict(l=60, r=24, t=110, b=50), hovermode="closest",
        title=dict(text=f"Pitch accuracy {r.accuracy_pct:.0f}%  ·  "
                        f"timing ±{r.mean_abs_drift_ms:.0f} ms",
                   font=dict(size=14, color=TEXT_2), x=0, xanchor="left"),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS)
    for a in fig.layout.annotations:
        if a.text == "Note-start timing vs your own average":
            a.update(font=dict(size=12, color=TEXT_2), x=0, xanchor="left")
    if x_range:
        fig.update_xaxes(range=list(x_range))
    return fig

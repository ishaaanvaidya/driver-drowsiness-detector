"""Streamlit dashboard for drowsiness detection session logs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
SESSION_DIR = ROOT / "data" / "sessions"

ALERT_ORDER = ["OK", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
ALERT_COLORS = {
    "OK": "#22c55e",
    "LOW": "#84cc16",
    "MEDIUM": "#f59e0b",
    "HIGH": "#ef4444",
    "CRITICAL": "#7f1d1d",
}


@dataclass
class SessionSummary:
    path: Path
    name: str
    rows: int
    duration_s: float
    estimated_fps: float
    max_score: float
    mean_score: float
    peak_perclos: float
    peak_microsleep: float
    alert_seconds: float
    reliability_pct: float


def apply_theme() -> None:
    st.set_page_config(
        page_title="Drowsiness Session Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f7f4;
            --panel: #ffffff;
            --ink: #17201b;
            --muted: #647067;
            --line: #dfe5de;
            --green: #16a34a;
            --amber: #f59e0b;
            --red: #dc2626;
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(22, 163, 74, 0.10), transparent 28rem),
                linear-gradient(180deg, #f8faf6 0%, #eef2ec 100%);
            color: var(--ink);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 10px 28px rgba(22, 32, 27, 0.07);
        }
        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-weight: 650;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        .hero {
            background: linear-gradient(135deg, #163326 0%, #21543c 55%, #657c3a 100%);
            border-radius: 18px;
            padding: 26px 30px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 18px 46px rgba(16, 24, 20, 0.20);
        }
        .hero h1 {
            font-size: 2.1rem;
            margin: 0 0 8px 0;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            color: rgba(255,255,255,0.78);
            font-size: 1rem;
        }
        .insight {
            background: rgba(255, 255, 255, 0.9);
            border-left: 5px solid #16a34a;
            border-radius: 12px;
            padding: 14px 16px;
            margin: 8px 0;
            box-shadow: 0 8px 22px rgba(23, 32, 27, 0.06);
        }
        .small-muted {
            color: #647067;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def list_session_files() -> list[Path]:
    if not SESSION_DIR.exists():
        return []
    return sorted(SESSION_DIR.glob("session_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


@st.cache_data(show_spinner=False)
def load_session(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    df = pd.read_csv(path)
    if df.empty:
        return df

    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    for column in [
        "frame",
        "ear",
        "mar",
        "perclos",
        "blink_rate",
        "score",
        "microsleep_duration",
        "pose_pitch",
        "pose_yaw",
        "pose_roll",
        "pose_score",
        "eyes_reliable",
    ]:
        df[column] = numeric_column(df, column, np.nan if column == "blink_rate" else 0.0)

    if "alert_level" not in df:
        df["alert_level"] = "OK"
    df["alert_level"] = df["alert_level"].fillna("OK").astype(str).str.upper()
    df.loc[~df["alert_level"].isin(ALERT_ORDER), "alert_level"] = "OK"

    df = df.sort_values(["timestamp", "frame"], na_position="last").reset_index(drop=True)

    if df["timestamp"].notna().sum() >= 2:
        start = df["timestamp"].dropna().iloc[0]
        df["elapsed_s"] = (df["timestamp"] - start).dt.total_seconds().ffill().fillna(0.0)
    else:
        df["elapsed_s"] = df.index / 30.0

    df["elapsed_s"] = df["elapsed_s"].clip(lower=0)
    fps = estimate_fps(df)
    window_5s = max(3, int(round(fps * 5)))
    window_15s = max(3, int(round(fps * 15)))

    df["score_smooth"] = df["score"].rolling(window_5s, min_periods=1).mean()
    df["ear_smooth"] = df["ear"].rolling(window_5s, min_periods=1).mean()
    df["perclos_pct"] = df["perclos"] * 100.0
    df["pose_magnitude"] = np.sqrt(df["pose_pitch"] ** 2 + df["pose_yaw"] ** 2 + df["pose_roll"] ** 2)
    df["pose_smooth"] = df["pose_magnitude"].rolling(window_5s, min_periods=1).mean()
    df["attention_load"] = (
        0.55 * df["score"].clip(0, 100)
        + 0.25 * df["perclos_pct"].clip(0, 100)
        + 0.20 * (df["pose_score"].clip(0, 30) / 30.0 * 100)
    ).rolling(window_15s, min_periods=1).mean()
    df["risk_band"] = pd.cut(
        df["score"],
        bins=[-0.1, 25, 45, 70, 90, 101],
        labels=["OK", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
    ).astype(str)
    df["minute"] = (df["elapsed_s"] // 60).astype(int)
    df["closed_hint"] = (df["perclos"] > 0.20) | (df["microsleep_duration"] > 0.5)
    return df


def estimate_fps(df: pd.DataFrame) -> float:
    if df.empty:
        return 30.0
    duration = float(df.get("elapsed_s", pd.Series([0])).max())
    if duration > 0 and len(df) > 2:
        return max(1.0, min(60.0, (len(df) - 1) / duration))
    return 30.0


def summarize(path: Path, df: pd.DataFrame) -> SessionSummary:
    duration = float(df["elapsed_s"].max()) if not df.empty and "elapsed_s" in df else 0.0
    fps = estimate_fps(df)
    alert_seconds = float((df["alert_level"] != "OK").sum() / fps) if not df.empty else 0.0
    reliability = float(df["eyes_reliable"].mean() * 100) if not df.empty and "eyes_reliable" in df else 100.0
    return SessionSummary(
        path=path,
        name=path.name,
        rows=len(df),
        duration_s=duration,
        estimated_fps=fps,
        max_score=float(df["score"].max()) if not df.empty else 0.0,
        mean_score=float(df["score"].mean()) if not df.empty else 0.0,
        peak_perclos=float(df["perclos_pct"].max()) if not df.empty else 0.0,
        peak_microsleep=float(df["microsleep_duration"].max()) if not df.empty else 0.0,
        alert_seconds=alert_seconds,
        reliability_pct=reliability,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def alert_episodes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["alert_level", "start_s", "end_s", "duration_s", "peak_score"])

    active = df[df["alert_level"] != "OK"].copy()
    if active.empty:
        return pd.DataFrame(columns=["alert_level", "start_s", "end_s", "duration_s", "peak_score"])

    group = (active["alert_level"] != active["alert_level"].shift()).cumsum()
    rows = []
    for _, part in active.groupby(group):
        rows.append(
            {
                "alert_level": part["alert_level"].iloc[0],
                "start_s": float(part["elapsed_s"].iloc[0]),
                "end_s": float(part["elapsed_s"].iloc[-1]),
                "duration_s": float(part["elapsed_s"].iloc[-1] - part["elapsed_s"].iloc[0]),
                "peak_score": float(part["score"].max()),
            }
        )
    return pd.DataFrame(rows)


def add_alert_bands(fig: go.Figure) -> go.Figure:
    bands = [
        (0, 25, "rgba(34,197,94,0.08)"),
        (25, 45, "rgba(132,204,22,0.10)"),
        (45, 70, "rgba(245,158,11,0.10)"),
        (70, 90, "rgba(239,68,68,0.10)"),
        (90, 100, "rgba(127,29,29,0.12)"),
    ]
    for y0, y1, color in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
    return fig


def risk_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    add_alert_bands(fig)
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_s"],
            y=df["score_smooth"],
            name="Smoothed score",
            mode="lines",
            line=dict(color="#16a34a", width=3),
            hovertemplate="Time %{x:.1f}s<br>Score %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_s"],
            y=df["score"],
            name="Frame score",
            mode="lines",
            line=dict(color="rgba(22,163,74,0.22)", width=1),
            hoverinfo="skip",
        )
    )
    alerts = df[df["alert_level"] != "OK"]
    if not alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=alerts["elapsed_s"],
                y=alerts["score"],
                name="Alert frames",
                mode="markers",
                marker=dict(
                    size=8,
                    color=alerts["alert_level"].map(ALERT_COLORS),
                    line=dict(color="white", width=1),
                ),
                text=alerts["alert_level"],
                hovertemplate="Time %{x:.1f}s<br>%{text}<br>Score %{y:.1f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Risk score", range=[0, 100]),
        xaxis=dict(title="Elapsed seconds"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def eye_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["elapsed_s"], y=df["ear"], name="EAR", line=dict(color="#2563eb", width=2)))
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_s"],
            y=df["ear_smooth"],
            name="EAR 5s avg",
            line=dict(color="#1d4ed8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_s"],
            y=df["perclos"],
            name="PERCLOS",
            yaxis="y2",
            fill="tozeroy",
            line=dict(color="rgba(245,158,11,0.8)", width=2),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Elapsed seconds"),
        yaxis=dict(title="EAR"),
        yaxis2=dict(title="PERCLOS", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def pose_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["elapsed_s"], y=df["pose_pitch"], name="Pitch", line=dict(color="#16a34a", width=2)))
    fig.add_trace(go.Scatter(x=df["elapsed_s"], y=df["pose_yaw"], name="Yaw", line=dict(color="#f59e0b", width=2)))
    fig.add_trace(go.Scatter(x=df["elapsed_s"], y=df["pose_roll"], name="Roll", line=dict(color="#dc2626", width=2)))
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_s"],
            y=df["pose_score"],
            name="Pose risk",
            yaxis="y2",
            line=dict(color="#111827", width=3, dash="dot"),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Elapsed seconds"),
        yaxis=dict(title="Degrees"),
        yaxis2=dict(title="Pose risk", overlaying="y", side="right", range=[0, max(30, df["pose_score"].max() + 2)]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def minute_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    grouped = df.groupby("minute", as_index=False).agg(
        avg_score=("score", "mean"),
        peak_score=("score", "max"),
        perclos=("perclos_pct", "mean"),
        pose=("pose_score", "mean"),
        microsleep=("microsleep_duration", "max"),
    )
    matrix = grouped[["avg_score", "peak_score", "perclos", "pose", "microsleep"]].T
    fig = px.imshow(
        matrix,
        labels=dict(x="Minute", y="Signal", color="Intensity"),
        x=[f"{m}m" for m in grouped["minute"]],
        y=["Avg score", "Peak score", "PERCLOS", "Pose risk", "Peak microsleep"],
        color_continuous_scale=["#ecfdf5", "#facc15", "#dc2626"],
        aspect="auto",
    )
    fig.update_layout(height=310, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def distribution_charts(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    fps = estimate_fps(df)
    alert_counts = df["alert_level"].value_counts().reindex(ALERT_ORDER, fill_value=0).reset_index()
    alert_counts.columns = ["alert_level", "frames"]
    alert_counts["seconds"] = alert_counts["frames"] / fps
    bar = px.bar(
        alert_counts,
        x="alert_level",
        y="seconds",
        color="alert_level",
        color_discrete_map=ALERT_COLORS,
        labels={"alert_level": "Alert level", "seconds": "Seconds"},
    )
    bar.update_layout(height=310, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))

    risk_counts = df["risk_band"].value_counts().reindex(ALERT_ORDER, fill_value=0).reset_index()
    risk_counts.columns = ["risk_band", "frames"]
    donut = px.pie(
        risk_counts,
        names="risk_band",
        values="frames",
        hole=0.58,
        color="risk_band",
        color_discrete_map=ALERT_COLORS,
    )
    donut.update_traces(textposition="inside", textinfo="percent+label")
    donut.update_layout(height=310, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    return bar, donut


def insight_lines(summary: SessionSummary, df: pd.DataFrame) -> list[str]:
    insights = []
    peak_row = df.loc[df["score"].idxmax()] if not df.empty else None
    if peak_row is not None:
        insights.append(
            f"Peak risk reached {summary.max_score:.0f}/100 at {peak_row['elapsed_s']:.1f}s, "
            f"with PERCLOS {peak_row['perclos_pct']:.1f}% and pose risk {peak_row['pose_score']:.0f}."
        )
    if summary.peak_microsleep >= 1.5:
        insights.append(f"Longest microsleep-like closure was {summary.peak_microsleep:.1f}s, which is alert-worthy.")
    elif summary.peak_microsleep > 0:
        insights.append(f"Eye closures stayed short; longest closure was {summary.peak_microsleep:.1f}s.")
    if summary.alert_seconds > 0:
        insights.append(f"Alerts were active for about {format_duration(summary.alert_seconds)} of the session.")
    else:
        insights.append("No alert episode was recorded in this session.")
    if summary.reliability_pct < 90:
        insights.append(f"Eye tracking reliability was {summary.reliability_pct:.0f}%; check camera angle or lighting.")
    else:
        insights.append(f"Eye tracking reliability was strong at {summary.reliability_pct:.0f}%.")
    if df["pose_score"].max() >= 12:
        insights.append("Head-pose movement contributed meaningfully to risk; review the pose chart around alert periods.")
    return insights


def render_session(df: pd.DataFrame, summary: SessionSummary) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>Driver Drowsiness Analytics</h1>
            <p>{summary.name} · {format_duration(summary.duration_s)} · {summary.rows:,} frames · estimated {summary.estimated_fps:.1f} FPS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Peak Score", f"{summary.max_score:.0f}/100")
    c2.metric("Average Score", f"{summary.mean_score:.1f}")
    c3.metric("Peak PERCLOS", f"{summary.peak_perclos:.1f}%")
    c4.metric("Peak Microsleep", f"{summary.peak_microsleep:.1f}s")
    c5.metric("Tracking", f"{summary.reliability_pct:.0f}%")

    st.subheader("Risk Timeline")
    st.plotly_chart(risk_timeline(df), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Eye Behavior")
        st.plotly_chart(eye_chart(df), use_container_width=True)
    with right:
        st.subheader("Head Pose")
        st.plotly_chart(pose_chart(df), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Alert Time")
        bar, donut = distribution_charts(df)
        st.plotly_chart(bar, use_container_width=True)
    with right:
        st.subheader("Risk Mix")
        st.plotly_chart(donut, use_container_width=True)

    st.subheader("Fatigue Load by Minute")
    st.plotly_chart(minute_heatmap(df), use_container_width=True)

    st.subheader("Session Notes")
    for line in insight_lines(summary, df):
        st.markdown(f"<div class='insight'>{line}</div>", unsafe_allow_html=True)

    episodes = alert_episodes(df)
    if not episodes.empty:
        st.subheader("Alert Episodes")
        pretty = episodes.copy()
        pretty["start"] = pretty["start_s"].map(format_duration)
        pretty["end"] = pretty["end_s"].map(format_duration)
        pretty["duration"] = pretty["duration_s"].map(format_duration)
        st.dataframe(
            pretty[["alert_level", "start", "end", "duration", "peak_score"]],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Raw engineered data"):
        st.dataframe(df, use_container_width=True, height=360)


def render_comparison(paths: list[Path]) -> None:
    rows = []
    for path in paths:
        df = load_session(str(path))
        rows.append(summarize(path, df).__dict__)
    if not rows:
        return
    summary_df = pd.DataFrame(rows)
    st.subheader("Session Comparison")
    st.dataframe(
        summary_df[
            [
                "name",
                "duration_s",
                "estimated_fps",
                "max_score",
                "mean_score",
                "peak_perclos",
                "peak_microsleep",
                "alert_seconds",
                "reliability_pct",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    fig = px.scatter(
        summary_df,
        x="mean_score",
        y="peak_microsleep",
        size="alert_seconds",
        color="max_score",
        hover_name="name",
        color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"],
        labels={
            "mean_score": "Average score",
            "peak_microsleep": "Peak microsleep",
            "alert_seconds": "Alert seconds",
            "max_score": "Peak score",
        },
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    apply_theme()
    files = list_session_files()
    if not files:
        st.warning(f"No session CSV files found in {SESSION_DIR}")
        return

    st.sidebar.title("Sessions")
    selected = st.sidebar.selectbox("Review session", files, format_func=lambda p: p.name)
    compare = st.sidebar.multiselect(
        "Compare sessions",
        files,
        default=files[: min(4, len(files))],
        format_func=lambda p: p.name,
    )
    st.sidebar.markdown(
        "<p class='small-muted'>Tip: run a driving session, quit with q, then refresh this dashboard.</p>",
        unsafe_allow_html=True,
    )

    df = load_session(str(selected))
    if df.empty:
        st.warning("Selected session is empty.")
        return

    summary = summarize(selected, df)
    tab_session, tab_compare = st.tabs(["Session Review", "Compare Runs"])
    with tab_session:
        render_session(df, summary)
    with tab_compare:
        render_comparison(compare)


if __name__ == "__main__":
    main()

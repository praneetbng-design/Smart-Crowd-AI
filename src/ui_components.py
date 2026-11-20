"""
Smart Crowd AI: UI Styling, Plotly Charts, and Visual Components
Delivers a high-contrast dark theme, risk gauges, time-series graphs,
and embedded offline audio-visual alert notifications.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components
from typing import Dict, Any, List, Optional


DARK_THEME_CSS = """
<style>
/* Global Dark Theme Settings */
.main {
    background-color: #0b0f19;
    color: #f1f5f9;
}
.stApp {
    background-color: #0b0f19;
}
header[data-testid="stHeader"] {
    background-color: rgba(11, 15, 25, 0.85);
    backdrop-filter: blur(8px);
}
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1f2937;
}

/* Glassmorphism Card Containers */
.smart-card {
    background: linear-gradient(145deg, #161f30, #111827);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.smart-card:hover {
    border-color: #38bdf8;
}

/* Metric Display Cards */
.metric-box {
    background: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: left;
    position: relative;
    overflow: hidden;
}
.metric-box::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}
.metric-box.blue::before { background-color: #38bdf8; }
.metric-box.green::before { background-color: #10b981; }
.metric-box.yellow::before { background-color: #f59e0b; }
.metric-box.red::before { background-color: #ef4444; }

.metric-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}
.metric-sub {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
}

/* Status Badges */
.badge-safe {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    background-color: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid #059669;
}
.badge-warning {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    background-color: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid #d97706;
}
.badge-critical {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    background-color: rgba(249, 115, 22, 0.18);
    color: #fb923c;
    border: 1px solid #ea580c;
}
.badge-stampede {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 800;
    background-color: rgba(239, 68, 68, 0.25);
    color: #f87171;
    border: 1px solid #dc2626;
    animation: pulse-red 1.5s infinite;
}
@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Simulation Pill */
.badge-simulated {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    background-color: #312e81;
    color: #c7d2fe;
    border: 1px dashed #6366f1;
    letter-spacing: 0.05em;
}
.badge-ai-real {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    background-color: #064e3b;
    color: #a7f3d0;
    border: 1px solid #10b981;
    letter-spacing: 0.05em;
}

/* Alert Notification Banner */
.alert-banner-danger {
    background: linear-gradient(90deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    color: #fef2f2;
    animation: flash-border 1.2s infinite alternate;
}
.alert-banner-warning {
    background: linear-gradient(90deg, #451a03, #78350f);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    color: #fffbeb;
}
@keyframes flash-border {
    from { border-color: #ef4444; }
    to { border-color: #fca5a5; }
}

/* Buttons and Inputs */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0px 0px;
    padding: 10px 18px;
}
</style>
"""


def render_metric_card(label: str, value: str, subtext: str, color_class: str = "blue") -> str:
    """Generates HTML for styled metric card."""
    return f"""
    <div class="metric-box {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """


def render_risk_badge(risk_level: str) -> str:
    """Returns styled HTML badge for risk."""
    if risk_level == "STAMPEDE RISK":
        return f'<span class="badge-stampede">🚨 {risk_level}</span>'
    elif risk_level == "CRITICAL":
        return f'<span class="badge-critical">⚠️ {risk_level}</span>'
    elif risk_level == "WARNING":
        return f'<span class="badge-warning">⚡ {risk_level}</span>'
    else:
        return f'<span class="badge-safe">🟢 {risk_level}</span>'


def render_origin_badge(is_simulated: bool, model_name: str = "YOLOv8") -> str:
    """Returns explicit badge distinguishing real AI detection from simulation."""
    if is_simulated:
        return '<span class="badge-simulated">🧪 SIMULATION MODE</span>'
    return f'<span class="badge-ai-real">⚡ LIVE AI: {model_name}</span>'


def render_audio_visual_alarm(risk_level: str, zone_name: str, message: str, advisory: str) -> str:
    """
    Renders high-priority alert banner card.
    """
    if risk_level not in ("CRITICAL", "STAMPEDE RISK", "WARNING"):
        return ""

    is_severe = risk_level in ("CRITICAL", "STAMPEDE RISK")
    banner_cls = "alert-banner-danger" if is_severe else "alert-banner-warning"
    icon = "🚨 CRITICAL STAMPEDE HAZARD" if risk_level == "STAMPEDE RISK" else ("⚠️ HIGH DENSITY BOTTLENECK" if risk_level == "CRITICAL" else "⚡ ELEVATED CROWD CAUTION")

    return f"""
    <div class="{banner_cls}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 16px; font-weight: 800; letter-spacing: 0.05em;">{icon}</span>
            <span style="font-size: 13px; font-weight: 700; background: rgba(0,0,0,0.3); padding: 3px 8px; border-radius: 4px;">Zone: {zone_name}</span>
        </div>
        <div style="font-size: 13px; margin-bottom: 8px;">{message}</div>
        <div style="font-size: 12px; font-weight: 600; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #f87171;">
            <b>Operational Directive:</b> {advisory}
        </div>
    </div>
    """


def play_alert_chime(risk_level: str, enable_audio: bool = True):
    """Triggers an offline Web Audio emergency chime using an invisible iframe."""
    if not enable_audio or risk_level not in ("CRITICAL", "STAMPEDE RISK"):
        return

    script = """
    <script>
    (function() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            const now = ctx.currentTime;
            [0, 0.22].forEach((offset) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(880, now + offset);
                osc.frequency.exponentialRampToValueAtTime(440, now + offset + 0.16);
                gain.gain.setValueAtTime(0.12, now + offset);
                gain.gain.exponentialRampToValueAtTime(0.01, now + offset + 0.16);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(now + offset);
                osc.stop(now + offset + 0.18);
            });
        } catch(e) {}
    })();
    </script>
    """
    components.html(script, height=0, width=0)


def create_risk_gauge_chart(score: float, risk_level: str) -> go.Figure:
    """Creates a circular risk score gauge."""
    if score >= 85:
        bar_color = "#ef4444"
    elif score >= 65:
        bar_color = "#f97316"
    elif score >= 40:
        bar_color = "#f59e0b"
    elif score >= 20:
        bar_color = "#3b82f6"
    else:
        bar_color = "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"Stampede Risk Index<br><span style='font-size:12px;color:#94a3b8;'>Status: {risk_level}</span>", 'font': {'size': 16, 'color': '#f1f5f9'}},
        number={'suffix': "/100", 'font': {'size': 28, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': "#94a3b8", 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "#1e293b",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.25)'},
                {'range': [40, 65], 'color': 'rgba(245, 158, 11, 0.25)'},
                {'range': [65, 85], 'color': 'rgba(249, 115, 22, 0.25)'},
                {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.35)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 3},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=220,
        margin=dict(l=25, r=25, t=45, b=15)
    )
    return fig


def create_density_timeline_chart(df: pd.DataFrame, warning_thresh: float = 2.0, crit_thresh: float = 3.5) -> go.Figure:
    """Creates real-time time-series crowd density and count chart."""
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No timeline detections recorded yet. Run a scan or start demo.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color="#94a3b8", size=13)
        )
    else:
        if "id" in df.columns:
            df_rev = df.sort_values(by="id").tail(30)
        else:
            df_rev = df.tail(30).copy()

        x_axis = df_rev["timestamp"].astype(str).apply(lambda t: t.split()[-1] if len(t.split()) > 1 else t)

        # Trace 1: Person Count
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df_rev["person_count"],
            name="Person Count",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=6, color="#38bdf8"),
            yaxis="y1"
        ))

        # Trace 2: Density
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df_rev["density"],
            name="Density (ppl/m²)",
            mode="lines",
            line=dict(color="#fbbf24", width=2, dash="dash"),
            yaxis="y2"
        ))

        # Threshold lines
        fig.add_hline(
            y=warning_thresh, yref="y2",
            line_dash="dot", line_color="#f59e0b",
            annotation_text=f"Warning ({warning_thresh} p/m²)",
            annotation_position="bottom right",
            annotation_font=dict(color="#f59e0b", size=10)
        )
        fig.add_hline(
            y=crit_thresh, yref="y2",
            line_dash="dot", line_color="#ef4444",
            annotation_text=f"Critical ({crit_thresh} p/m²)",
            annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=10)
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        height=270,
        margin=dict(l=40, r=40, t=25, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1")),
        xaxis=dict(
            title=dict(text="", font=dict(size=10)),
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8", size=10)
        ),
        yaxis=dict(
            title=dict(text="People Count", font=dict(color="#38bdf8", size=11)),
            tickfont=dict(color="#94a3b8", size=10),
            gridcolor="#1e293b"
        ),
        yaxis2=dict(
            title=dict(text="Density (p/m²)", font=dict(color="#fbbf24", size=11)),
            tickfont=dict(color="#94a3b8", size=10),
            overlaying="y",
            side="right"
        )
    )
    return fig


def create_zone_occupancy_bar_chart(zones: List[Dict[str, Any]]) -> go.Figure:
    """Renders horizontal comparison of zone capacity and current crowd counts."""
    fig = go.Figure()

    if not zones:
        fig.add_annotation(text="No zone telemetry available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#94a3b8"))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    zone_names = [z.get("zone_name", "Zone") for z in zones]
    counts = [z.get("current_count", 0) for z in zones]
    capacities = [z.get("max_capacity", 100) for z in zones]
    colors = [
        "#ef4444" if z.get("current_risk") == "STAMPEDE RISK"
        else ("#f97316" if z.get("current_risk") == "CRITICAL"
        else ("#f59e0b" if z.get("current_risk") == "WARNING" else "#10b981"))
        for z in zones
    ]

    fig.add_trace(go.Bar(
        y=zone_names,
        x=capacities,
        name="Max Rated Capacity",
        orientation='h',
        marker=dict(color='rgba(51, 65, 85, 0.45)', line=dict(color='#475569', width=1))
    ))

    fig.add_trace(go.Bar(
        y=zone_names,
        x=counts,
        name="Current Occupancy",
        orientation='h',
        marker=dict(color=colors)
    ))

    fig.update_layout(
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        height=max(220, len(zones) * 35),
        margin=dict(l=10, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1")),
        xaxis=dict(title=dict(text="Persons", font=dict(color="#94a3b8")), gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#e2e8f0", size=11))
    )
    return fig

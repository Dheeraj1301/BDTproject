"""Streamlit dashboard for the Quantum Traffic Flow Anomaly Analytics System."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    accident_hotspot_ranking,
    calculate_kpis,
    congestion_trend,
    filter_dataset,
    forecast_traffic_trend,
    prepare_dataset,
    top_anomalies,
    zone_summary,
)
from generate_data import DATA_FILE, DEFAULT_RECORDS, save_traffic_data

st.set_page_config(
    page_title="Quantum Traffic Flow Anomaly Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEON_TEMPLATE = "plotly_dark"


def inject_styles() -> None:
    """Add futuristic glassmorphism styles to Streamlit."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background: radial-gradient(circle at top left, #12305c 0, transparent 30%),
                        radial-gradient(circle at top right, #3b0d61 0, transparent 28%),
                        linear-gradient(135deg, #050816 0%, #090d1f 48%, #030712 100%);
            color: #e5f7ff;
        }
        section[data-testid="stSidebar"] {
            background: rgba(4, 10, 28, 0.88);
            border-right: 1px solid rgba(0, 245, 255, 0.24);
        }
        .hero {
            padding: 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(0,245,255,0.18), rgba(157,78,221,0.16));
            border: 1px solid rgba(0, 245, 255, 0.32);
            box-shadow: 0 0 42px rgba(0, 245, 255, 0.12);
            margin-bottom: 1.2rem;
        }
        .hero h1 {
            font-size: clamp(2.2rem, 5vw, 4.5rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-bottom: 0.4rem;
            color: #ffffff;
        }
        .hero p { color: #bdefff; font-size: 1.05rem; max-width: 900px; }
        .metric-card {
            min-height: 142px;
            padding: 1.2rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.075);
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 12px 34px rgba(0,0,0,0.22);
        }
        .metric-card .label { color: #93c5fd; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; }
        .metric-card .value { color: #ffffff; font-size: 2rem; font-weight: 800; margin-top: 0.35rem; }
        .metric-card .caption { color: #9ca3af; font-size: 0.85rem; margin-top: 0.4rem; }
        div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {
            border-radius: 999px;
            border: 1px solid rgba(0,245,255,0.45);
            background: linear-gradient(90deg, #00f5ff, #9d4edd);
            color: #020617;
            font-weight: 800;
        }
        .section-title {
            font-weight: 800;
            color: #f8fafc;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and prepare generated CSV data."""
    df = pd.read_csv(path)
    return prepare_dataset(df)


def ensure_data_exists() -> None:
    """Generate the default CSV if the project is running for the first time."""
    if not Path(DATA_FILE).exists():
        with st.spinner("Generating 100,000 quantum traffic telemetry records..."):
            save_traffic_data(DEFAULT_RECORDS)
        st.success("Synthetic traffic data generated successfully.")


def metric_card(label: str, value: str, caption: str) -> None:
    """Render a reusable glassmorphism metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply common neon chart formatting."""
    fig.update_layout(
        template=NEON_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,10,28,0.35)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="#e5f7ff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main() -> None:
    """Run the Streamlit analytics dashboard."""
    inject_styles()
    ensure_data_exists()

    with st.sidebar:
        st.title("🚦 Control Deck")
        st.caption("Filter the synthetic smart-city telemetry stream.")
        if st.button("Regenerate Synthetic Data", use_container_width=True):
            with st.spinner("Rebuilding synthetic traffic universe..."):
                save_traffic_data(DEFAULT_RECORDS)
                load_data.clear()
            st.success("Fresh synthetic data generated.")
            st.rerun()

    df = load_data(str(DATA_FILE))

    with st.sidebar:
        zones = st.multiselect("City zone", sorted(df["city_zone"].unique()))
        weather = st.multiselect("Weather", sorted(df["weather_condition"].unique()))
        vehicle_types = st.multiselect("Vehicle type", sorted(df["vehicle_type"].unique()))
        congestion_levels = st.multiselect(
            "Congestion level",
            ["Low", "Moderate", "High", "Critical"],
        )
        st.download_button(
            "Download Filtered CSV",
            data=filter_dataset(df, zones, weather, vehicle_types, congestion_levels).to_csv(index=False),
            file_name="filtered_quantum_traffic_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    filtered = filter_dataset(df, zones, weather, vehicle_types, congestion_levels)
    kpis = calculate_kpis(filtered)

    st.markdown(
        """
        <div class="hero">
            <div>⚡ SMART CITY BIG DATA SIMULATION</div>
            <h1>Quantum Traffic Flow Anomaly Analytics System</h1>
            <p>Analyze 100,000+ synthetic vehicle events with congestion intelligence, accident-risk scoring,
            anomaly detection, and lightweight trend forecasting in a futuristic dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("No records match the current filters. Adjust the sidebar controls to continue.")
        return

    card_cols = st.columns(5)
    with card_cols[0]:
        metric_card("Vehicles analyzed", f"{int(kpis['total_vehicles']):,}", "Filtered telemetry records")
    with card_cols[1]:
        metric_card("Congestion", f"{kpis['congestion_percentage']:.1f}%", "High or critical flow")
    with card_cols[2]:
        metric_card("AI anomaly score", f"{kpis['average_anomaly_score']:.3f}", "Average model-inspired risk")
    with card_cols[3]:
        metric_card("Accident risk", f"{kpis['average_accident_probability']:.1f}%", "Mean probability")
    with card_cols[4]:
        metric_card("Avg speed", f"{kpis['average_speed']:.1f} km/h", "Network velocity")

    left, right = st.columns((1.35, 1))
    with left:
        st.markdown('<h3 class="section-title">Congestion Trend + Lightweight Forecast</h3>', unsafe_allow_html=True)
        trend = congestion_trend(filtered)
        forecast = forecast_traffic_trend(filtered)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["timestamp_hour"], y=trend["traffic_density"], mode="lines", name="Observed density", line=dict(color="#00f5ff", width=3)))
        fig.add_trace(go.Scatter(x=trend["timestamp_hour"], y=trend["ai_anomaly_score"] * 100, mode="lines", name="AI anomaly x100", line=dict(color="#ff4ecd", width=2)))
        fig.add_trace(go.Scatter(x=forecast["timestamp_hour"], y=forecast["forecast_density"], mode="lines+markers", name="Forecast density", line=dict(color="#facc15", dash="dash", width=3)))
        st.plotly_chart(plotly_layout(fig), use_container_width=True)
    with right:
        st.markdown('<h3 class="section-title">Vehicle Type Distribution</h3>', unsafe_allow_html=True)
        vehicle_counts = filtered["vehicle_type"].value_counts().reset_index()
        vehicle_counts.columns = ["vehicle_type", "count"]
        fig = px.pie(vehicle_counts, names="vehicle_type", values="count", hole=0.48, color_discrete_sequence=px.colors.sequential.Plasma_r)
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    heat_col, risk_col = st.columns(2)
    with heat_col:
        st.markdown('<h3 class="section-title">Congestion Heat Analysis</h3>', unsafe_allow_html=True)
        heat = filtered.pivot_table(index="city_zone", columns="hour", values="traffic_density", aggfunc="mean").fillna(0)
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="turbo", labels=dict(color="Density"))
        st.plotly_chart(plotly_layout(fig), use_container_width=True)
    with risk_col:
        st.markdown('<h3 class="section-title">Accident Hotspot Ranking</h3>', unsafe_allow_html=True)
        hotspots = accident_hotspot_ranking(filtered)
        fig = px.bar(
            hotspots,
            x="accident_probability",
            y="city_zone",
            orientation="h",
            color="ai_anomaly_score",
            color_continuous_scale="magma",
            labels={"accident_probability": "Accident probability", "city_zone": "Zone"},
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    weather_col, speed_col = st.columns(2)
    with weather_col:
        st.markdown('<h3 class="section-title">Weather Impact on Traffic</h3>', unsafe_allow_html=True)
        weather_impact = filtered.groupby("weather_condition", as_index=False).agg(
            traffic_density=("traffic_density", "mean"), accident_probability=("accident_probability", "mean")
        )
        fig = px.scatter(
            weather_impact,
            x="traffic_density",
            y="accident_probability",
            size="traffic_density",
            color="weather_condition",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(plotly_layout(fig), use_container_width=True)
    with speed_col:
        st.markdown('<h3 class="section-title">Speed Distribution</h3>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x="average_speed", color="congestion_level", nbins=45, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(plotly_layout(fig), use_container_width=True)

    st.markdown('<h3 class="section-title">Zone-wise Analytics</h3>', unsafe_allow_html=True)
    zone_df = zone_summary(filtered)
    st.dataframe(
        zone_df.style.format(
            {
                "traffic_density": "{:.2f}",
                "average_speed": "{:.2f}",
                "signal_wait_time": "{:.2f}",
                "accident_probability": "{:.2%}",
                "ai_anomaly_score": "{:.3f}",
                "vehicle_count": "{:,}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    anomaly_col, top_zone_col = st.columns((1.45, 1))
    with anomaly_col:
        st.markdown('<h3 class="section-title">AI Anomaly Detection Table</h3>', unsafe_allow_html=True)
        st.dataframe(top_anomalies(filtered), use_container_width=True, hide_index=True)
    with top_zone_col:
        st.markdown('<h3 class="section-title">Top Anomaly Zones</h3>', unsafe_allow_html=True)
        top_zones = zone_df.head(8)
        fig = px.bar(top_zones, x="city_zone", y="ai_anomaly_score", color="traffic_density", color_continuous_scale="electric")
        st.plotly_chart(plotly_layout(fig, height=390), use_container_width=True)


if __name__ == "__main__":
    main()

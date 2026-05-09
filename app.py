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
from icu_batch_preprocessing import (
    DEFAULT_ICU_RECORDS,
    ICU_PREPROCESSED_FILE,
    ICU_RAW_FILE,
    icu_batch_summary,
    run_pyspark_batch_simulation,
    save_icu_data,
)

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


@st.cache_data(show_spinner=False)
def load_icu_preprocessed(path: str) -> pd.DataFrame:
    """Load a preprocessed ICU batch output CSV when available."""
    return pd.read_csv(path)


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
            <div>⚡ SMART CITY BIG DATA + ICU BATCH SIMULATION</div>
            <h1>Quantum Traffic Flow Anomaly Analytics System</h1>
            <p>Analyze 100,000+ synthetic vehicle events and simulate PySpark-style preprocessing for a
            large ICU dataset from one beginner-friendly, futuristic analytics frontend.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    traffic_tab, icu_tab, data_hub_tab = st.tabs([
        "🚦 Traffic Command Center",
        "🏥 ICU PySpark Batch Simulation",
        "🧬 Synthetic Data Hub",
    ])

    with traffic_tab:
        if filtered.empty:
            st.warning("No records match the current filters. Adjust the sidebar controls to continue.")
        else:
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

    with icu_tab:
        st.markdown('<h3 class="section-title">PySpark Batch Preprocessing Simulation for Large ICU Dataset</h3>', unsafe_allow_html=True)
        st.caption(
            "This simulates a Spark batch pipeline locally: extract ICU CSV rows, repartition, transform partitions, "
            "compute severity features, aggregate alert rates, and write a preprocessed dataset."
        )
        icu_controls, icu_log_panel = st.columns((1, 1.2))
        with icu_controls:
            icu_records = st.slider("Synthetic ICU rows", min_value=10_000, max_value=250_000, value=DEFAULT_ICU_RECORDS, step=10_000)
            partitions = st.slider("Simulated Spark partitions", min_value=2, max_value=16, value=8, step=1)
            if st.button("Generate ICU Synthetic Data", use_container_width=True):
                with st.spinner("Generating large synthetic ICU dataset..."):
                    save_icu_data(num_records=icu_records)
                    load_icu_preprocessed.clear()
                st.success(f"Generated {icu_records:,} ICU records at {ICU_RAW_FILE}.")
            if st.button("Run PySpark Batch Preprocessing Simulation", use_container_width=True):
                with st.spinner("Running simulated PySpark batch preprocessing..."):
                    preprocessed, logs = run_pyspark_batch_simulation(partitions=partitions)
                    load_icu_preprocessed.clear()
                st.session_state["icu_batch_logs"] = logs
                st.session_state["icu_batch_rows"] = len(preprocessed)
                st.success("ICU batch preprocessing simulation completed.")

        with icu_log_panel:
            st.markdown('<h4 class="section-title">Batch Event Log</h4>', unsafe_allow_html=True)
            for event in st.session_state.get("icu_batch_logs", ["Waiting for ICU batch simulation event..."]):
                st.code(event, language="text")

        if Path(ICU_PREPROCESSED_FILE).exists():
            icu_df = load_icu_preprocessed(str(ICU_PREPROCESSED_FILE))
            icu_kpis = icu_batch_summary(icu_df)
            icu_cards = st.columns(4)
            with icu_cards[0]:
                metric_card("ICU rows processed", f"{icu_kpis['rows']:,}", "Preprocessed batch output")
            with icu_cards[1]:
                metric_card("Batch alert rate", f"{icu_kpis['alert_rate']:.1f}%", "Rows requiring review")
            with icu_cards[2]:
                metric_card("Avg mortality risk", f"{icu_kpis['avg_mortality_risk']:.3f}", "Synthetic risk score")
            with icu_cards[3]:
                metric_card("Highest alert unit", str(icu_kpis["top_unit"]), "By alert percentage")

            chart_col, table_col = st.columns((1.1, 1))
            with chart_col:
                severity_counts = icu_df["icu_severity_band"].value_counts().reset_index()
                severity_counts.columns = ["icu_severity_band", "count"]
                fig = px.bar(severity_counts, x="icu_severity_band", y="count", color="icu_severity_band", color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(plotly_layout(fig, height=360), use_container_width=True)
            with table_col:
                alert_summary = icu_df.groupby("icu_unit", as_index=False).agg(
                    rows=("patient_id", "count"),
                    alert_rate=("batch_alert", "mean"),
                    avg_risk=("mortality_risk", "mean"),
                    avg_shock_index=("shock_index", "mean"),
                ).sort_values("alert_rate", ascending=False)
                st.dataframe(
                    alert_summary.style.format({"alert_rate": "{:.2%}", "avg_risk": "{:.3f}", "avg_shock_index": "{:.2f}", "rows": "{:,}"}),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download Preprocessed ICU CSV",
                data=icu_df.to_csv(index=False),
                file_name="icu_preprocessed_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Generate ICU data and run the batch preprocessing simulation to unlock ICU KPIs and charts.")

    with data_hub_tab:
        st.markdown('<h3 class="section-title">Synthetic Data Hub</h3>', unsafe_allow_html=True)
        st.write(
            "Use this frontend hub to manage both generated datasets and track the latest data events for demos, viva, and screenshots."
        )
        hub_cols = st.columns(2)
        with hub_cols[0]:
            st.markdown("#### Traffic synthetic data")
            st.write(f"Default output: `{DATA_FILE}`")
            st.write(f"Default rows: `{DEFAULT_RECORDS:,}`")
            st.download_button(
                "Download Current Traffic CSV",
                data=df.to_csv(index=False),
                file_name="traffic_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with hub_cols[1]:
            st.markdown("#### ICU synthetic data")
            st.write(f"Raw output: `{ICU_RAW_FILE}`")
            st.write(f"Preprocessed output: `{ICU_PREPROCESSED_FILE}`")
            st.write("Frontend events are shown in the ICU tab after every batch run.")
            if Path(ICU_RAW_FILE).exists():
                st.download_button(
                    "Download Current ICU Raw CSV",
                    data=pd.read_csv(ICU_RAW_FILE).to_csv(index=False),
                    file_name="icu_raw_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()

"""Analytics helpers for the Quantum Traffic Flow Anomaly Analytics dashboard."""

from __future__ import annotations

import pandas as pd

HIGH_CONGESTION_LEVELS = {"High", "Critical"}


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean DataFrame with parsed timestamps and helper columns."""
    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["hour"] = prepared["timestamp"].dt.hour
    prepared["date"] = prepared["timestamp"].dt.date
    prepared["is_anomaly"] = prepared["ai_anomaly_score"] >= 0.78
    prepared["is_high_congestion"] = prepared["congestion_level"].isin(HIGH_CONGESTION_LEVELS)
    return prepared


def filter_dataset(
    df: pd.DataFrame,
    zones: list[str],
    weather: list[str],
    vehicle_types: list[str],
    congestion_levels: list[str],
) -> pd.DataFrame:
    """Apply dashboard sidebar filters without mutating the original data."""
    filtered = df.copy()
    if zones:
        filtered = filtered[filtered["city_zone"].isin(zones)]
    if weather:
        filtered = filtered[filtered["weather_condition"].isin(weather)]
    if vehicle_types:
        filtered = filtered[filtered["vehicle_type"].isin(vehicle_types)]
    if congestion_levels:
        filtered = filtered[filtered["congestion_level"].isin(congestion_levels)]
    return filtered


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Calculate headline metrics for dashboard cards."""
    if df.empty:
        return {
            "total_vehicles": 0,
            "congestion_percentage": 0.0,
            "average_anomaly_score": 0.0,
            "average_accident_probability": 0.0,
            "average_speed": 0.0,
        }

    return {
        "total_vehicles": float(len(df)),
        "congestion_percentage": float(df["is_high_congestion"].mean() * 100),
        "average_anomaly_score": float(df["ai_anomaly_score"].mean()),
        "average_accident_probability": float(df["accident_probability"].mean() * 100),
        "average_speed": float(df["average_speed"].mean()),
    }


def accident_hotspot_ranking(df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Rank zones by average accident risk and severe congestion volume."""
    if df.empty:
        return pd.DataFrame(columns=["city_zone", "accident_probability", "ai_anomaly_score", "is_high_congestion"])
    return (
        df.groupby("city_zone", as_index=False)
        .agg(
            accident_probability=("accident_probability", "mean"),
            ai_anomaly_score=("ai_anomaly_score", "mean"),
            is_high_congestion=("is_high_congestion", "sum"),
        )
        .sort_values(["accident_probability", "ai_anomaly_score"], ascending=False)
        .head(limit)
    )


def zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize zone-level density, speed, wait time, risk, and anomaly performance."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("city_zone", as_index=False)
        .agg(
            traffic_density=("traffic_density", "mean"),
            average_speed=("average_speed", "mean"),
            signal_wait_time=("signal_wait_time", "mean"),
            accident_probability=("accident_probability", "mean"),
            ai_anomaly_score=("ai_anomaly_score", "mean"),
            vehicle_count=("vehicle_id", "count"),
        )
        .sort_values("ai_anomaly_score", ascending=False)
    )


def congestion_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Build an hourly congestion trend for Plotly line charts."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp_hour", "traffic_density", "ai_anomaly_score", "accident_probability"])
    trend = df.set_index("timestamp").resample("h").agg(
        traffic_density=("traffic_density", "mean"),
        ai_anomaly_score=("ai_anomaly_score", "mean"),
        accident_probability=("accident_probability", "mean"),
    )
    return trend.reset_index().rename(columns={"timestamp": "timestamp_hour"}).dropna()


def forecast_traffic_trend(df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    """Create a simple lightweight traffic forecast using rolling averages.

    This is intentionally not a heavy ML model. It gives the dashboard a clear
    prediction component while remaining easy to understand for beginners.
    """
    trend = congestion_trend(df)
    if trend.empty:
        return pd.DataFrame(columns=["timestamp_hour", "forecast_density"])

    recent = trend.tail(24).copy()
    base = recent["traffic_density"].rolling(window=6, min_periods=1).mean().iloc[-1]
    momentum = recent["traffic_density"].diff().tail(6).mean()
    momentum = 0 if pd.isna(momentum) else momentum
    future_index = pd.date_range(
        start=trend["timestamp_hour"].max() + pd.Timedelta(hours=1), periods=periods, freq="h"
    )
    forecast_values = [max(0, min(100, base + momentum * (step + 1))) for step in range(periods)]
    return pd.DataFrame({"timestamp_hour": future_index, "forecast_density": forecast_values})


def top_anomalies(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Return the highest AI anomaly records for the detection table."""
    columns = [
        "timestamp",
        "vehicle_id",
        "city_zone",
        "vehicle_type",
        "traffic_density",
        "average_speed",
        "accident_probability",
        "congestion_level",
        "ai_anomaly_score",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df.sort_values("ai_anomaly_score", ascending=False).head(limit)[columns]

"""Synthetic data generator for the Quantum Traffic Flow Anomaly Analytics System.

The generator intentionally stays beginner-friendly: it uses NumPy arrays for fast
bulk generation, Faker for realistic-looking vehicle IDs, and simple statistical
rules to inject traffic anomalies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from faker import Faker

DATA_DIR: Final[Path] = Path("data")
DATA_FILE: Final[Path] = DATA_DIR / "traffic_data.csv"
DEFAULT_RECORDS: Final[int] = 100_000
RANDOM_SEED: Final[int] = 42

CITY_ZONES: Final[list[str]] = [
    "Neon North",
    "Quantum Square",
    "Metro Central",
    "Skyline East",
    "Hyperloop West",
    "Aurora Industrial",
    "Solaris South",
    "Cyber Harbor",
]
VEHICLE_TYPES: Final[list[str]] = ["Car", "Bus", "Truck", "Motorbike", "EV Shuttle", "Autonomous Cab"]
WEATHER_CONDITIONS: Final[list[str]] = ["Clear", "Cloudy", "Rain", "Fog", "Storm", "Heatwave"]
ROAD_CONDITIONS: Final[list[str]] = ["Smooth", "Wet", "Under Repair", "Potholes", "Low Visibility"]
CONGESTION_LABELS: Final[list[str]] = ["Low", "Moderate", "High", "Critical"]


def _classify_congestion(density: np.ndarray) -> np.ndarray:
    """Convert numerical density values into readable congestion categories."""
    return pd.cut(
        density,
        bins=[-1, 35, 60, 80, 101],
        labels=CONGESTION_LABELS,
    ).astype(str)


def generate_traffic_data(num_records: int = DEFAULT_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a synthetic smart-city traffic dataset.

    Args:
        num_records: Number of rows to generate. The default satisfies the
            project's 100,000+ record requirement.
        seed: Random seed for reproducible classroom demos.

    Returns:
        A pandas DataFrame containing traffic telemetry and AI-inspired scores.
    """
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    end_time = pd.Timestamp.now().floor("min")
    timestamps = pd.date_range(end=end_time, periods=num_records, freq="min")

    city_zone = rng.choice(CITY_ZONES, size=num_records, p=[0.12, 0.14, 0.18, 0.11, 0.12, 0.11, 0.1, 0.12])
    vehicle_type = rng.choice(VEHICLE_TYPES, size=num_records, p=[0.42, 0.1, 0.13, 0.18, 0.08, 0.09])
    weather = rng.choice(WEATHER_CONDITIONS, size=num_records, p=[0.44, 0.2, 0.14, 0.08, 0.06, 0.08])

    road_by_weather = {
        "Clear": [0.76, 0.04, 0.08, 0.1, 0.02],
        "Cloudy": [0.62, 0.1, 0.1, 0.12, 0.06],
        "Rain": [0.18, 0.55, 0.08, 0.11, 0.08],
        "Fog": [0.22, 0.1, 0.08, 0.08, 0.52],
        "Storm": [0.08, 0.48, 0.12, 0.12, 0.2],
        "Heatwave": [0.58, 0.03, 0.18, 0.18, 0.03],
    }
    road_condition = np.array([
        rng.choice(ROAD_CONDITIONS, p=road_by_weather[condition]) for condition in weather
    ])

    hour = timestamps.hour.to_numpy()
    peak_multiplier = np.where(((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 20)), 1.32, 0.9)
    weather_pressure = np.select(
        [weather == "Storm", weather == "Rain", weather == "Fog", weather == "Heatwave"],
        [18, 11, 9, 6],
        default=0,
    )
    zone_pressure = pd.Series(city_zone).map(
        {
            "Metro Central": 14,
            "Quantum Square": 11,
            "Cyber Harbor": 8,
            "Neon North": 7,
            "Hyperloop West": 6,
            "Skyline East": 5,
            "Aurora Industrial": 9,
            "Solaris South": 3,
        }
    ).to_numpy()

    base_density = rng.normal(42, 14, num_records) * peak_multiplier + weather_pressure + zone_pressure
    traffic_density = np.clip(base_density, 5, 100)

    signal_wait_time = np.clip(traffic_density * rng.uniform(0.45, 1.25, num_records) + rng.normal(6, 8, num_records), 0, 180)
    speed_penalty = traffic_density * rng.uniform(0.38, 0.58, num_records)
    average_speed = np.clip(88 - speed_penalty - weather_pressure * 0.6 + rng.normal(0, 7, num_records), 3, 110)

    road_risk = pd.Series(road_condition).map(
        {"Smooth": 0.02, "Wet": 0.08, "Under Repair": 0.11, "Potholes": 0.13, "Low Visibility": 0.15}
    ).to_numpy()
    accident_probability = np.clip(
        0.03 + traffic_density / 260 + signal_wait_time / 750 + road_risk + rng.normal(0, 0.025, num_records),
        0,
        0.98,
    )

    # Synthetic anomaly injection: spikes combine gridlock, speed collapse, and risk surges.
    anomaly_count = max(1, int(num_records * 0.035))
    anomaly_idx = rng.choice(num_records, size=anomaly_count, replace=False)
    traffic_density[anomaly_idx] = np.clip(traffic_density[anomaly_idx] + rng.uniform(22, 42, anomaly_count), 0, 100)
    average_speed[anomaly_idx] = np.clip(average_speed[anomaly_idx] - rng.uniform(18, 45, anomaly_count), 3, 110)
    signal_wait_time[anomaly_idx] = np.clip(signal_wait_time[anomaly_idx] + rng.uniform(30, 95, anomaly_count), 0, 180)
    accident_probability[anomaly_idx] = np.clip(accident_probability[anomaly_idx] + rng.uniform(0.16, 0.36, anomaly_count), 0, 0.98)

    density_norm = traffic_density / 100
    wait_norm = signal_wait_time / 180
    speed_inverse = 1 - (average_speed / 110)
    ai_anomaly_score = np.clip(
        (density_norm * 0.35 + wait_norm * 0.25 + speed_inverse * 0.22 + accident_probability * 0.18)
        + rng.normal(0, 0.035, num_records),
        0,
        1,
    )
    ai_anomaly_score[anomaly_idx] = np.clip(ai_anomaly_score[anomaly_idx] + rng.uniform(0.12, 0.28, anomaly_count), 0, 1)

    vehicle_ids = [f"QTF-{fake.unique.bothify(text='??-#####').upper()}" for _ in range(num_records)]

    return pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "timestamp": timestamps,
            "city_zone": city_zone,
            "vehicle_type": vehicle_type,
            "traffic_density": np.round(traffic_density, 2),
            "average_speed": np.round(average_speed, 2),
            "weather_condition": weather,
            "road_condition": road_condition,
            "signal_wait_time": np.round(signal_wait_time, 2),
            "accident_probability": np.round(accident_probability, 4),
            "congestion_level": _classify_congestion(traffic_density),
            "ai_anomaly_score": np.round(ai_anomaly_score, 4),
        }
    )


def save_traffic_data(num_records: int = DEFAULT_RECORDS, output_path: Path = DATA_FILE) -> pd.DataFrame:
    """Generate traffic data and save it as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_traffic_data(num_records=num_records)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generated = save_traffic_data()
    print(f"Generated {len(generated):,} synthetic traffic records at {DATA_FILE}")

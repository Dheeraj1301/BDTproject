"""PySpark-style batch preprocessing simulation for a large ICU dataset.

This module intentionally avoids requiring a real Spark cluster so the project
stays easy to run on a laptop. The function and log names mirror a PySpark batch
job: extract, repartition, transform, aggregate, and write. Internally, Pandas
and NumPy perform the work in chunks to simulate distributed preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

ICU_RAW_FILE: Final[Path] = Path("data") / "icu_raw_data.csv"
ICU_PREPROCESSED_FILE: Final[Path] = Path("data") / "icu_preprocessed_data.csv"
DEFAULT_ICU_RECORDS: Final[int] = 120_000
DEFAULT_PARTITIONS: Final[int] = 8
RANDOM_SEED: Final[int] = 2026

ICU_UNITS: Final[list[str]] = [
    "Cardiac ICU",
    "Neuro ICU",
    "Trauma ICU",
    "Pediatric ICU",
    "Medical ICU",
    "Surgical ICU",
]
DIAGNOSES: Final[list[str]] = ["Sepsis", "Respiratory Failure", "Stroke", "Cardiac Arrest", "Trauma", "Post Surgery"]
VENTILATION_STATUS: Final[list[str]] = ["Room Air", "Oxygen Mask", "Non-invasive", "Ventilator"]


def generate_icu_data(num_records: int = DEFAULT_ICU_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a large synthetic ICU monitoring dataset.

    The generated values are artificial and must not be used for clinical
    decisions. They exist only to demonstrate batch preprocessing and analytics.
    """
    rng = np.random.default_rng(seed)
    end_time = pd.Timestamp.now().floor("min")
    timestamps = pd.date_range(end=end_time, periods=num_records, freq="2min")

    age = np.clip(rng.normal(58, 18, num_records).round(), 1, 96).astype(int)
    heart_rate = np.clip(rng.normal(88, 22, num_records), 32, 190)
    systolic_bp = np.clip(rng.normal(118, 24, num_records), 55, 230)
    respiratory_rate = np.clip(rng.normal(20, 6, num_records), 6, 48)
    oxygen_saturation = np.clip(rng.normal(94, 5, num_records), 55, 100)
    temperature_c = np.clip(rng.normal(37.1, 1.1, num_records), 33, 42)
    glucose_mg_dl = np.clip(rng.normal(142, 54, num_records), 45, 440)
    lactate_mmol_l = np.clip(rng.gamma(2.0, 1.15, num_records), 0.3, 14)
    length_of_stay_hours = np.clip(rng.gamma(5.2, 18, num_records), 4, 720)

    # Inject synthetic clinical deterioration events to make preprocessing useful.
    event_count = max(1, int(num_records * 0.045))
    event_idx = rng.choice(num_records, size=event_count, replace=False)
    heart_rate[event_idx] = np.clip(heart_rate[event_idx] + rng.uniform(25, 70, event_count), 32, 190)
    systolic_bp[event_idx] = np.clip(systolic_bp[event_idx] - rng.uniform(25, 55, event_count), 55, 230)
    oxygen_saturation[event_idx] = np.clip(oxygen_saturation[event_idx] - rng.uniform(10, 30, event_count), 55, 100)
    lactate_mmol_l[event_idx] = np.clip(lactate_mmol_l[event_idx] + rng.uniform(2, 7, event_count), 0.3, 14)

    diagnosis = rng.choice(DIAGNOSES, size=num_records, p=[0.24, 0.22, 0.13, 0.12, 0.14, 0.15])
    ventilation = rng.choice(VENTILATION_STATUS, size=num_records, p=[0.38, 0.28, 0.16, 0.18])
    icu_unit = rng.choice(ICU_UNITS, size=num_records)

    severity_score = np.clip(
        (heart_rate - 60) / 130 * 0.18
        + (130 - systolic_bp) / 90 * 0.22
        + (100 - oxygen_saturation) / 45 * 0.26
        + lactate_mmol_l / 14 * 0.22
        + respiratory_rate / 48 * 0.12,
        0,
        1,
    )
    mortality_risk = np.clip(severity_score + rng.normal(0, 0.045, num_records), 0, 1)

    return pd.DataFrame(
        {
            "patient_id": [f"ICU-{idx:07d}" for idx in range(1, num_records + 1)],
            "admission_timestamp": timestamps,
            "icu_unit": icu_unit,
            "age": age,
            "diagnosis": diagnosis,
            "heart_rate": np.round(heart_rate, 1),
            "systolic_bp": np.round(systolic_bp, 1),
            "respiratory_rate": np.round(respiratory_rate, 1),
            "oxygen_saturation": np.round(oxygen_saturation, 1),
            "temperature_c": np.round(temperature_c, 1),
            "glucose_mg_dl": np.round(glucose_mg_dl, 1),
            "lactate_mmol_l": np.round(lactate_mmol_l, 2),
            "ventilation_status": ventilation,
            "length_of_stay_hours": np.round(length_of_stay_hours, 1),
            "mortality_risk": np.round(mortality_risk, 4),
            "critical_event": mortality_risk >= 0.72,
        }
    )


def save_icu_data(num_records: int = DEFAULT_ICU_RECORDS, output_path: Path = ICU_RAW_FILE) -> pd.DataFrame:
    """Generate and save synthetic ICU data for the simulated batch job."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_icu_data(num_records=num_records)
    df.to_csv(output_path, index=False)
    return df


def _preprocess_partition(partition: pd.DataFrame, partition_id: int) -> pd.DataFrame:
    """Transform one simulated Spark partition."""
    transformed = partition.copy()
    transformed["partition_id"] = partition_id
    transformed["admission_timestamp"] = pd.to_datetime(transformed["admission_timestamp"])
    transformed["admission_hour"] = transformed["admission_timestamp"].dt.hour
    transformed["shock_index"] = transformed["heart_rate"] / transformed["systolic_bp"].replace(0, np.nan)
    transformed["oxygen_gap"] = 100 - transformed["oxygen_saturation"]
    transformed["icu_severity_band"] = pd.cut(
        transformed["mortality_risk"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Stable", "Watchlist", "Critical"],
    ).astype(str)
    transformed["batch_alert"] = (
        (transformed["shock_index"] >= 0.9)
        | (transformed["oxygen_saturation"] <= 88)
        | (transformed["lactate_mmol_l"] >= 4)
        | (transformed["critical_event"])
    )
    return transformed


def run_pyspark_batch_simulation(
    input_path: Path = ICU_RAW_FILE,
    output_path: Path = ICU_PREPROCESSED_FILE,
    partitions: int = DEFAULT_PARTITIONS,
) -> tuple[pd.DataFrame, list[str]]:
    """Run a PySpark-style batch preprocessing simulation over ICU CSV data.

    Returns the preprocessed DataFrame plus a frontend-friendly event log.
    """
    if not input_path.exists():
        save_icu_data(output_path=input_path)

    raw = pd.read_csv(input_path)
    logs = [
        f"SparkSession simulated for local batch mode",
        f"Extracted {len(raw):,} ICU rows from {input_path}",
        f"Repartitioned source dataset into {partitions} logical partitions",
    ]

    partition_frames = np.array_split(raw, partitions)
    transformed_partitions: list[pd.DataFrame] = []
    for partition_id, partition in enumerate(partition_frames, start=1):
        transformed = _preprocess_partition(partition, partition_id)
        transformed_partitions.append(transformed)
        logs.append(f"Partition {partition_id:02d}: transformed {len(transformed):,} rows")

    preprocessed = pd.concat(transformed_partitions, ignore_index=True)
    unit_summary = preprocessed.groupby("icu_unit")["batch_alert"].mean().sort_values(ascending=False)
    logs.append("Aggregated batch alert rates by ICU unit")
    logs.append(f"Highest alert unit: {unit_summary.index[0]} ({unit_summary.iloc[0] * 100:.1f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed.to_csv(output_path, index=False)
    logs.append(f"Wrote preprocessed ICU dataset to {output_path}")
    return preprocessed, logs


def icu_batch_summary(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Create concise KPIs for the ICU batch frontend panel."""
    if df.empty:
        return {"rows": 0, "alert_rate": 0.0, "avg_mortality_risk": 0.0, "top_unit": "N/A"}

    top_unit = df.groupby("icu_unit")["batch_alert"].mean().sort_values(ascending=False).index[0]
    return {
        "rows": int(len(df)),
        "alert_rate": float(df["batch_alert"].mean() * 100),
        "avg_mortality_risk": float(df["mortality_risk"].mean()),
        "top_unit": str(top_unit),
    }


if __name__ == "__main__":
    save_icu_data()
    _, event_log = run_pyspark_batch_simulation()
    print("\n".join(event_log))

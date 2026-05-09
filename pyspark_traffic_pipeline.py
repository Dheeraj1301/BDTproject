"""Real PySpark preprocessing pipeline for traffic telemetry.

Unlike the ICU module, this file uses an actual ``pyspark.sql.SparkSession``.
The Streamlit frontend calls it on demand so users can demonstrate a genuine
batch preprocessing job without making the main dashboard depend on Spark at
startup.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Final

import pandas as pd

from generate_data import DATA_FILE, DEFAULT_RECORDS, save_traffic_data

TPPP_DIR: Final[Path] = Path("data") / "pyspark_tppp"
TPPP_SUMMARY_FILE: Final[Path] = TPPP_DIR / "traffic_zone_weather_summary.csv"
TPPP_ANOMALY_FILE: Final[Path] = TPPP_DIR / "traffic_top_anomalies.csv"
TPPP_CURATED_DIR: Final[Path] = TPPP_DIR / "curated_parquet"


def is_pyspark_available() -> bool:
    """Return whether PySpark is importable in the current Python environment."""
    return importlib.util.find_spec("pyspark") is not None


def run_traffic_pyspark_pipeline(
    input_path: Path = DATA_FILE,
    summary_path: Path = TPPP_SUMMARY_FILE,
    anomaly_path: Path = TPPP_ANOMALY_FILE,
    curated_output_dir: Path = TPPP_CURATED_DIR,
    anomaly_limit: int = 100,
) -> dict[str, object]:
    """Run a real PySpark Traffic Preprocessing & Profiling Pipeline (TPPP).

    The job reads synthetic traffic CSV data, performs Spark SQL transformations,
    creates window-based features, aggregates zone/weather congestion profiles,
    exports the most severe anomalies, and writes a curated Parquet dataset.
    """
    if not is_pyspark_available():
        raise RuntimeError("PySpark is not installed. Install it with: pip install pyspark")

    if not input_path.exists():
        save_traffic_data(DEFAULT_RECORDS, output_path=input_path)

    # Imports are intentionally local so the rest of the app can run without Spark.
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("QuantumTrafficTPPP")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    logs: list[str] = []
    try:
        raw_df = spark.read.csv(str(input_path), header=True, inferSchema=True)
        logs.append(f"Spark read {raw_df.count():,} traffic records from {input_path}")

        typed_df = (
            raw_df.withColumn("event_timestamp", F.to_timestamp("timestamp"))
            .withColumn("event_hour", F.hour("event_timestamp"))
            .withColumn(
                "congestion_weight",
                F.when(F.col("congestion_level") == "Critical", F.lit(4))
                .when(F.col("congestion_level") == "High", F.lit(3))
                .when(F.col("congestion_level") == "Moderate", F.lit(2))
                .otherwise(F.lit(1)),
            )
            .withColumn(
                "risk_index",
                F.round(
                    (F.col("traffic_density") / F.lit(100) * F.lit(0.32))
                    + ((F.lit(110) - F.col("average_speed")) / F.lit(110) * F.lit(0.22))
                    + (F.col("signal_wait_time") / F.lit(180) * F.lit(0.18))
                    + (F.col("accident_probability") * F.lit(0.18))
                    + (F.col("ai_anomaly_score") * F.lit(0.10)),
                    4,
                ),
            )
            .withColumn(
                "severe_event",
                (F.col("risk_index") >= F.lit(0.72))
                | (F.col("ai_anomaly_score") >= F.lit(0.82))
                | ((F.col("average_speed") <= F.lit(15)) & (F.col("traffic_density") >= F.lit(80))),
            )
        )
        logs.append("Spark created typed columns, congestion weights, risk_index, and severe_event flags")

        rolling_window = Window.partitionBy("city_zone").orderBy("event_timestamp").rowsBetween(-5, 0)
        curated_df = (
            typed_df.withColumn("rolling_density_6_event", F.round(F.avg("traffic_density").over(rolling_window), 2))
            .withColumn("speed_drop_from_zone_avg", F.round(F.avg("average_speed").over(rolling_window) - F.col("average_speed"), 2))
            .cache()
        )
        logs.append("Spark calculated rolling density and speed-drop features using window functions")

        summary_df = (
            curated_df.groupBy("city_zone", "weather_condition", "congestion_level")
            .agg(
                F.count("vehicle_id").alias("vehicle_events"),
                F.round(F.avg("traffic_density"), 2).alias("avg_density"),
                F.round(F.avg("average_speed"), 2).alias("avg_speed"),
                F.round(F.avg("risk_index"), 4).alias("avg_risk_index"),
                F.round(F.avg("ai_anomaly_score"), 4).alias("avg_ai_anomaly_score"),
                F.sum(F.col("severe_event").cast("int")).alias("severe_events"),
            )
            .withColumn("severe_event_rate", F.round(F.col("severe_events") / F.col("vehicle_events") * F.lit(100), 2))
            .orderBy(F.desc("severe_event_rate"), F.desc("avg_risk_index"))
        )

        top_anomalies_df = (
            curated_df.select(
                "event_timestamp",
                "vehicle_id",
                "city_zone",
                "weather_condition",
                "congestion_level",
                "traffic_density",
                "average_speed",
                "signal_wait_time",
                "accident_probability",
                "ai_anomaly_score",
                "risk_index",
                "rolling_density_6_event",
                "speed_drop_from_zone_avg",
            )
            .orderBy(F.desc("risk_index"), F.desc("ai_anomaly_score"))
            .limit(anomaly_limit)
        )

        summary_pdf = summary_df.toPandas()
        anomalies_pdf = top_anomalies_df.toPandas()
        summary_pdf.to_csv(summary_path, index=False)
        anomalies_pdf.to_csv(anomaly_path, index=False)
        logs.append(f"Spark wrote frontend summary CSV to {summary_path}")
        logs.append(f"Spark wrote top anomaly CSV to {anomaly_path}")

        curated_df.write.mode("overwrite").parquet(str(curated_output_dir))
        logs.append(f"Spark wrote curated Parquet dataset to {curated_output_dir}")

        return {
            "input_rows": int(raw_df.count()),
            "summary_rows": int(summary_pdf.shape[0]),
            "anomaly_rows": int(anomalies_pdf.shape[0]),
            "summary_path": str(summary_path),
            "anomaly_path": str(anomaly_path),
            "curated_output_dir": str(curated_output_dir),
            "logs": logs,
        }
    finally:
        spark.stop()


def load_tppp_summary(path: Path = TPPP_SUMMARY_FILE) -> pd.DataFrame:
    """Load the latest TPPP summary CSV for the Streamlit frontend."""
    return pd.read_csv(path)


def load_tppp_anomalies(path: Path = TPPP_ANOMALY_FILE) -> pd.DataFrame:
    """Load the latest TPPP top anomaly CSV for the Streamlit frontend."""
    return pd.read_csv(path)


if __name__ == "__main__":
    result = run_traffic_pyspark_pipeline()
    print("\n".join(result["logs"]))

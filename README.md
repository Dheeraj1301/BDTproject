# Quantum Traffic Flow Anomaly Analytics System

A lightweight, futuristic smart-city Big Data analytics demo built with Python, Streamlit, Pandas, NumPy, Faker, Matplotlib, and Plotly.

The system generates 100,000 synthetic traffic records and visualizes congestion patterns, accident risk, anomaly scores, zone analytics, weather impact, speed distribution, and a simple rolling traffic forecast. It also includes a **real PySpark Traffic Preprocessing & Profiling Pipeline (TPPP)** that starts a local SparkSession, transforms traffic telemetry with Spark SQL/window functions, writes curated Parquet, and exports useful CSV summaries. A separate ICU module still demonstrates Spark-style batch preprocessing concepts for large healthcare data.

## Project Structure

```text
.
├── app.py                    # Streamlit dashboard and frontend controls
├── analytics.py              # Reusable traffic analytics calculations
├── generate_data.py          # Synthetic traffic data generator
├── icu_batch_preprocessing.py # ICU generator + Spark-style batch simulation
├── pyspark_traffic_pipeline.py # Real PySpark traffic preprocessing pipeline
├── requirements.txt          # Python dependencies
└── data/                     # Generated CSV output folder
```

## Run Locally

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

The dashboard opens at Streamlit's default local URL, usually `http://localhost:8501`.

## Dashboard Modules

- **Traffic Command Center**: futuristic KPI cards, congestion trend forecast, heat analysis, hotspot ranking, weather impact, speed distribution, anomaly table, and zone analytics.
- **TPPP Real PySpark**: frontend controls that run a real PySpark job over `data/traffic_data.csv`, compute risk features, rolling-window metrics, zone/weather severe-event rates, top anomalies, and curated Parquet output.
- **ICU Batch Simulation**: frontend buttons to generate a large ICU dataset, run a Spark-style preprocessing job, review event logs, inspect alert-rate summaries, and download preprocessed ICU data.
- **Synthetic Data Hub**: a frontend home/hub for generated CSV files, download actions, dataset locations, Spark output paths, and demo-friendly data management.

## Real PySpark TPPP Outputs

The `pyspark_traffic_pipeline.py` module performs the actual PySpark work. From the frontend, open **TPPP Real PySpark** and click **Run TPPP Real PySpark Job**. The job creates:

- `data/pyspark_tppp/traffic_zone_weather_summary.csv` with zone/weather/congestion aggregates.
- `data/pyspark_tppp/traffic_top_anomalies.csv` with Spark-ranked high-risk traffic events.
- `data/pyspark_tppp/curated_parquet/` with the transformed Spark dataset including `risk_index`, `severe_event`, `rolling_density_6_event`, and `speed_drop_from_zone_avg`.

## Generated Traffic Dataset Columns

- `vehicle_id`
- `timestamp`
- `city_zone`
- `vehicle_type`
- `traffic_density`
- `average_speed`
- `weather_condition`
- `road_condition`
- `signal_wait_time`
- `accident_probability`
- `congestion_level`
- `ai_anomaly_score`

## Generated ICU Dataset Columns

- `patient_id`
- `admission_timestamp`
- `icu_unit`
- `age`
- `diagnosis`
- `heart_rate`
- `systolic_bp`
- `respiratory_rate`
- `oxygen_saturation`
- `temperature_c`
- `glucose_mg_dl`
- `lactate_mmol_l`
- `ventilation_status`
- `length_of_stay_hours`
- `mortality_risk`
- `critical_event`

The preprocessing simulation adds batch features such as `partition_id`, `admission_hour`, `shock_index`, `oxygen_gap`, `icu_severity_band`, and `batch_alert`.

## Viva Summary

This project simulates a futuristic smart-city traffic monitoring infrastructure using synthetic large-scale traffic datasets. It performs anomaly analytics, congestion forecasting, and accident-risk evaluation using statistical and AI-inspired metrics. The TPPP module demonstrates a real PySpark batch preprocessing pipeline by partitioning, transforming, windowing, aggregating, and exporting large traffic telemetry outputs. The ICU module demonstrates the same batch-processing concept for large healthcare data.

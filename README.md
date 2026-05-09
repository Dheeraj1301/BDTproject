# Quantum Traffic Flow Anomaly Analytics System

A lightweight, futuristic smart-city Big Data analytics demo built with Python, Streamlit, Pandas, NumPy, Faker, Matplotlib, and Plotly.

The system generates 100,000 synthetic traffic records and visualizes congestion patterns, accident risk, anomaly scores, zone analytics, weather impact, speed distribution, and a simple rolling traffic forecast. It also includes a **PySpark-style batch preprocessing simulation** for a large synthetic ICU dataset, so the project can demonstrate distributed-data concepts without requiring a real Spark cluster.

## Project Structure

```text
.
├── app.py                    # Streamlit dashboard and frontend controls
├── analytics.py              # Reusable traffic analytics calculations
├── generate_data.py          # Synthetic traffic data generator
├── icu_batch_preprocessing.py # ICU generator + PySpark-style batch simulation
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
- **ICU PySpark Batch Simulation**: frontend buttons to generate a large ICU dataset, run a simulated Spark batch preprocessing job, review event logs, inspect alert-rate summaries, and download preprocessed ICU data.
- **Synthetic Data Hub**: a frontend home/hub for generated CSV files, download actions, dataset locations, and demo-friendly data management.

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

This project simulates a futuristic smart-city traffic monitoring infrastructure using synthetic large-scale traffic datasets. It performs anomaly analytics, congestion forecasting, and accident-risk evaluation using statistical and AI-inspired metrics. The added ICU module demonstrates how a PySpark batch preprocessing pipeline works conceptually by partitioning and transforming a large healthcare dataset locally.

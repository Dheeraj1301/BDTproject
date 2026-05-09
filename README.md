# Quantum Traffic Flow Anomaly Analytics System

A lightweight, futuristic smart-city Big Data analytics demo built with Python, Streamlit, Pandas, NumPy, Faker, Matplotlib, and Plotly.

The system generates 100,000 synthetic traffic records and visualizes congestion patterns, accident risk, anomaly scores, zone analytics, weather impact, speed distribution, and a simple rolling traffic forecast.

## Project Structure

```text
.
├── app.py              # Streamlit dashboard
├── analytics.py        # Reusable analytics calculations
├── generate_data.py    # Synthetic traffic data generator
├── requirements.txt    # Python dependencies
└── data/               # Generated CSV output folder
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

## Generated Dataset Columns

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

## Viva Summary

This project simulates a futuristic smart-city traffic monitoring infrastructure using synthetic large-scale traffic datasets. It performs anomaly analytics, congestion forecasting, and accident-risk evaluation using statistical and AI-inspired metrics while staying beginner-friendly internally.

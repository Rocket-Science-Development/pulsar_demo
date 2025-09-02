# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a Pulsar.ML demo repository showcasing ML model monitoring and performance management. The architecture consists of:

- **Data collection**: Uses `pulsar-data-collection` to capture model predictions and metadata from inference
- **Metrics computation**: Leverages `pulsar-metrics` to compute drift and performance metrics on collected data
- **Storage**: InfluxDB for time-series data storage of predictions and computed metrics
- **Visualization**: Grafana dashboards for monitoring model performance
- **Automation**: Cron-based services running in Docker containers

## Key Components

### Core Python Modules
- `compute_metrics_script.py` - Main metrics computation service that queries InfluxDB and calculates metrics
- `fake_data_script.py` - Data generation service that creates synthetic data with drift simulation
- `gen_data_and_simulate_drift.py` - Core data generation and drift simulation functionality
- `training_script.py` - Model training utilities with CatBoost classifier
- `roi.py` - Return on Investment custom metric implementation

### Data Flow
1. Data captured via `pulsar-data-collection` and stored in InfluxDB `prediction` index
2. `compute-metrics` service (runs every 4 minutes via cron) queries latest data and computes metrics
3. Computed metrics stored in InfluxDB `metrics` index
4. Grafana displays metrics from `metrics` index in real-time dashboards

## Development Commands

### Docker Environment
```bash
# Set required environment variables
export GRAFANA_USERNAME=admin
export GRAFANA_PASSWORD=pass123
export DB_USER=admin
export DB_PASSWORD=pass123

# Start full environment
docker-compose up --build

# Access services
# Grafana: http://localhost:3000
# Chronograf: http://localhost:8889
# InfluxDB: http://localhost:8086
```

### Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### Running Scripts Locally
```bash
# Generate fake data with drift
python src/fake_data_script.py

# Compute metrics (requires InfluxDB connection)
python src/compute_metrics_script.py

# Train model
python src/app.py
```

## Configuration

### Cron Jobs (Container)
- Metrics computation: Every 4 minutes (`*/4 * * * *`)
- Data generation: Every 2 minutes (`*/2 * * * *`)

### Database Connection
- Host: `influxdb` (in Docker network) or `localhost:8086` (local)
- Default credentials: `admin/pass123`
- Prediction data index: `prediction`
- Metrics index: `metrics`

### Key Dependencies
- `pulsar-data-collection` - Data capture SDK
- `pulsar-metrics` - Metrics computation library  
- `catboost==1.2` - ML model framework
- `pandas==1.5.3` - Data manipulation
- `numpy==1.21.0` - Numerical computing
- `copulas` - Synthetic data generation

## Testing

No specific test framework is configured. To add tests, check if the codebase uses any testing patterns or frameworks in the existing code.

## Notes

- The system uses Pokemon dataset for classification demo (`class_Legendary_model.pkl`)
- Custom ROI metric implemented for business impact measurement
- Drift simulation supports configurable intensity levels (LOW, MODERATE, HIGH)
- All services are containerized and communicate via Docker network
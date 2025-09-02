# Pulsar.ML Local demo

This repository contains an example use-case demonstrating how to utilize [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection) and [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) to implement model monitoring and performance management.

## Quickstart

### Prerequisites
- Docker and Docker Compose installed

### Setup Instructions

1. **Configure Environment Variables (Optional)**:
   
   Copy the example environment file and customize if needed:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred credentials (optional)
   ```
   
   Or export environment variables manually:
   ```bash
   export GRAFANA_USERNAME=admin
   export GRAFANA_PASSWORD=pass123
   export DB_USER=admin
   export DB_PASSWORD=pass123
   ```

2. **Start the Services**:
   ```bash
   docker-compose up --build
   ```

3. **Access the Dashboard**:
   Open a web browser and go to http://localhost:3000
   
   Use the credentials:
   - Username: admin (or your custom GRAFANA_USERNAME)
   - Password: pass123 (or your custom GRAFANA_PASSWORD)

### Available Services
- **Grafana Dashboard**: http://localhost:3000
- **Chronograf**: http://localhost:8889
- **InfluxDB**: http://localhost:8086

### Environment Variables
All services use consistent environment variables defined in `.env.example`:
- `DB_USER` / `DB_PASSWORD`: InfluxDB credentials
- `GRAFANA_USERNAME` / `GRAFANA_PASSWORD`: Grafana admin credentials
- `DB_HOST` / `DB_PORT`: Database connection settings

## Pulsar.ML Architecture

![Architecture](images/PulsarDemoFlowDiagram_v1.jpeg)

## Demo Components

![Demo workflow](images/pulsar-demo-workflow.png)

Here is a description of the steps within the workflow

1. Data is captured from an inference container or a notebook using [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection)
2. Collected data point, predictions, and other relevant configured metadata are written into an index inside influxdb
3. `compute-metrics` service will query the latest entries from the database then leverage [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) in order to compute the different metrics.
4. All computed metrics are then written to another index in InfluxDB for display in Grafana.

## About [Pulsar.ML](https://pulsar.ml/)

Pulsar.ML is an application designed to assist with monitoring your machine learning models and gaining powerful insights into their performance.

We have released two open-source packages:

- [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection) :  A lightweight Python SDK for collecting features, predictions, and metadata from an ML model serving code or microservice.
- [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) : A library for evaluating and monitoring data and concept drift with an extensive set of metrics. It also provides the flexibility to use custom metrics defined by the user.

We also created [pulsar demo](https://github.com/Rocket-Science-Development/pulsar_demo) to showcase an example use-case that demonstrates how to leverage both packages for model monitoring and performance management.

For further interaction with our community, Please join our [slack channel](https://pulsarml.slack.com)

Powered by [Rocket Science Development](https://rocketscience.one/)

## Contributing

1. Fork this repository, develop, and test your changes
2. open an issue
3. Submit a pull request with a reference to the issue

# Architecture Overview

## System Architecture

The Pulsar.ML demo is a containerized ML monitoring system that demonstrates real-time model performance tracking and drift detection. The architecture follows a microservices pattern with clearly separated concerns for data collection, metrics computation, storage, and visualization.

![Architecture](../../images/PulsarDemoFlowDiagram_v1.jpeg)

## Component Architecture

### Core Services

#### 1. InfluxDB (Time-Series Database)
**Purpose**: Primary data storage for predictions and computed metrics
**Technology**: InfluxDB 1.x
**Port**: 8086
**Databases**:
- `prediction` - Stores raw model predictions and metadata
- `metrics` - Stores computed drift and performance metrics

**Key Features**:
- Time-series optimized storage
- High write throughput for real-time data ingestion
- SQL-like query language for data retrieval
- Built-in data retention policies

#### 2. Grafana (Visualization Layer)
**Purpose**: Real-time dashboards and alerting
**Technology**: Grafana Latest
**Port**: 3000
**Configuration**:
- Pre-provisioned dashboards from `dashboards/` directory
- InfluxDB data source auto-configured
- Custom dashboard for ML metrics visualization

**Key Features**:
- Real-time metric visualization
- Configurable alerts and notifications
- Custom dashboard templates
- User management and access control

#### 3. Chronograf (Database Management UI)
**Purpose**: InfluxDB administration and query interface
**Technology**: Chronograf Latest
**Port**: 8889
**Configuration**:
- Connected to InfluxDB instance
- Provides query builder and data exploration tools

#### 4. Fake Data Generator Service
**Purpose**: Simulates real-world ML inference data with configurable drift
**Technology**: Python 3.9+ with custom drift simulation
**Schedule**: Every 2 minutes via cron
**Key Components**:
- `fake_data_script.py` - Main execution script
- `gen_data_and_simulate_drift.py` - Drift simulation engine
- `training_script.py` - Model utilities

**Data Generation Process**:
1. Loads reference dataset (`bank_num.csv`)
2. Generates synthetic data using copulas
3. Applies configurable drift patterns (LOW, MODERATE, HIGH)
4. Simulates model predictions using pre-trained classifier
5. Captures data using `pulsar-data-collection` SDK

#### 5. Metrics Computation Service
**Purpose**: Calculates drift and performance metrics from collected data
**Technology**: Python 3.9+ with pulsar-metrics library
**Schedule**: Every 4 minutes via cron
**Key Components**:
- `compute_metrics_script.py` - Main metrics computation
- `roi.py` - Custom ROI metric implementation

**Metrics Computation Process**:
1. Queries latest predictions from InfluxDB
2. Compares current data against reference dataset
3. Computes various drift metrics using `pulsar-metrics`
4. Calculates custom business metrics (ROI)
5. Stores results in metrics database

## Data Flow Architecture

### 1. Data Collection Flow
```
Model Inference → pulsar-data-collection → InfluxDB (prediction index)
```

**Data Points Captured**:
- Feature values
- Model predictions
- Prediction probabilities
- Metadata (timestamp, model_id, model_version)
- Business context data

### 2. Metrics Computation Flow
```
InfluxDB (prediction) → Metrics Service → pulsar-metrics → InfluxDB (metrics)
```

**Metrics Computed**:
- **Drift Metrics**: Data drift, concept drift, population stability
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Custom Metrics**: ROI calculations, business impact metrics
- **Statistical Metrics**: Distribution comparisons, feature importance changes

### 3. Visualization Flow
```
InfluxDB (metrics) → Grafana → Real-time Dashboards
```

**Dashboard Components**:
- Time-series metric trends
- Drift detection alerts
- Performance degradation indicators
- Custom business metric visualizations

## Technology Stack

### Core Technologies
- **Container Orchestration**: Docker Compose
- **Database**: InfluxDB 1.x (Time-series database)
- **Visualization**: Grafana (Dashboard and alerting)
- **Language**: Python 3.9+
- **ML Framework**: CatBoost for classification
- **Task Scheduling**: Cron (within containers)

### Key Dependencies
- **pulsar-data-collection**: Data capture SDK
- **pulsar-metrics**: Metrics computation library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **copulas**: Synthetic data generation
- **catboost**: ML model framework

## Network Architecture

### Docker Network Configuration
- **Network Name**: `pulsar_demo_default`
- **Type**: Bridge network
- **DNS Resolution**: Automatic container name resolution

### Service Communication
```
fake-data-generator ─┐
                    ├─→ InfluxDB ←─── compute-metrics
Chronograf ──────────┘              │
                                    └─→ Grafana
```

### Port Mapping
- **3000**: Grafana Web UI
- **8086**: InfluxDB API
- **8889**: Chronograf UI
- **Internal**: Inter-service communication via Docker network

## Security Architecture

### Authentication & Authorization
- **InfluxDB**: Username/password authentication
- **Grafana**: Admin user with configurable credentials
- **Network**: Isolated Docker network for internal communication

### Data Security
- **Environment Variables**: Sensitive credentials stored in .env file
- **Container Isolation**: Services isolated in separate containers
- **Network Segmentation**: Internal communication only between required services

## Scalability Considerations

### Horizontal Scaling
- **Data Generation**: Can run multiple generator instances
- **Metrics Computation**: Supports distributed metric calculation
- **Database**: InfluxDB clustering available for enterprise deployments

### Vertical Scaling
- **Memory**: Configurable memory limits per service
- **CPU**: Adjustable CPU allocation for compute-intensive tasks
- **Storage**: Persistent volumes for data retention

## Monitoring and Observability

### System Monitoring
- **Container Health**: Docker Compose health checks
- **Service Logs**: Centralized logging via Docker
- **Metrics**: Self-monitoring capabilities via Grafana

### Application Monitoring
- **Data Quality**: Automated drift detection alerts
- **Performance**: Real-time metric computation and visualization
- **Business Impact**: Custom ROI and business metric tracking

## Integration Points

### External Integrations
- **Model Serving**: Compatible with any ML inference service
- **Alert Systems**: Grafana alerting to external notification systems
- **Data Sources**: Configurable data input from various sources

### API Interfaces
- **InfluxDB API**: RESTful interface for data operations
- **Grafana API**: Dashboard and configuration management
- **Custom Metrics**: Extensible metric computation framework

## Configuration Management

### Environment Configuration
- **Docker Compose**: Service orchestration and configuration
- **Environment Variables**: Runtime configuration via .env file
- **Volume Mounts**: Persistent configuration and data storage

### Customization Points
- **Dashboards**: JSON-based dashboard configuration
- **Metrics**: Custom metric implementations
- **Data Sources**: Configurable reference datasets
- **Drift Simulation**: Adjustable drift patterns and intensity
# Development Guidelines

## Overview

This document provides comprehensive guidelines for developing and customizing the Pulsar.ML demo system. It covers code structure, development workflow, and best practices for extending the system.

## Code Structure

### Project Organization

```
pulsar_demo/
├── src/                          # Python source code
│   ├── __init__.py
│   ├── app.py                    # Main application entry point
│   ├── compute_metrics_script.py # Metrics computation service
│   ├── fake_data_script.py       # Data generation service
│   ├── gen_data_and_simulate_drift.py # Drift simulation engine
│   ├── training_script.py        # Model training utilities
│   ├── model_training_script.py  # Extended model training
│   └── roi.py                    # Custom ROI metric implementation
├── datasets/                     # Reference datasets
├── models/                       # Trained model artifacts
├── dashboards/                   # Grafana dashboard configurations
├── grafana-provisioning/         # Grafana setup configuration
├── docker/                       # Docker-specific configurations
├── docs/                         # Documentation
├── tests/                        # Test files (when implemented)
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Poetry configuration
├── docker-compose.yaml          # Service orchestration
└── .env.example                 # Environment template
```

### Core Modules

#### Data Generation (`fake_data_script.py`)
**Purpose**: Generates synthetic data with configurable drift patterns

**Key Classes**:
- `GenerateFakeData`: Main data generation controller
- `DriftSimulator`: Handles drift pattern application
- `Classifier`: Model inference wrapper

**Configuration**:
```python
SAMPLE_SIZE = 1000           # Number of samples to generate
pokemon_test_data = '/app/datasets/bank_num.csv'  # Reference dataset
target = 'default'           # Target variable name
```

#### Metrics Computation (`compute_metrics_script.py`)
**Purpose**: Calculates ML monitoring metrics from collected data

**Key Components**:
- Reference data loading and preparation
- Database connection and data retrieval
- Metric computation using pulsar-metrics
- Custom metric implementations (ROI)

**Metric Configuration**:
```python
FPR_THRESHOLD = 0.5     # False Positive Rate threshold
FNR_THRESHOLD = 0.2     # False Negative Rate threshold
cost_per_FP = 100       # Cost per False Positive
cost_per_FN = 300       # Cost per False Negative
```

#### Drift Simulation (`gen_data_and_simulate_drift.py`)
**Purpose**: Core engine for data drift simulation

**Drift Intensity Levels**:
```python
class DriftIntensity:
    LOW = 0.1      # Subtle drift
    MODERATE = 0.3 # Moderate drift
    HIGH = 0.5     # Significant drift
```

## Development Environment Setup

### Local Development

#### Prerequisites
```bash
# Python 3.9+
python3 --version

# Poetry (recommended) or pip
curl -sSL https://install.python-poetry.org | python3 -
```

#### Setup Steps
```bash
# Clone repository
git clone https://github.com/Rocket-Science-Development/pulsar_demo.git
cd pulsar_demo

# Install dependencies using Poetry
poetry install
poetry shell

# Or using pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

#### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.sortImports.args": ["--profile", "black"]
}
```

**PyCharm Configuration**:
- Set interpreter to project virtual environment
- Enable code formatting with Black
- Configure import optimization with isort
- Enable type checking with mypy

### Development Workflow

#### Branch Strategy
```bash
# Feature development
git checkout -b feature/new-metric-implementation
git checkout -b fix/data-generation-bug
git checkout -b docs/api-documentation

# Naming convention
feature/description-of-feature
fix/description-of-fix
docs/description-of-documentation
```

#### Code Quality Tools
```bash
# Install development tools
pip install black isort flake8 mypy pytest

# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/

# Run tests (when implemented)
pytest tests/
```

#### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
-   repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    -   id: black
        language_version: python3.9
-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: ["--profile", "black"]
-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
EOF

# Install hooks
pre-commit install
```

## Customization Guides

### Adding New Metrics

#### 1. Custom Metric Implementation

Create a new metric in `src/custom_metrics.py`:
```python
from pulsar_metrics.metrics.base import CustomMetric
import numpy as np

@CustomMetric
def my_custom_metric(current, reference, **kwargs):
    """
    Custom metric implementation.
    
    Args:
        current: Current dataset
        reference: Reference dataset
        **kwargs: Additional parameters
    
    Returns:
        float: Computed metric value
    """
    # Your metric calculation logic
    metric_value = calculate_my_metric(current, reference)
    return metric_value

def calculate_my_metric(current, reference):
    """Helper function for metric calculation."""
    # Implementation details
    return np.mean(current) - np.mean(reference)
```

#### 2. Register Metric in Computation Script

Add to `compute_metrics_script.py`:
```python
from custom_metrics import my_custom_metric

# In the metrics computation section
analyzer.add_custom_metric('my_custom_metric', my_custom_metric)
```

#### 3. Update Grafana Dashboard

Create a new panel in Grafana:
```json
{
  "title": "My Custom Metric",
  "type": "stat",
  "targets": [
    {
      "query": "SELECT mean(\"metric_value\") FROM \"computed_metrics\" WHERE \"metric_name\" = 'my_custom_metric' AND time >= now() - 24h GROUP BY time(1h)"
    }
  ]
}
```

### Customizing Drift Simulation

#### 1. Add New Drift Patterns

Extend `gen_data_and_simulate_drift.py`:
```python
class CustomDriftSimulator(DriftSimulator):
    def apply_seasonal_drift(self, data, intensity=0.3):
        """Apply seasonal drift pattern."""
        import datetime
        
        # Get current month
        current_month = datetime.datetime.now().month
        
        # Apply seasonal adjustment
        seasonal_factor = 1 + (intensity * np.sin(2 * np.pi * current_month / 12))
        
        # Modify numerical features
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] *= seasonal_factor
        
        return data
    
    def apply_gradual_drift(self, data, intensity=0.1, days_since_reference=0):
        """Apply gradual drift over time."""
        drift_factor = 1 + (intensity * days_since_reference / 365)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] *= drift_factor
        
        return data
```

#### 2. Configure Drift Patterns

Update `fake_data_script.py`:
```python
# Add drift configuration
DRIFT_CONFIG = {
    'pattern': 'seasonal',  # or 'gradual', 'sudden'
    'intensity': DriftIntensity.MODERATE,
    'features_affected': ['age', 'balance', 'duration'],
    'drift_probability': 0.8  # Probability of applying drift
}

# Apply custom drift
if random.random() < DRIFT_CONFIG['drift_probability']:
    drift_simulator = CustomDriftSimulator()
    if DRIFT_CONFIG['pattern'] == 'seasonal':
        sampled_data = drift_simulator.apply_seasonal_drift(
            sampled_data, DRIFT_CONFIG['intensity']
        )
```

### Customizing Dashboards

#### 1. Dashboard Structure

Grafana dashboards are stored in `dashboards/` directory as JSON files:
```
dashboards/
├── ml-monitoring-overview.json    # Main dashboard
├── drift-detection.json          # Drift-specific metrics
└── custom-metrics.json           # Custom business metrics
```

#### 2. Creating Custom Dashboards

**Basic Dashboard Template**:
```json
{
  "dashboard": {
    "id": null,
    "title": "Custom ML Monitoring",
    "tags": ["pulsar", "ml-monitoring"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-24h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Model Accuracy Trend",
        "type": "graph",
        "targets": [
          {
            "query": "SELECT mean(\"metric_value\") FROM \"computed_metrics\" WHERE \"metric_name\" = 'accuracy' GROUP BY time(1h)"
          }
        ]
      }
    ]
  }
}
```

#### 3. Advanced Panel Configurations

**Threshold Alerts**:
```json
{
  "alert": {
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A"
        },
        "reducer": {
          "params": [],
          "type": "last"
        },
        "evaluator": {
          "params": [0.8],
          "type": "lt"
        }
      }
    ],
    "executionErrorState": "alerting",
    "for": "5m",
    "frequency": "10s",
    "handler": 1,
    "name": "Model Accuracy Alert",
    "noDataState": "no_data",
    "notifications": []
  }
}
```

**Custom Variables**:
```json
{
  "templating": {
    "list": [
      {
        "name": "model_id",
        "type": "query",
        "query": "SHOW TAG VALUES FROM \"computed_metrics\" WITH KEY = \"model_id\"",
        "refresh": 1,
        "includeAll": false,
        "multi": false
      }
    ]
  }
}
```

### Adding New Data Sources

#### 1. Database Configuration

Add new data source to `fake_data_script.py`:
```python
class CustomDataSource:
    def __init__(self, config):
        self.config = config
    
    def load_data(self):
        """Load data from custom source."""
        if self.config['type'] == 'api':
            return self.load_from_api()
        elif self.config['type'] == 'database':
            return self.load_from_database()
        else:
            raise ValueError(f"Unsupported data source type: {self.config['type']}")
    
    def load_from_api(self):
        """Load data from REST API."""
        import requests
        response = requests.get(self.config['url'])
        return pd.DataFrame(response.json())
    
    def load_from_database(self):
        """Load data from external database."""
        import sqlalchemy
        engine = sqlalchemy.create_engine(self.config['connection_string'])
        return pd.read_sql(self.config['query'], engine)
```

#### 2. Integration

Update data loading logic:
```python
# Configure custom data source
DATA_SOURCE_CONFIG = {
    'type': 'api',
    'url': 'https://api.example.com/data',
    'refresh_interval': 300  # seconds
}

# Use custom data source
if os.getenv('USE_CUSTOM_DATA_SOURCE', 'false').lower() == 'true':
    data_source = CustomDataSource(DATA_SOURCE_CONFIG)
    reference_data = data_source.load_data()
else:
    reference_data = pd.read_csv('/app/datasets/bank_num.csv')
```

### Model Integration

#### 1. Custom Model Support

Create model wrapper in `src/model_wrapper.py`:
```python
import pickle
import numpy as np
from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    @abstractmethod
    def predict(self, data):
        pass
    
    @abstractmethod
    def predict_proba(self, data):
        pass

class PickleModelWrapper(ModelWrapper):
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, data):
        return self.model.predict(data)
    
    def predict_proba(self, data):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(data)
        else:
            # Fallback for models without probability prediction
            predictions = self.predict(data)
            return np.column_stack([1 - predictions, predictions])

class ONNXModelWrapper(ModelWrapper):
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, data):
        result = self.session.run(None, {self.input_name: data.values})
        return result[0]
    
    def predict_proba(self, data):
        result = self.session.run(None, {self.input_name: data.values})
        return result[1] if len(result) > 1 else self._convert_to_proba(result[0])
```

#### 2. Model Registration

Update `training_script.py`:
```python
from model_wrapper import PickleModelWrapper, ONNXModelWrapper

class ModelRegistry:
    def __init__(self):
        self.models = {}
    
    def register_model(self, name, model_wrapper):
        self.models[name] = model_wrapper
    
    def get_model(self, name):
        return self.models.get(name)
    
    def list_models(self):
        return list(self.models.keys())

# Global model registry
model_registry = ModelRegistry()

# Register models
model_registry.register_model(
    'catboost_classifier', 
    PickleModelWrapper('/app/models/class_Legendary_model.pkl')
)

model_registry.register_model(
    'onnx_classifier',
    ONNXModelWrapper('/app/models/model.onnx')
)
```

## Testing Guidelines

### Test Structure

Create `tests/` directory structure:
```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
├── unit/                    # Unit tests
│   ├── test_drift_simulation.py
│   ├── test_metrics_computation.py
│   └── test_data_generation.py
├── integration/             # Integration tests
│   ├── test_database_integration.py
│   └── test_end_to_end.py
└── fixtures/                # Test data
    ├── sample_data.csv
    └── test_model.pkl
```

### Unit Testing

**Example Test** (`tests/unit/test_drift_simulation.py`):
```python
import pytest
import pandas as pd
import numpy as np
from src.gen_data_and_simulate_drift import DriftSimulator, DriftIntensity

class TestDriftSimulator:
    def setup_method(self):
        """Setup test data."""
        self.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.uniform(0, 10, 100),
            'target': np.random.choice([0, 1], 100)
        })
        self.simulator = DriftSimulator()
    
    def test_apply_drift_changes_data(self):
        """Test that drift simulation modifies data."""
        original_data = self.test_data.copy()
        drifted_data = self.simulator.apply_drift(
            self.test_data, 
            intensity=DriftIntensity.MODERATE
        )
        
        # Data should be modified
        assert not original_data.equals(drifted_data)
        
        # Shape should remain the same
        assert original_data.shape == drifted_data.shape
    
    def test_drift_intensity_affects_change_magnitude(self):
        """Test that higher intensity creates more drift."""
        low_drift = self.simulator.apply_drift(
            self.test_data.copy(), 
            intensity=DriftIntensity.LOW
        )
        high_drift = self.simulator.apply_drift(
            self.test_data.copy(), 
            intensity=DriftIntensity.HIGH
        )
        
        # Calculate differences from original
        low_diff = np.abs(low_drift['feature1'] - self.test_data['feature1']).mean()
        high_diff = np.abs(high_drift['feature1'] - self.test_data['feature1']).mean()
        
        # High intensity should create more drift
        assert high_diff > low_diff

@pytest.fixture
def sample_reference_data():
    """Fixture providing sample reference data."""
    return pd.DataFrame({
        'age': np.random.normal(35, 10, 1000),
        'balance': np.random.lognormal(8, 1, 1000),
        'default': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })
```

### Integration Testing

**Database Integration Test** (`tests/integration/test_database_integration.py`):
```python
import pytest
import pandas as pd
from src.compute_metrics_script import DatabaseLogin, DataCapture

class TestDatabaseIntegration:
    @pytest.fixture
    def database_login(self):
        """Fixture for test database connection."""
        return DatabaseLogin(
            db_host="localhost",
            db_port=8086,
            db_user="test_user",
            db_password="test_password",
            protocol="line"
        )
    
    def test_data_capture_connection(self, database_login):
        """Test database connection establishment."""
        data_capture = DataCapture(
            storage_engine="influxdb",
            operation_type="PREDICTION",
            login_url=database_login
        )
        
        # Should not raise exception
        assert data_capture is not None
    
    @pytest.mark.integration
    def test_end_to_end_data_flow(self, database_login, sample_reference_data):
        """Test complete data flow from generation to storage."""
        # This test requires running InfluxDB instance
        # Skip if database is not available
        try:
            data_capture = DataCapture(
                storage_engine="influxdb",
                operation_type="PREDICTION",
                login_url=database_login
            )
            
            # Test data capture
            # ... implementation details
            
        except ConnectionError:
            pytest.skip("Database not available for integration test")
```

### Performance Testing

**Load Testing Script** (`tests/performance/load_test.py`):
```python
import time
import concurrent.futures
from src.fake_data_script import GenerateFakeData

def test_data_generation_performance():
    """Test data generation performance under load."""
    sample_sizes = [100, 500, 1000, 5000]
    
    for size in sample_sizes:
        start_time = time.time()
        
        generator = GenerateFakeData(
            path_ref_data='/app/datasets/bank_num.csv',
            sample_size=size,
            target='default'
        )
        
        data = generator.get_dataclass_sampling()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Sample size {size}: {processing_time:.2f}s")
        
        # Performance assertion
        assert processing_time < size * 0.01  # Should process 100 samples per second

def test_concurrent_data_generation():
    """Test concurrent data generation."""
    def generate_data():
        generator = GenerateFakeData(
            path_ref_data='/app/datasets/bank_num.csv',
            sample_size=1000,
            target='default'
        )
        return generator.get_dataclass_sampling()
    
    # Test concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_data) for _ in range(4)]
        results = [future.result() for future in futures]
    
    # All should complete successfully
    assert len(results) == 4
    for result in results:
        assert result is not None
        assert len(result) > 0
```

## Documentation Standards

### Code Documentation

**Function Documentation**:
```python
def compute_custom_metric(current_data: pd.DataFrame, 
                         reference_data: pd.DataFrame,
                         threshold: float = 0.05) -> float:
    """
    Compute custom business metric comparing current and reference data.
    
    Args:
        current_data: Current period data for comparison
        reference_data: Baseline reference data
        threshold: Significance threshold for metric calculation
        
    Returns:
        float: Computed metric value between 0 and 1
        
    Raises:
        ValueError: If data is empty or incompatible
        
    Example:
        >>> current = pd.DataFrame({'feature': [1, 2, 3]})
        >>> reference = pd.DataFrame({'feature': [2, 3, 4]})
        >>> metric = compute_custom_metric(current, reference, 0.1)
        >>> print(f"Metric value: {metric}")
    """
    if current_data.empty or reference_data.empty:
        raise ValueError("Input data cannot be empty")
    
    # Metric calculation logic
    metric_value = calculate_difference(current_data, reference_data)
    return metric_value
```

**Class Documentation**:
```python
class CustomDriftDetector:
    """
    Advanced drift detection with configurable algorithms.
    
    This class implements multiple drift detection algorithms and provides
    a unified interface for drift monitoring in ML models.
    
    Attributes:
        algorithm (str): Selected drift detection algorithm
        sensitivity (float): Detection sensitivity level
        window_size (int): Rolling window size for detection
        
    Example:
        >>> detector = CustomDriftDetector(algorithm='ks_test', sensitivity=0.05)
        >>> drift_detected = detector.detect_drift(current_data, reference_data)
        >>> if drift_detected:
        ...     print("Drift detected!")
    """
    
    def __init__(self, algorithm: str = 'ks_test', 
                 sensitivity: float = 0.05,
                 window_size: int = 1000):
        """
        Initialize drift detector with specified configuration.
        
        Args:
            algorithm: Drift detection algorithm ('ks_test', 'psi', 'chi_square')
            sensitivity: P-value threshold for significance testing
            window_size: Number of samples in rolling window
        """
        self.algorithm = algorithm
        self.sensitivity = sensitivity
        self.window_size = window_size
```

This comprehensive development guide provides all the necessary information for developers to understand, modify, and extend the Pulsar.ML demo system effectively.
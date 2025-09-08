# Data Collection API Documentation

## Overview

The Pulsar.ML demo uses the `pulsar-data-collection` library to capture model predictions and metadata. This document describes the API interfaces and data structures used for data collection.

## Data Collection Components

### DatabaseLogin Class

**Purpose**: Manages database connection credentials and configuration.

```python
from pulsar_data_collection.data_capture import DatabaseLogin

database_login = DatabaseLogin(
    db_host=os.getenv("DB_HOST", "influxdb"), 
    db_port=int(os.getenv("DB_PORT", "8086")), 
    db_user=os.getenv("DB_USER", "admin"), 
    db_password=os.getenv("DB_PASSWORD", "pass123"), 
    protocol="line"
)
```

**Parameters**:
- `db_host` (str): Database hostname or IP address
- `db_port` (int): Database port number
- `db_user` (str): Database username
- `db_password` (str): Database password
- `protocol` (str): Communication protocol ("line" for InfluxDB line protocol)

### DataCapture Class

**Purpose**: Main interface for data collection operations.

```python
from pulsar_data_collection.data_capture import DataCapture

data_capture = DataCapture(
    storage_engine="influxdb",
    operation_type="PREDICTION",  # or "METRICS"
    login_url=database_login
)
```

**Parameters**:
- `storage_engine` (str): Target storage system ("influxdb")
- `operation_type` (str): Type of operation ("PREDICTION" or "METRICS")
- `login_url` (DatabaseLogin): Database connection configuration

### DataWithPrediction Class

**Purpose**: Data structure for model predictions with metadata.

```python
from pulsar_data_collection.data_capture import DataWithPrediction

data_point = DataWithPrediction(
    data=feature_data,           # pandas DataFrame with features
    prediction=prediction_value, # Model prediction
    model_id="model_v1",        # Model identifier
    model_version="1.0",        # Model version
    pred_timestamp=datetime.now(), # Prediction timestamp
    period="current"            # Data period identifier
)
```

**Attributes**:
- `data` (pandas.DataFrame): Feature data for the prediction
- `prediction`: Model prediction value
- `model_id` (str): Unique model identifier
- `model_version` (str): Model version string
- `pred_timestamp` (datetime): When the prediction was made
- `period` (str): Data period classification ("current", "reference")

## Data Collection Operations

### Capturing Prediction Data

**Example Implementation** (from `fake_data_script.py`):

```python
# Initialize data capture for predictions
data_capture = DataCapture(
    storage_engine="influxdb",
    operation_type="PREDICTION",
    login_url=database_login
)

# Create prediction data point
for index, row in sampled_data.iterrows():
    prediction_data = DataWithPrediction(
        data=row.to_frame().T,           # Feature data
        prediction=prediction[index],     # Model prediction
        model_id='1',                    # Model ID
        model_version='2',               # Version
        pred_timestamp=datetime.now(),   # Current timestamp
        period='current'                 # Period classification
    )
    
    # Send to database
    data_capture.capture(prediction_data)
```

### Retrieving Evaluation Timestamps

**Purpose**: Get the last evaluation timestamp for incremental processing.

```python
# Get last processed timestamp
last_eval_timestamp = data_capture.collect_eval_timestamp()

if last_eval_timestamp:
    print(f"Last evaluation: {last_eval_timestamp}")
else:
    print("No previous evaluations found")
```

### Querying Latest Data

**Purpose**: Retrieve recent prediction data for metrics computation.

```python
# Set time window for data retrieval
data_capture.set_data_timewindow(start_time, end_time)

# Get data for metrics computation
current_data = data_capture.collect_data_for_metrics(
    model_id='1',
    model_version='2'
)
```

## Database Schema

### Prediction Index Structure

**Database**: `prediction`
**Measurement**: `pokemon_predictions` (or model-specific name)

**Fields**:
- `model_id` (tag): Model identifier
- `model_version` (tag): Model version
- `period` (tag): Data period ("current", "reference")
- `prediction` (field): Model prediction value
- `feature_*` (fields): Individual feature values
- `timestamp` (time): Prediction timestamp

**Example InfluxDB Line Protocol**:
```
pokemon_predictions,model_id=1,model_version=2,period=current prediction=0.85,feature_age=25,feature_balance=1000 1634567890000000000
```

### Metrics Index Structure

**Database**: `metrics`
**Measurement**: `computed_metrics`

**Fields**:
- `model_id` (tag): Model identifier
- `model_version` (tag): Model version
- `metric_name` (tag): Name of the metric
- `metric_value` (field): Computed metric value
- `timestamp` (time): Computation timestamp

## Configuration Options

### Environment Variables

**Required Variables**:
```bash
DB_HOST=influxdb          # Database hostname
DB_PORT=8086              # Database port
DB_USER=admin             # Database username
DB_PASSWORD=pass123       # Database password
```

**Optional Variables**:
```bash
INFLUXDB_DATABASE=prediction  # Default database name
RETENTION_POLICY=autogen      # Data retention policy
```

### Connection Parameters

**Timeout Settings**:
- Default connection timeout: 30 seconds
- Default read timeout: 60 seconds
- Retry attempts: 3

**Batch Configuration**:
- Default batch size: 100 records
- Flush interval: 10 seconds
- Maximum buffer size: 1000 records

## Error Handling

### Common Exceptions

```python
from pulsar_data_collection.exceptions import (
    ConnectionError,
    DataValidationError,
    StorageError
)

try:
    data_capture.capture(prediction_data)
except ConnectionError as e:
    print(f"Database connection failed: {e}")
except DataValidationError as e:
    print(f"Invalid data format: {e}")
except StorageError as e:
    print(f"Storage operation failed: {e}")
```

### Retry Logic

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def capture_with_retry(data_capture, prediction_data):
    return data_capture.capture(prediction_data)
```

## Performance Considerations

### Batch Operations

```python
# Collect multiple predictions
predictions_batch = []
for prediction_data in prediction_list:
    predictions_batch.append(prediction_data)
    
    # Process in batches of 100
    if len(predictions_batch) >= 100:
        data_capture.capture_batch(predictions_batch)
        predictions_batch.clear()

# Process remaining items
if predictions_batch:
    data_capture.capture_batch(predictions_batch)
```

### Asynchronous Operations

```python
import asyncio
from pulsar_data_collection.async_capture import AsyncDataCapture

async def async_capture_example():
    async_capture = AsyncDataCapture(
        storage_engine="influxdb",
        operation_type="PREDICTION",
        login_url=database_login
    )
    
    tasks = []
    for prediction_data in prediction_list:
        task = async_capture.capture(prediction_data)
        tasks.append(task)
    
    # Wait for all captures to complete
    await asyncio.gather(*tasks)
```

## Data Validation

### Required Fields

All prediction data must include:
- Valid feature data (pandas DataFrame)
- Prediction value (numeric or categorical)
- Model ID and version
- Timestamp information

### Validation Rules

```python
def validate_prediction_data(data_point):
    """Validate prediction data before capture."""
    
    # Check required fields
    required_fields = ['data', 'prediction', 'model_id', 'model_version']
    for field in required_fields:
        if not hasattr(data_point, field) or getattr(data_point, field) is None:
            raise DataValidationError(f"Missing required field: {field}")
    
    # Validate data types
    if not isinstance(data_point.data, pd.DataFrame):
        raise DataValidationError("Data must be a pandas DataFrame")
    
    # Validate timestamp
    if data_point.pred_timestamp is None:
        data_point.pred_timestamp = datetime.now()
    
    return True
```

## Integration Examples

### Real-time Model Serving

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    input_data = pd.DataFrame(request.json)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Capture prediction data
    prediction_data = DataWithPrediction(
        data=input_data,
        prediction=prediction[0],
        model_id=request.headers.get('model_id', 'default'),
        model_version=request.headers.get('model_version', '1.0'),
        pred_timestamp=datetime.now(),
        period='current'
    )
    
    # Async capture to avoid blocking response
    asyncio.create_task(
        async_data_capture.capture(prediction_data)
    )
    
    return jsonify({'prediction': prediction[0]})
```

### Batch Processing Integration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

def batch_prediction_capture(**context):
    """Airflow task for batch prediction capture."""
    
    # Load batch data
    batch_data = load_batch_data(context['ds'])
    
    # Make predictions
    predictions = model.predict(batch_data)
    
    # Capture all predictions
    data_capture = DataCapture(
        storage_engine="influxdb",
        operation_type="PREDICTION",
        login_url=database_login
    )
    
    prediction_list = []
    for idx, (_, row) in enumerate(batch_data.iterrows()):
        prediction_data = DataWithPrediction(
            data=row.to_frame().T,
            prediction=predictions[idx],
            model_id='batch_model',
            model_version='1.0',
            pred_timestamp=datetime.now(),
            period='current'
        )
        prediction_list.append(prediction_data)
    
    data_capture.capture_batch(prediction_list)

dag = DAG('ml_batch_prediction', schedule_interval='@daily')
capture_task = PythonOperator(
    task_id='capture_predictions',
    python_callable=batch_prediction_capture,
    dag=dag
)
```
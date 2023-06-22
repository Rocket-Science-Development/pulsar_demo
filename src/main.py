import pickle as pkl
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
# from influxdb_client import InfluxDBClient
# !pip install pulsar_data_collection-0.0.0-py3-none-any.whl --force-reinstall
from pulsar_data_collection.config import factories
from pulsar_data_collection.models import DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.database_connectors.influxdb import Influxdb
from gen_data_and_simulate_drift import DriftIntensity, DriftSimulator, GenerateFakeData
from training_script import Classifier
import sys
sys.path.append('../src/')
from datetime import timezone, datetime
from io import BytesIO

app = FastAPI()

# Load environment variables from abc.env
# env_vars = dotenv_values('../env_files/influxv2.env')

# Set up InfluxDB connection
# token = env_vars['DOCKER_INFLUXDB_INIT_ADMIN_TOKEN']
# org = env_vars['DOCKER_INFLUXDB_INIT_ORG']
# bucket_name = env_vars['DOCKER_INFLUXDB_INIT_BUCKET']
token = 'mytoken'
org = 'pulsarml'
bucket_name = 'demo'
url = 'http://influxdb:8086'

@app.get("/")
def index():
    return {"title": "Hello world"}

@app.post("/predict")
async def predict_pokemon(file: UploadFile = File(...)):
    content = await file.read()
    df_user_input = pd.read_csv(BytesIO(content))
    # Convert the provided file data to a pandas DataFrame

    target = 'Legendary'
    genertor_fake_data = GenerateFakeData(path_ref_data=df_user_input, sample_size=len(df_user_input), target=target)
    sampled_data = genertor_fake_data.get_dataclass_sampling()

    # if the task is classification
    pok_classifier = Classifier(df_train=sampled_data.train_data,
                                num_features=sampled_data.list_num_col,
                                cat_features=None,
                                target=target,
                                pkl_file_path=f'class_{target}_model.pkl')
    pok_classifier.train()
    pok_classifier.serialize()

    drift_sim_info = DriftSimulator(sampled_data, nb_cols_to_drift=1, drift_intensity=DriftIntensity.MODERATE)
    df_test_drifted = drift_sim_info.get_test_data_drifted()

    prediction = pok_classifier.predict(df_test=df_test_drifted)
    prediction_int = [1 if e == 'True' else 0 for e in prediction]

    df_test_drifted = drift_sim_info.get_test_data_drifted()
    df_test_drifted["timestamp"] = datetime.now()
    
    prediction = pok_classifier.predict(df_test=df_test_drifted)
    prediction_int = [1 if e=='True' else 0 for e in prediction]
    prediction_numpy = np.asarray(prediction_int)

    # Create an instance of Influxdb
    influxdb = Influxdb().get_database_actions()

    db_login = {
        "url": url,
        "token": token,
        "org": org,
        "bucket_name": bucket_name
    }

    # Test InfluxDB connection
    # db_connection = influxdb.make_connection(**db_login)

    params = PulseParameters(
        model_id="mod1",
        model_version="ver1",
        data_id="dat1",
        reference_data_storage=df_user_input,
        target_name="Legendary",
        storage_engine="influxdb",
        timestamp_column_name="timestamp",
        login=db_login,
        other_labels={"timezone": "EST"},
    )

    pulse = Pulse(data=params)

    if not df_test_drifted.index.is_unique:
        df_test_drifted = df_test_drifted.reset_index(drop=True)

    time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    capture_params = DataWithPrediction(
        data_points=df_test_drifted,
        predictions=pd.DataFrame(prediction_numpy, columns=["prediction"]),
        timestamp=time,
        features_names=df_test_drifted.columns.tolist(),
    )

    pulse.capture_data(data=capture_params)

    df_user_input = df_user_input.astype(str)

    response_content = df_user_input.to_dict(orient='records')

    return {"response_content": response_content}
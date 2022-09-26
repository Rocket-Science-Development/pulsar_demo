import pandas as pd
import pickle as pkl
from pulsar_data_collection.data_capture import DataCapture, DatabaseLogin, DataWithPrediction


if __name__ == "__main__":
    print("Pushing data")
    data_test = pd.read_csv('/app/datasets/test_data_no_class.csv')

    model = pkl.load(open("/app/models/kidney_disease.pkl", "rb"))

    test_data = data_test.sample(frac=0.3, random_state=1)

    prediction = model.predict(test_data)

    database_login = DatabaseLogin(
        db_host="influxdb",
        db_port=8086,
        db_user="admin",
        db_password="pass123",
        protocol="line"
    )

    dat_capture = DataCapture(
        storage_engine="influxdb",
        model_id="1",
        model_version="2",
        data_id="FluxDB",
        y_name="y_pred",
        pred_name="clf_target",
        operation_type="INSERT_PREDICTION",
        login_url=database_login
    )

    dat_predict = DataWithPrediction(prediction=prediction, data_points=test_data)

    dat_capture.push(dat_predict)

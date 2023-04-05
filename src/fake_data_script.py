import pickle as pkl

import pandas as pd
from pulsar_data_collection.data_capture import (
    DatabaseLogin,
    DataCapture,
    DataWithPrediction,
)

from gen_data_and_simulate_drift import DriftIntensity, DriftSimulator, GenerateFakeData
from training_script import Classifier

# if __name__ == "__main__":
    # print("Pushing data")
    # data_test = pd.read_csv('/app/datasets/test_data_no_class.csv')

    # model = pkl.load(open("/app/models/kidney_disease.pkl", "rb"))

    # test_data = data_test.sample(frac=0.3, random_state=1)

    # prediction = model.predict(test_data)


if __name__ == '__main__':
    target = 'Legendary'
    genertor_fake_data = GenerateFakeData(path_ref_data='pokemon.csv', sample_size = 1000, target=target)
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
    # to get test_data after drifting
    df_test_drifted = drift_sim_info.get_test_data_drifted()

    prediction = pok_classifier._predict(df_test=df_test_drifted)

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

    dat_predict = DataWithPrediction(prediction=prediction, data_points=df_test_drifted)

    dat_capture.push(dat_predict)

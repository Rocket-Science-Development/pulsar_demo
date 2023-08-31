import pickle as pkl
import os
import pandas as pd
import numpy as numpy
from pulsar_data_collection.data_capture import (
    DatabaseLogin,
    DataCapture,
    DataWithPrediction,
)

from gen_data_and_simulate_drift import DriftIntensity, DriftSimulator, GenerateFakeData
from training_script import Classifier

# Below is the code for Kidney disease model
# kidneydisease_test_data='/app/datasets/test_data_no_class.csv'
# kidneydisease_model="/app/models/kidney_disease.pkl"
# if __name__ == "__main__":
#     print("Pushing data")
#     data_test = pd.read_csv(kidneydisease_test_data)

#     model = pkl.load(open(kidneydisease_model, "rb"))

#     test_data = data_test.sample(frac=0.3, random_state=1)

#     prediction = model.predict(test_data)

# Below is the code for Pokemon model
pokemon_test_data='/app/datasets/bank_num.csv'
SAMPLE_SIZE=1000
if __name__ == '__main__':
    target = 'default'
    genertor_fake_data = GenerateFakeData(path_ref_data=pokemon_test_data, sample_size=SAMPLE_SIZE, target=target, model_name = 'bank_generator.pkl')
    sampled_data = genertor_fake_data.get_dataclass_sampling()

    # if the task is classification
    # pok_classifier = Classifier(df_train=sampled_data.train_data,
    #             num_features=sampled_data.list_num_col,
    #             cat_features=None,
    #             target=target,
    #             pkl_file_path=f'class_{target}_model.pkl')
    pok_classifier = Classifier(df_train=sampled_data.train_data,
                num_features=sampled_data.list_num_col,
                cat_features=None,
                target=target,
                pkl_file_path=os.path.join('/app', f'class_{target}_model.pkl')
                ,iterations = 200, depth = None, learning_rate = 0.03) 
    
    pok_classifier.train()
    pok_classifier.serialize()

    # pok_classifier.load_model()

    drift_sim_info = DriftSimulator(sampled_data, nb_cols_to_drift=1, drift_intensity=DriftIntensity.MODERATE)
    # to get test_data after drifting
   
    df_test_drifted = drift_sim_info.get_test_data_drifted()
    print('info:',df_test_drifted.dtypes)
    # df_test_drifted[target] = df_test_drifted[target].astype(int)

    prediction = pok_classifier.predict(df_test=df_test_drifted.drop(target, axis=1))
    prediction_int = [1 if e=='yes' else 0 for e in prediction]

    conversion_dict = {'yes': 1, 'no': 0}
    df_test_drifted['default'] = df_test_drifted['default'].map(conversion_dict)
    # prediction = prediction.astype(int)
    # print('prediction: ', prediction)
    # print('prediction.type',prediction.dtypes)
    # print('prediction_int: ', prediction_int)
    # print('prediction_int_type',prediction_int.dtypes)

    # Adding the logic to Flip a % of values in our Prediction array before ingesting the data in DB so as to have FP & FN values generated successfully.
    # Introduce random coefficient
    # randomization_percentage = 0.2  # Can be adjusted. 0.2 here means 20% of the values will be flipped. Accuracy hence would be 80% in this scenario.

    # Calculate the number of values to be changed based on the coefficient
    # num_values_to_change = int(len(prediction_int) * randomization_percentage)

    # Generate random indices to select positions for randomization
    # random_indices = numpy.random.choice(len(prediction_int), size=num_values_to_change, replace=False)

    # Iterate through the random indices and flip the corresponding values in prediction_int
    # for index in random_indices:
    #     prediction_int[index] = 1 - prediction_int[index]

    # print('prediction_int after: ', prediction_int)

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

    prediction_numpy = numpy.asarray(prediction_int)

    dat_predict = DataWithPrediction(prediction=prediction_numpy, data_points=df_test_drifted)

    dat_capture.push(dat_predict)

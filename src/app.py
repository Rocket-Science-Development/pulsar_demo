from gen_data_and_simulate_drift import GenerateFakeData, DriftSimulator, DriftIntensity
from training_script import Classifier


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

    drift_sim_info = DriftSimulator(sampled_data, nb_cols_to_drift=1, drift_intensity=DriftIntensity.MODERATE)
    # to get test_data after drifting
    df_test_drifted = drift_sim_info.get_test_data_drifted()
import pickle as pkl
import os
import pandas as pd
from gen_data_and_simulate_drift import GenerateFakeData
from training_script import Classifier

# Load and train the classifier
target = 'Legendary'
reference_data = 'pokemon.csv'
SAMPLE_SIZE = 1000

if __name__ == '__main__':
    genertor_fake_data = GenerateFakeData(path_ref_data=reference_data, sample_size=SAMPLE_SIZE, target=target)
    sampled_data = genertor_fake_data.get_dataclass_sampling()

    pok_classifier = Classifier(df_train=sampled_data.train_data,
                            num_features=sampled_data.list_num_col,
                            cat_features=None,
                            target=target,
                            pkl_file_path=os.path.join('/app', f'class_{target}_model.pkl'))
    
    pok_classifier.train()
    pok_classifier.serialize()
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from enum import Enum
import warnings
from dataclasses import dataclass
from typing import List, Dict
import logging
import random
from sklearn.model_selection import train_test_split
import numpy as np
logging.getLogger().setLevel(logging.INFO)

warnings.filterwarnings("ignore")

class SamplingMethod(Enum):
    COPULAS_GAUSS_MULT = 1

class DriftIntensity(Enum):
    MODERATE = 1,
    EXTREME = 2

@dataclass
class SampledData:
    df_ref_data: pd.DataFrame
    df_sampled: pd.DataFrame
    train_data:pd.DataFrame
    test_data:pd.DataFrame
    list_num_col: List
    used_distribution: SamplingMethod


def create_dict_type_for_df(df_ref:pd.DataFrame):
    '''
    :param df_ref: dataframe contains the data reference (original)
    :return: a dictionary (key = column name; value = type of column)
    '''
    dict_col_type= {}
    for col in df_ref.columns:
        dict_col_type[col] = df_ref[col].dtype
    return dict_col_type

def get_shift_coef(drift_intensity:DriftIntensity):
    '''
    :param drift_intensity: enum with the level of intensity to be used
    :return: the new mean and stdev; used to drift features
    '''
    if drift_intensity == DriftIntensity.MODERATE:
        mean_coef = random.uniform(1.2, 1.7)
        std_coef = random.uniform(1.5, 2)

    if drift_intensity == DriftIntensity.EXTREME:
        mean_coef = random.uniform(1.5, 2.7)
        std_coef = random.uniform(2, 5)

    return mean_coef, std_coef


class GenerateFakeData():
    def __init__(self, path_ref_data:str='pokemon.csv',
                 sample_size = 1000,
                 sampling_method= SamplingMethod.COPULAS_GAUSS_MULT,
                 target = 'Legendary'):
        '''

        :param path_ref_data: the dataset (reference)
        :param sample_size: number of samples to generate
        :param sampling_method: COPULAS method to be used to generate new data
        :param target: target of the dataset (we assume a supervised task)
        '''
        self.df_ref = None
        self.num_cols = []
        self.df_samples= pd.DataFrame({})
        self.data_ref_target= target
        self.sampling_method = sampling_method
        self.sample_size = sample_size
        self.train_set = pd.DataFrame({})
        self.test_set = pd.DataFrame({})

        self.path_ref_data=path_ref_data
        # logging.INFO('Read reference data...')
        self.read_data_reference()
        # logging.INFO('Keep numerical columns...')
        self.keep_numerical_col()
        self.dict_col_type = create_dict_type_for_df(self.df_ref)

        # logging.INFO('Generate fake/synthetic data...')
        if self.sampling_method == SamplingMethod.COPULAS_GAUSS_MULT:
            self.generate_fake_data_using_copulas()
        # TODO add more distributions

        self.apply_right_type_to_generated_columns()

    def keep_numerical_col(self):
        '''
        keep only the numerical columns and target
        :return: the data ref with num cols and target
        '''
        target_col = self.df_ref[self.data_ref_target]
        self.df_ref = self.df_ref.select_dtypes(['number'])
        self.num_cols = list(self.df_ref.columns)
        self.df_ref[self.data_ref_target]= target_col

    def read_data_reference(self):
        self.df_ref = pd.read_csv(self.path_ref_data)


    def generate_fake_data_using_copulas(self):
        dist= None

        # generate new data using COPULAS_GAUSS_MULT method
        # TODO add more distribution option
        if self.sampling_method == SamplingMethod.COPULAS_GAUSS_MULT:
            dist = GaussianMultivariate()

        for a_target in self.df_ref[self.data_ref_target].unique():
            dist.fit(self.df_ref[self.df_ref[self.data_ref_target] == a_target].drop([self.data_ref_target], axis=1))
            sampled = dist.sample(self.sample_size)
            sampled[self.data_ref_target] = a_target
            self.df_samples = self.df_samples.append(sampled)

    def apply_right_type_to_generated_columns(self):
        '''
        if the generated data don't have the best type
        almost for the integer (p.ex avoid to have 'age' as float)
        :return: generated dataframe with the good type
        '''
        for c, type in self.dict_col_type.items():
            self.df_samples[c] = self.df_samples[c].astype(type)

    def get_dataclass_sampling(self):
        '''

        :return: data class with the relevant information/data
        '''
        df_train, df_test = train_test_split(self.df_samples, test_size=0.2)
        return SampledData(df_ref_data= self.df_ref,
                           df_sampled=self.df_samples,
                           list_num_col=self.num_cols,
                           used_distribution=self.sampling_method,
                           train_data=df_train,
                           test_data=df_test
                           )


class DriftSimulator():
    '''
    class used to drift the 'selected'(random or specified) columns.

    there is two options:
    1) give the number N of columns to be drifted; and then select randomly these N columns.
    2) give a list of the columns to be drifted

    # TODO specify for each column the intensity of the drift (MEDIUM or EXTREME)
    '''
    def __init__(self, sampled_data:SampledData,
                 nb_cols_to_drift:int,
                 drift_intensity:DriftIntensity = None,
                 selected_columns_to_drift = []):
        self.input_data = sampled_data
        self.selected_columns_to_drift= selected_columns_to_drift
        self.nb_col_to_drift = nb_cols_to_drift
        self.drift_intensity = drift_intensity
        self.test_data_drifted = self.input_data.test_data.copy()
        # assert ((drift_intensity == None and len(selected_columns_to_drift) > 0)
        #         or (self.selected_columns_to_drift == 0))
        #
        # assert (len(self.selected_columns_to_drift) == 0
        #         or (len(self.selected_columns_to_drift) == self.nb_col_to_drift)
        #         ), 'Number to columns must be equal to  the "number of selected columns"'

        self.nb_col_to_drift  = min(len(self.input_data.list_num_col), nb_cols_to_drift)
        print(f'number of columns to drift is : {self.nb_col_to_drift}')

        if len(self.selected_columns_to_drift) == 0:
            print('select random column to drift ...')
            self.selected_columns_to_drift = random.sample(self.input_data.list_num_col,
                                                          self.nb_col_to_drift)

        self.run_drifting()


    def run_drifting(self):

        size_test_data = self.test_data_drifted.shape[0]
        for col in self.selected_columns_to_drift:
            print(f'Drifting column {col}')
            mean = self.test_data_drifted[col].mean()
            std  = self.test_data_drifted[col].std()
            # get the coefs for the to be applied for the mean and the std for the actual distribution
            # new_mean = actual_mean * coef_mean (obtained randomly, based on the intensity)
            # new_std = actual_std  * coef_std (obtained randomly, based on the intensity)
            coef_mean, coef_std = get_shift_coef(self.drift_intensity)

            # apply drifting to the test set using the new distribution
            # TODO add more distribution option
            drifted_data = np.random.normal(mean*coef_mean, std*coef_std, size_test_data)
            self.test_data_drifted[col] = drifted_data

    def get_test_data_drifted(self):
        return self.test_data_drifted

if __name__ == '__main__':
    '''
    Example how to simulate drift
    pokemon dataset is used per default
    '''
    genertor_fake_data = GenerateFakeData()
    samplet_data = genertor_fake_data.get_dataclass_sampling()
    ds = DriftSimulator(samplet_data, nb_cols_to_drift=1, drift_intensity=DriftIntensity.MODERATE)
    a = ds.get_test_data_drifted()



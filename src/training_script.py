import pickle
from typing import List

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self, df_train:pd.DataFrame,
                num_features:List,
                cat_features:List,
                target:str,
                 pkl_file_path:str):
        self.df_train = df_train
        self.num_features = num_features
        self.target = target
        self.pkl_file_path= pkl_file_path
        if not cat_features:
            self.cat_features = list(set(self.df_train.columns) - set(num_features))

    def serialize(self):
        with open(self.pkl_file_path, 'wb') as file:
            pickle.dump(self, file)

    def train(self):
        self.model = CatBoostClassifier(iterations=2,
                                   learning_rate=1,
                                   depth=2)
        self.model.fit(self.df_train[self.num_features + self.cat_features], self.df_train[self.target])



    def _predict(self, df_test):

            # Use the loaded model to make predictions
            return self.model.predict(df_test)
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
            try:
                 pickle.dump(self, file)
            except pickle.PickleError as e:
                 # handle any PickleError exceptions that may occur during serialization
                 print("Error occurred while pickling:", e)

    """
    Trains a machine learning model using a dataframe (self. df_train and specified features [self.num_features + self.cat_features] 
    and target (self.target) attributes.
    """
    def train(self):
        self.model = CatBoostClassifier(iterations=2,
                                   learning_rate=1,
                                   depth=2)
        self.model.fit(self.df_train[self.num_features + self.cat_features], self.df_train[self.target])



    def predict(self, df_test):
            self.load_model()
            # Use the loaded model to make predictions
            return self.model.predict(df_test)
    
    def load_model(self):
         with open(self.pkl_file_path, 'rb') as file:
            loaded_classifier = pickle.load(file)
            self.model = loaded_classifier.model

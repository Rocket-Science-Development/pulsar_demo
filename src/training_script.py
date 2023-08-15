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

            # Use the loaded model to make predictions
            return self.model.predict(df_test)
    
    @classmethod
    def load_model(cls, pkl_file_path):
        """
        Loads a serialized Classifier object from the specified .pkl file.

        Parameters:
            pkl_file_path (str): Path to the .pkl file.

        Returns:
            Classifier: A loaded Classifier object.
        """
        with open(pkl_file_path, 'rb') as file:
            try:
                classifier_obj = pickle.load(file)
                if not isinstance(classifier_obj, cls):
                    raise ValueError("Invalid pickled object. Expected Classifier object.")
                return classifier_obj
            except pickle.PickleError as e:
                # handle any PickleError exceptions that may occur during deserialization
                print("Error occurred while unpickling:", e)
                return None

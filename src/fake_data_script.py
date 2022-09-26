import pandas as pd
import pickle as pkl
from pydantic import BaseModel, validator
import numpy as np
from datetime import datetime
from enum import Enum
from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
import pytz
from pulsar_data_collection.data_capture import DataCapture, DatabaseLogin, DataWithPrediction


data_test = pd.read_csv('/app/datasets/test_data_no_class.csv')

model = pkl.load(open("/app/models/kidney_disease.pkl", "rb"))

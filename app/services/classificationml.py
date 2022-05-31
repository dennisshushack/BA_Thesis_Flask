import os 
import sys 
import time 
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class classificationml:
    """
    This class is for training & testing ML algorithms 
    for classification.
    """
    @staticmethod
    def train(df:pd.DataFrame, train_path: str, feature: str):
        """
        This function trains the data.
        :input: df: dataframe, train_path: path to train data, feature: feature to train
        :output: df: dataframe
        """
        # List, that will be returned
        model_list = []

        


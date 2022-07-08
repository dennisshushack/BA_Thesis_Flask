import os
import sys
import time
import pickle
import warnings
import pandas as pd
import numpy as np
from app.database.dbqueries import dbqueries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class anomalyml:
    """
    This class is for training & testing ML algorithms 
    for anomaly detection.
    """

    @staticmethod
    def validate(df:pd.DataFrame, train_path: str, feature: str):
        """
        This function validates the data.
        :input: df: dataframe, train_path: path to train data, feature: feature to validate
        :output: list with metrics
        """
        # List, that will be returned 
        model_list = []

        ml_path = train_path + '/MLmodels/'

        classifiers = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]

        # Create a list from the last column of the dataframe
        # And create a numpy array:
        X = df[feature].tolist()
        X = np.array(X)

        # Labels y for later use to calculate TPR:
        y = [1 for i in range(0,len(X))]

        # Train Loop:
        for classifier in classifiers:
            results = {classifier: []}

            # Load the trained model:
            with open(ml_path + feature + classifier + ".pickle", "rb") as f:
                model = pickle.load(f)
            
            t1 = time.time()
            y_pred = model.predict(X)
            t2 = time.time()
            test_time = t2 - t1
            val_score = accuracy_score(y,y_pred)
            results[classifier].append(val_score)
            results[classifier].append(test_time)
            model_list.append(results)

        return model_list

    @staticmethod
    def validate_live(df:pd.DataFrame, train_path: str, feature: str, db_connection, device):
        ml_path = train_path + '/MLmodels/'
        classifiers = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        X = df[feature].tolist()
        X = np.array(X)

        timestamp = df['timestamps'].tolist()
        print(timestamp)

         # Train Loop:
        for classifier in classifiers:

            # Load the trained model:
            with open(ml_path + feature + classifier + ".pickle", "rb") as f:
                model = pickle.load(f)
            
            t1 = time.time()
            print(f"{classifier} {feature}: Start at : {t1}")
            y_pred = model.predict(X)
            t2 = time.time()
            print(f"{classifier} {feature}: End at : {t2}")
            test_time = t2 - t1
            print(f"{classifier} {feature}: Test time: {test_time}")
            time_stamp = timestamp[0]
            y_pred = y_pred[0]
            y_pred = int(y_pred)
            dbqueries.insert_into_live(db_connection, device, "anomaly", classifier, feature, time_stamp, y_pred, test_time)





    @staticmethod
    def train(df: pd.DataFrame, train_path: str, feature: str) -> list:
        """
        This function trains the ML algorithms for anomaly detection
        :input: df: dataframe with the data to train the ML algorithms,
        :input: train_path: path to the directory where the trained models will be saved,
        :input: feature: the feature to train the ML algorithms on
        :output: a dictionary with the trained models
        """
        # List, that will be returned 
        # Contains dictonnairies {name of the model: [accuracy, time]}
        model_list = []

        # Check if the directory exists, if not create it:
        ml_path = train_path + '/MLmodels/'
        if not os.path.exists(ml_path):
            os.makedirs(ml_path)

        # Defined contamination:s
        contamination_factor = 0.05

        classifiers = {
            'IsolationForest':IForest(random_state=42, contamination=contamination_factor),
            'OneClassSVM':OCSVM(kernel='rbf',gamma=0.0001, nu=0.3, contamination=contamination_factor),
            'LocalOutlierFactor': LOF(n_neighbors=50, contamination=contamination_factor)
        }
        
        # Create a list from the last column of the dataframe
        # And create a numpy array:
        X = df[feature].tolist()
        X = np.array(X)

        # Labels y for later use to calculate FPR:
        y = [0 for i in range(0,len(X))]

        # Create a train-test split:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

        # Train the models and save them:
        # Train the models and save them:
        for name, clf in classifiers.items():
            results = {name: []}
            t1 = time.time()
            clf.fit(X_train)
            t2=time.time()
            y_pred = clf.predict(X_val)
            val_score = accuracy_score(y_val,y_pred)
            results[name].append(val_score)
            training_time = t2 - t1
            results[name].append(training_time)
            model_list.append(results)
            # Save the model:
            with open(ml_path + feature +  name + '.pickle', 'wb') as f:
                pickle.dump(clf, f)
        
        return model_list



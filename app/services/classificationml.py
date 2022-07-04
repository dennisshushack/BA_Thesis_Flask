import os 
import sys 
import time 
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn.linear_model import LogisticRegression


if not sys.warnoptions:
    warnings.simplefilter("ignore")


class classification:
    """
    This class is for training & testing ML algorithms 
    for classification.
    """
    @staticmethod
    def evaluate(df:pd.DataFrame, train_path: str, feature: str):
        """
        This function evaluates, to which class the sample belongs too.
        """
        X = df[feature].tolist()
        X = np.array(X)

        # Loads the model from the pickle file

        with open(file, 'rb') as f:
            model = pickle.load(f)
        




    @staticmethod
    def train(df:pd.DataFrame, train_path: str, feature: str):
        """
        This function trains the data.
        :input: df: dataframe, train_path: path to train data, feature: feature to train
        :output: df: dataframe
        """
        # List, that will be returned
        y = df["behavior"]
        X = df[feature].tolist()
        X = np.array(X)

        # Create a train-test split:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, shuffle=True, random_state=42)

        # Check if the directory exists, if not create it:
        ml_path = train_path + '/MLmodels/'
        if not os.path.exists(ml_path):
            os.makedirs(ml_path)
        

        # Classifiers for Multiclass classification:
        clf = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000) 
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
    
        # Get classification report:
        report = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose()

        # Check if outputpath exists, if not create it:
        output_path = train_path + "/Reports/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save the model:
        with open(ml_path + feature +  "LogisticRegression" + '.pickle', 'wb') as f:
            pickle.dump(clf, f)            
        
        # Save the report as a .csv file in the output folder:
        report.to_csv(output_path + feature + "LogisticRegression" + '_report.csv')

        
        


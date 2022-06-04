import os 
import sys 
import time 
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

if not sys.warnoptions:
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
        classifiers = {
            "Logistic Regression": LogisticRegression(solver='saga',multi_class='ovr', max_iter=5000),
            "SVM": SVC(gamma='auto', kernel='rbf'),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier()
        }
        # Train the models and save them:
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
      
            # Get recall precision and f1 score & accuracy score:
            # Get classification report:
            report = classification_report(y_val, y_pred)
            print(report)            
            report = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose()

            # Check if outputpath exists, if not create it:
            output_path = train_path + "/Output/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            # Save the model:
            with open(ml_path + feature +  name + '.pickle', 'wb') as f:
                pickle.dump(clf, f)            
            
            # Save the report as a .csv file in the output folder:
            report.to_csv(output_path + feature + name + '_report.csv')

        
        


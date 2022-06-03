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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, shuffle=False, random_state=42)

        # Check if the directory exists, if not create it:
        ml_path = train_path + '/MLmodels/'
        if not os.path.exists(ml_path):
            os.makedirs(ml_path)
        

        # Classifiers for Multiclass classification:
        classifiers = {
            "Logistic Regression": LogisticRegression(solver='saga',multi_class='ovr', max_iter=5000),
            "SVM": SVC(gamma='auto', kernel='rbf'),
            "Random Forest": RandomForestClassifier(n_estimators=100, ),
            "KNN": KNeighborsClassifier()
        }
        # Train the models and save them:
        for name, clf in classifiers.items():
            results = {name: []}
            t1 = time.time()
            clf.fit(X_train, y_train)
            t2 = time.time()
            y_pred = clf.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            results[name].append(accuracy)
            results[name].append(t2-t1)

            # Get recall precision and f1 score:
            report = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True))
            results[name].append(report)
            # Save the result as a text file:
            with open(ml_path + name + '.txt', 'w') as f:
                f.write(str(results[name]))

            # Save the model:
            with open(ml_path + feature +  name + '.pickle', 'wb') as f:
                pickle.dump(clf, f)            
            
            model_list.append(results)
        
        return model_list
        


import os
from random import shuffle
import sys
import time
import pickle
import warnings
import pandas as pd
import numpy as np
from app.database.dbqueries import dbqueries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    def train(features, training_path, device, db) -> list:
        
        # Defined contamination + algorithms to train:
        contamination_factor = 0.05
        classifiers = {
            'IsolationForest':IForest(random_state=42, contamination=contamination_factor),
            'OneClassSVM':OCSVM(kernel='rbf',gamma=0.0001, nu=0.3, contamination=contamination_factor),
            'LocalOutlierFactor': LOF(n_neighbors=50, contamination=contamination_factor)
        }

        # Checking all paths:
        if not os.path.exists(training_path + '/models/'):
            os.makedirs(training_path + '/models/')

        if not os.path.exists(training_path + '/scalers/'):
            os.makedirs(training_path + '/scalers/')

        for featurename, corpus in features[2].items():
            y = [0 for i in range(0,len(corpus))]
            print("Training ML: " + featurename)


            # Split the data into training and testing:
            X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.3, random_state=42, shuffle=False)
            
            # Scaling the data:
            scaler = StandardScaler()
            scaler.fit(X_train)

            # Save the scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'wb') as f:
                pickle.dump(scaler, f)
    
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Train Loop:
            for classifier_name, classifier in classifiers.items():
                print("Start training: " + featurename + classifier_name + " " + str(time.time()))
                classifier.fit(X_train)
                print("End training: " + featurename + classifier_name + " " + str(time.time()))
                # Save the model:
                with open(training_path + '/models/' + featurename + classifier_name + ".pickle", "wb") as f:
                    pickle.dump(classifier, f)
                
                # Predict the test data:
                print("Start predicting: " + classifier_name + " " + str(time.time()))
                y_pred = classifier.predict(X_test)
                print("End predicting: " + classifier_name + " " + str(time.time()))

                # Calculate the accuracy:
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy: " + str(accuracy))

                # Insert the accuracy into the database:
                dbqueries.insert_into_anomaly_detection(db, device , featurename, classifier_name, accuracy)


    @staticmethod
    def validate(features, training_path, device, db, experiment, behavior):

        for featurename, corpus in features[2].items():
            y = [1 for i in range(0,len(corpus))]
            print("Testing ML: " + featurename)

            classifiers = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]

            # Load scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            
            # Transform the data:
            corpus = scaler.transform(corpus)

            # Load the trained model:
            for classifier_name in classifiers:
                with open(training_path + '/models/' + featurename + classifier_name + ".pickle", "rb") as f:
                    classifier = pickle.load(f)

                y_pred = classifier.predict(corpus)
                print(classifier_name + featurename)
                accuracy = accuracy_score(y, y_pred)
                primary_key = dbqueries.get_foreign_key_ml(db, device, featurename, classifier_name)
                dbqueries.create_ml_anomaly_testing(db, device, experiment, behavior, featurename, classifier_name, accuracy, primary_key)

        return

        
    @staticmethod
    def validate_live(features, training_path, device, db):
        
        for featurename, corpus in features[2].items():
            y = [1 for i in range(0,len(corpus))]
            timestamp = features[0][0]
            classifiers = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]

            # Load scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            corpus = scaler.transform(corpus)

            for classifier_name in classifiers:
                with open(training_path + '/models/' + featurename + classifier_name + ".pickle", "rb") as f:
                    classifier = pickle.load(f)
                start = time.time()
                y_pred = classifier.predict(corpus)
                end = time.time()
                test_time = end - start
                dbqueries.insert_into_live(db, device, "anomaly", classifier, featurename, timestamp, y_pred, test_time)

        return
       


        

    
        
        




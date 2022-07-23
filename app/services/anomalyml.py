import os
import sys
import time
import pickle
import warnings
import pandas as pd
import numpy as np
import csv
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

        if not os.path.exists(training_path + '/preprocessed/'):
                os.makedirs(training_path + '/preprocessed/')

        for featurename, corpus in features[2].items():

            y = [0 for i in range(0,len(corpus))]
            print("Training ML: " + featurename)

            # Split the data into training and testing:
            X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.1, random_state=42, shuffle=True)
            
            # Scaling the data:
            scaler = StandardScaler()
            scaler.fit(X_train)

            # Save the scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'wb') as f:
                pickle.dump(scaler, f)
    
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            corpus = scaler.transform(corpus)

            # Create a datatframe:
            timestamp = features[0]
            behavior = features[1]
            normaled_df = pd.DataFrame([timestamp, behavior, corpus.tolist()]).transpose()
            normaled_df.columns = ['timestamp', 'behavior', featurename]
            # Save the dataframe:
            normaled_df.to_csv(training_path + '/preprocessed/' + featurename + '_preprocessed.csv', index=False)

            # Train Loop:
            for classifier_name, classifier in classifiers.items():
                
                # Train the classifier:
                start_train = time.time()
                classifier.fit(X_train)
                end_train = time.time()
                train_time = end_train - start_train

                # Save the model:
                with open(training_path + '/models/' + featurename + classifier_name + ".pickle", "wb") as f:
                    pickle.dump(classifier, f)
                
                # Predict the test data:
                start_prediction = time.time()
                y_pred = classifier.predict(X_test)
                end_prediction = time.time()
                test_time = end_prediction - start_prediction

                # Calculate the accuracy:
                accuracy = accuracy_score(y_test, y_pred)

                # Save the metrics:
                with open('/tmp/output.csv', 'a', newline='') as csvfile:
                    fieldnames = ['Name', 'Start', 'End', 'Duration']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'Name':f'Training {classifier_name} with feature {featurename}', 'Start': start_train, 'End': end_train, 'Duration': end_train - start_train})
                    writer.writerow({'Name':f'Prediction {classifier_name} with feature {featurename}', 'Start': start_prediction, 'End': end_prediction, 'Duration': end_prediction - start_prediction})
                    # Close the file
                    csvfile.close()

                # Insert the accuracy into the database:
                dbqueries.insert_into_anomaly_detection(db, device , featurename, classifier_name, accuracy, train_time, test_time)


    @staticmethod
    def validate(features, training_path, device, db, experiment, behavior, path):

        if not os.path.exists(path + '/preprocessedanomaly/'):
                os.makedirs(path + '/preprocessedanomaly/')

        for featurename, corpus in features[2].items():
            y = [1 for i in range(0,len(corpus))]

            classifiers = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]

            # Load scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            
            # Transform the data:
            corpus = scaler.transform(corpus)

            # Saving the data:
            # Create a datatframe:
            timestamp = features[0]
            behaviors = features[1]
            normaled_df = pd.DataFrame([timestamp, behaviors, corpus.tolist()]).transpose()
            normaled_df.columns = ['timestamp', 'behavior', featurename]
            # Save the dataframe:
            normaled_df.to_csv(path + '/preprocessedanomaly/' + featurename + '_preprocessed.csv', index=False)

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
       


        

    
        
        




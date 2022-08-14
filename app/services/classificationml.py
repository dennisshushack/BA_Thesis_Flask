import os 
import sys 
import time 
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from app.database.dbqueries import dbqueries
import csv

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class classification:
    """
    This class is for training & testing ML algorithms 
    for classification.
    """
    @staticmethod
    def train(features, training_path):

         # Checking all paths:
        if not os.path.exists(training_path + '/models/'):
            os.makedirs(training_path + '/models/')

        if not os.path.exists(training_path + '/scalers/'):
            os.makedirs(training_path + '/scalers/')

        if not os.path.exists(training_path + "/Reports/"):
            os.makedirs(training_path + "/Reports/")

        if not os.path.exists(training_path + "/preprocessed/"):
            os.makedirs(training_path + "/preprocessed/")


        classifiers = {
            'LogisticRegression': LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'SVM': SVC(gamma='auto', kernel='rbf'),
            'RandomForestClassifier': RandomForestClassifier()
        }

        for featurename, corpus in features[2].items():
            
            # Behavioral features:s
            y = features[1]

            # Create a train-test split:
            X_train, X_val, y_train, y_val = train_test_split(corpus, y, test_size=.3, shuffle=True, random_state=42)

            # Standardize the data:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            corpus = scaler.transform(corpus)

            # Save the scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'wb') as f:
                pickle.dump(scaler, f)
            
            # Create a datatframe:
            timestamp = features[0]
            behavior = features[1]
            normaled_df = pd.DataFrame([timestamp, behavior, corpus.tolist()]).transpose()
            normaled_df.columns = ['timestamp', 'behavior', featurename]
            # Save the dataframe:
            normaled_df.to_csv(training_path + '/preprocessed/' + featurename + '_preprocessed.csv', index=False)

            # Train the models:
            for name, clf in classifiers.items():
                start_train = time.time()
                clf.fit(X_train, y_train)
                end_train = time.time()
                training_time = end_train - start_train
                # Save the model:
                with open(training_path + '/models/' + featurename + '_' + name + '.pickle', 'wb') as f:
                    pickle.dump(clf, f)

                # Evaluate the model:
                start_eval = time.time()
                y_pred = clf.predict(X_val)
                end_eval = time.time()
                testing_time = end_eval - start_eval
                print(classification_report(y_val, y_pred))
                with open(training_path + '/Reports/' + featurename + '_' + name + '_report.txt', 'w') as f:
                    f.write(classification_report(y_val, y_pred))

            
                    # Save the metrics:
                with open('output.csv', 'a', newline='') as csvfile:
                    fieldnames = ['Name', 'Start', 'End', 'Duration']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'Name':f'Training Classification {name} with feature {featurename}', 'Start': start_train, 'End': end_train, 'Duration': end_train - start_train})
                    writer.writerow({'Name':f'Testing Classification {name} with feature {featurename}', 'Start': start_eval, 'End': end_eval, 'Duration': end_eval - start_eval})
                    csvfile.close()
                    
        return
    
    @staticmethod
    def validate_live(features, training_path, device, db):
        
        for featurename, corpus in features[1].items():
            timestamp = features[0][0]
            timestamp = int(timestamp)
            classifiers = ["LogisticRegression", "DecisionTreeClassifier", "SVM", "RandomForestClassifier"]

            # Load scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)

            corpus = scaler.transform(corpus)

            for classifier_name in classifiers:
                # Load model:
                with open(training_path + '/models/' + featurename + '_' + classifier_name + '.pickle', 'rb') as f:
                    clf = pickle.load(f)

                # Predict:
                start = time.time()
                y_pred = clf.predict(corpus)
                end = time.time()
                duration = end - start
                y_pred = y_pred[0]
                if y_pred == "normal":
                    y_pred = 0
                elif y_pred == "raas":
                    y_pred = 1
                elif y_pred == "poc":
                    y_pred = 2
                else:
                    y_pred = 3
                dbqueries.insert_into_live(db, device, "classification", classifier_name, featurename, timestamp, y_pred, duration)
        
        return
                


        



    @staticmethod
    def test(features, training_path, path):
        classifiers = ["LogisticRegression", "DecisionTreeClassifier", "SVM", "RandomForestClassifier"]
        # Create direcory for evaluation:
        if not os.path.exists(path + '/evaluation/'):
            os.makedirs(path + '/evaluation/')

        # Create a preprocessed data folder:
        if not os.path.exists(path + '/preprocessed/'):
            os.makedirs(path + '/preprocessed/')

        for featurename, corpus in features[2].items():
        
            # Load the scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            
            # Standardize the data:
            corpus = scaler.transform(corpus)

            timestamp = features[0]
            behaviors = features[1]
            normaled_df = pd.DataFrame([timestamp, behaviors, corpus.tolist()]).transpose()
            normaled_df.columns = ['timestamp', 'behavior', featurename]
            normaled_df.to_csv(path + '/preprocessed/' + featurename + '_preprocessed.csv', index=False)

            # Load the model and do the training:
            for name in classifiers:
                with open(training_path + '/models/' + featurename + '_' + name + '.pickle', 'rb') as f:
                    clf = pickle.load(f)

                # Predict the labels:
                y_pred = clf.predict(corpus)
                normaled_df = pd.DataFrame([timestamp, behaviors, y_pred]).transpose()
                normaled_df.columns = ['timestamp', 'actualbehavior', 'prediction']
                normaled_df.to_csv(path + '/evaluation/' + featurename + '_' + name + '_evaluation.csv', index=False)
        return
               


  
        
        


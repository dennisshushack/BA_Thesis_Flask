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
from app.database.dbqueries import dbqueries


if not sys.warnoptions:
    warnings.simplefilter("ignore")


class classification:
    """
    This class is for training & testing ML algorithms 
    for classification.
    """
    @staticmethod
    def validate_live(df, training_path, feature, db, device):
        """
        This function evaluates, to which class the sample belongs too.
        """
        X = df[feature].tolist()
        X = np.array(X)
        time_stamp = df['timestamps'].tolist()
        time_stamp = time_stamp[0]


        # Loads the model from the pickle file
        with open(training_path + "/MLmodels/" + feature + "LogisticRegression" + ".pickle", "rb") as f:
            model = pickle.load(f)
        
        # Predict the class of the sample
        start_time = time.time()
        y_pred = model.predict(X)
        end_time = time.time()
        testing_time = end_time - start_time
        y_pred = y_pred[0] 
        if y_pred == "normal":
            y_pred = 0
        elif y_pred == "poc":
            y_pred = 1
        elif y_pred == "dark":
            y_pred = 2
        else:
            y_pred = 3
        
        # Insert the prediction into the database
        dbqueries.insert_into_live(db, device, "classification","LogisticRegression", feature, time_stamp,y_pred, testing_time)



    @staticmethod
    def train(features, training_path):

         # Checking all paths:
        if not os.path.exists(training_path + '/models/'):
            os.makedirs(training_path + '/models/')

        if not os.path.exists(training_path + '/scalers/'):
            os.makedirs(training_path + '/scalers/')

        if not os.path.exists(training_path + "/Reports/"):
            os.makedirs(training_path + "/Reports/")

        # Classifiers for Multiclass classification:
        clf = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000) 

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

            # Save the scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'wb') as f:
                pickle.dump(scaler, f)

            print("Start training: " + featurename + "LR" + " " + str(time.time()))
            clf.fit(X_train, y_train)
            print("End training: " + featurename + "LR" + " " + str(time.time()))

            # Save the model:
            with open(training_path + '/models/' + featurename + "LR" + ".pickle", "wb") as f:
                    pickle.dump(clf, f)

            # Evaluate the model:
            print("Start evaluating: " + featurename + "LR" + " " + str(time.time()))
            y_pred = clf.predict(X_val)
            print("End evaluating: " + featurename + "LR" + " " + str(time.time()))
    
            # Get classification report:
            report = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose()


            # Save the report as a .csv file in the output folder:
            report.to_csv(training_path + "/Reports/" + featurename + "LR" + '_report.csv')
        
        return

        
        


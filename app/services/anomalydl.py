import os
from random import shuffle
import sys
import time
import pickle
import numpy as np
import pandas as pd
import csv
from app.database.dbqueries import dbqueries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Used to ignore annoying tensorflow warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Anomaly Deep Learning
# --------------------
class AnomalyDetector(Model):
    """
    This is the Autoencoder where the amount of features decides on 
    how many neurons are in the hidden layers.
    """
    def __init__(self, input_dim, layer_one, layer_two, layer_three, layer_four):
        super(AnomalyDetector, self).__init__()
        self.encoder = Sequential([
            Dense(layer_one, activation="relu"),
            Dense(layer_two, activation="relu"),
            Dense(layer_three, activation="relu"),
            Dense(layer_four, activation="relu")])

        self.decoder = Sequential([
            Dense(layer_three, activation="relu"),
            Dense(layer_two, activation="relu"),
            Dense(layer_one, activation="relu"),
            Dense(input_dim, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class anomalydl:

    @staticmethod
    def find_threshold(model, X_test):
        reconstruction = model.predict(X_test)
        train_loss = mae(X_test, reconstruction)
        # Lower threshold: 
        Q1,Q3 = np.percentile(train_loss , [25,75])
        IQR = Q3 - Q1
        IQR_lower_threshold = Q1 - 1.5*IQR
        IQR_upper_threshold = Q3 + 1.5*IQR
        STD_lower_threshold = np.mean(train_loss) - 2*np.std(train_loss)
        STD_upper_threshold = np.mean(train_loss) + 2*np.std(train_loss)
        return IQR_lower_threshold,IQR_upper_threshold, STD_lower_threshold, STD_upper_threshold, train_loss

    @staticmethod
    def append_labels(train_loss, lower_threshold, upper_threshold):
        y_pred = []
        for i in range(0,len(train_loss)):
            if train_loss[i] < upper_threshold and train_loss[i] > lower_threshold:
                y_pred.append(0)
            else:
                y_pred.append(1)
        return y_pred
    
    @staticmethod
    def validate_live(features, training_path, device, db):
        
        for featurename, corpus in features[1].items():
            timestamp = features[0][0]
            timestamp = int(timestamp)
            output = dbqueries.get_threshold(db, device, featurename)
            # Unpack SQL row:
            STD_lower = float(output[0][0])
            STD_upper = float(output[0][1])
            IQR_lower = float(output[0][2])
            IQR_upper = float(output[0][3])

            # Load scaler:
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)

            corpus = scaler.transform(corpus)
            # Load the trained model:
            with open(training_path + '/models/' + featurename + "autoencoder" + ".pickle", "rb") as f:
                model = pickle.load(f)

            start = time.time()
            reconstruction = model.predict(corpus)
            end = time.time()
            test_time = end - start
            reconstruction = model.predict(corpus)
            loss = mae(corpus, reconstruction)
            y_pred_STD = anomalydl.append_labels(loss, STD_lower, STD_upper)
            y_pred_IQR = anomalydl.append_labels(loss, IQR_lower, IQR_upper)
            dbqueries.insert_into_live(db, device, "anomaly", "autoencoder_STD", featurename, timestamp, int(y_pred_STD[0]), test_time)
            dbqueries.insert_into_live(db, device, "anomaly", "autoencoder_IQR", featurename, timestamp, int(y_pred_IQR[0]), test_time)

        return








    @staticmethod
    def validate(features, training_path, device, db, experiment, behavior):
        for featurename, corpus in features[2].items():
            y = [1 for i in range(0,len(corpus))]
            print("Testing DL: " + featurename)
            output = dbqueries.get_threshold(db, device, featurename)
            if output is None:
                print("No threshold found for " + featurename)
                continue
            # Unpack SQL row:
            STD_lower = float(output[0][0])
            STD_upper = float(output[0][1])
            IQR_lower = float(output[0][2])
            IQR_upper = float(output[0][3])
        
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            
            # Transform the data:
            corpus = scaler.transform(corpus)

            # Load the trained model:
            with open(training_path + '/models/' + featurename + "autoencoder" + ".pickle", "rb") as f:
                model = pickle.load(f)

            reconstruction = model.predict(corpus)
            loss = mae(corpus, reconstruction)
            y_pred_STD = anomalydl.append_labels(loss, STD_lower, STD_upper)
            accuracy_STD = accuracy_score(y, y_pred_STD)
            y_pred_IQR = anomalydl.append_labels(loss, IQR_lower, IQR_upper)
            accuracy_IQR = accuracy_score(y, y_pred_IQR)
            primary_key = dbqueries.get_foreign_key_dl(db, device, featurename, "autoencoder")
            dbqueries.create_dl_anomaly_testing(db, device, experiment, behavior, featurename, "autoencoder",accuracy_STD, accuracy_IQR, primary_key)
        return

    @staticmethod
    def train(features, training_path, device, db) -> dict:
        
        # Checking all paths:
        if not os.path.exists(training_path + '/models/'):
            os.makedirs(training_path + '/models/')

        
        for featurename, corpus in features[2].items():
            y = [0 for i in range(0,len(corpus))]
            print("Training DL: " + featurename)
            X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.1, random_state=42, shuffle=True)

            # Load scaler at training_path + '/scalers/' + featurename + '_scaler.pickle'
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', "rb") as f:
                scaler = pickle.load(f)
            
            # Scale the data:
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Get the diffent variables needed to define the Autoencoder:
            input_dim = X_train.shape[1]
            layer_one = input_dim - input_dim % 10
            layer_two = round(layer_one / 2)
            layer_three = round(layer_two / 2)
            layer_four = round(layer_three / 2)
            hidden_layers = [layer_one, layer_two, layer_three, layer_four, layer_three, layer_two, layer_one, input_dim]

            model = AnomalyDetector(input_dim, layer_one, layer_two, layer_three, layer_four)
            early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
            model.compile(optimizer='adam', loss="mae", metrics=["mae","accuracy"])

            start_training = time.time()
            model.fit(X_train, X_train, epochs=500, batch_size=120, shuffle=True, validation_data=(X_test, X_test), callbacks=[early_stopping], verbose=0)
            end_training = time.time()
            training_time = end_training - start_training

            start_prediction = time.time()
            IQR_lower, IQR_upper, STD_lower, STD_upper, train_loss = anomalydl.find_threshold(model, X_test)
            end_prediction = time.time()
            prediction_time = end_prediction - start_prediction
            
            # Prediction using STD:
            y_pred_STD = anomalydl.append_labels(train_loss, STD_lower, STD_upper)
            # Prediction using IQR:
            y_pred_IQR = anomalydl.append_labels(train_loss, IQR_lower, IQR_upper)

            accuracy_STD = accuracy_score(y_test, y_pred_STD)
            accuracy_IQR = accuracy_score(y_test, y_pred_IQR)

             # Save the metrics:
            with open('output.csv', 'a', newline='') as csvfile:
                fieldnames = ['Name', 'Start', 'End', 'Duration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Name':f'Training autoencoder with feature {featurename}', 'Start': start_training, 'End': end_training, 'Duration': end_training - start_training})
                writer.writerow({'Name':f'Prediction autoencoder with feature {featurename}', 'Start': start_prediction, 'End': end_prediction, 'Duration': end_prediction - start_prediction})
                # Close the file
                csvfile.close()

            # Save the model as a pickle file:
            with open(training_path + '/models/' + featurename + "autoencoder" + ".pickle", "wb") as f:
                pickle.dump(model, f)

            # DB Query:
            dbqueries.create_dl_anomaly(db, device, featurename, "autoencoder", accuracy_STD, accuracy_IQR, STD_lower, STD_upper, IQR_lower, IQR_upper, str(hidden_layers), training_time, prediction_time)
            
        return

            
    

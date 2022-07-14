import os
from random import shuffle
import sys
import time
import pickle
import numpy as np
import pandas as pd
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
    def find_threshold(model, X_test, feature):
        print("Predict autoencoder..." + feature + str(time.time()))
        reconstruction = model.predict(X_test)
        print("Predict autoencoder...done" + feature + str(time.time()))
        train_loss = mae(X_test, reconstruction)
        threshold = np.mean(train_loss) + 2*np.std(train_loss)
        return threshold, train_loss

    @staticmethod
    def append_labels(train_loss, threshold):
        y_pred = []
        for i in range(0,len(train_loss)):
            if train_loss[i] < threshold:
                y_pred.append(0)
            else:
                y_pred.append(1)
        return y_pred
    
    @staticmethod
    def validate_live(df, train_path, feature, db, device, threshold):
        model_list = []
        X = df[feature].tolist()
        X = np.array(X)
        timestamp = df['timestamps'].tolist()
        time_stamp = timestamp[0]
        y = [1]
         
        # Load the trained model:
        with open(train_path + "/MLmodels/" + feature + "autoencoder.pickle", "rb") as f:
            model = pickle.load(f)
        
        t1 = time.time()
        reconstruction = model.predict(X)
        t2 = time.time()
        test_time = t2 - t1
        loss = mae(X, reconstruction)
        # Get the y_pred:
        y_pred = anomalydl.append_labels(loss, threshold)
        y_pred = y_pred[0]
        y_pred = int(y_pred)
        dbqueries.insert_into_live(db, device, "anomaly", "autoencoder", feature, time_stamp, y_pred, test_time)


    @staticmethod
    def validate(features, training_path, device, db, experiment, behavior):
        for featurename, corpus in features[2].items():
            y = [1 for i in range(0,len(corpus))]
            print("Testing ML: " + featurename)
            threshhold = dbqueries.get_threshold(db, device, featurename)
            threshhold = float(threshhold)
            
            with open(training_path + '/scalers/' + featurename + '_scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            
            # Transform the data:
            corpus = scaler.transform(corpus)

            # Load the trained model:
            with open(training_path + '/models/' + featurename + "autoencoder" + ".pickle", "rb") as f:
                model = pickle.load(f)

            reconstruction = model.predict(corpus)
            loss = mae(corpus, reconstruction)
            y_pred = anomalydl.append_labels(loss, threshhold)
            accuracy = accuracy_score(y, y_pred)
            primary_key = dbqueries.get_foreign_key_dl(db, device, featurename, "autoencoder")
            dbqueries.create_dl_anomaly_testing(db, device, experiment, behavior, featurename, "autoencoder",accuracy, primary_key)
        return

    @staticmethod
    def train(features, training_path, device, db) -> dict:
        
        # Checking all paths:
        if not os.path.exists(training_path + '/models/'):
            os.makedirs(training_path + '/models/')

        
        for featurename, corpus in features[2].items():
            y = [0 for i in range(0,len(corpus))]
            print("Training DL: " + featurename)
            X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.3, random_state=42, shuffle=False)

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

            print("Start training autoencoder " + featurename + " " + str(time.time()))
            model.fit(X_train, X_train, epochs=500, batch_size=120, shuffle=True, validation_data=(X_test, X_test), callbacks=[early_stopping], verbose=0)
            print("End training autoencoder " + featurename + " " + str(time.time()))

            threshold, train_loss = anomalydl.find_threshold(model, X_test, featurename)
            y_pred = anomalydl.append_labels(train_loss, threshold)
            accuracy = accuracy_score(y_test, y_pred)

            # Save the model as a pickle file:
            with open(training_path + '/models/' + featurename + "autoencoder" + ".pickle", "wb") as f:
                pickle.dump(model, f)

            # DB Query:
            dbqueries.create_dl_anomaly(db, device, featurename, "autoencoder", accuracy, threshold, str(hidden_layers))

        return

            
    

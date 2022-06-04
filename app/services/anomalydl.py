import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    # Classiclass anomalydl:
    """
    This class contains the functions for anomaly detection.
    Uses scikit learn and tensorflow for the anomaly detection with deep learning.
    In this case uses Autoencoders to do the anomaly detection.
    """

    @staticmethod
    def find_threshold(model, X_test):
        """
        This function finds the threshold for the anomaly detection.
        :input: model: trained model, X_test: test data
        :output: threshold: float, train_loss: float
        """
        reconstruction = model.predict(X_test)
        train_loss = mae(X_test, reconstruction)
        threshold = np.mean(train_loss) + 2*np.std(train_loss)
        return threshold, train_loss

    @staticmethod
    def append_labels(train_loss, threshold):
        """
        This function takes the reconstruction of the test data and the threshold
        and returns a list of labels for each sample.
        If the reconstruction is above the threshold, the sample is considered
        an anomaly (-1) else it is considered normal (1).
        :input: reconstruction: the reconstruction of the test data (numpy array) and
        threshold: the threshold for the anomaly detection (float)
        """
        y_pred = []
        for i in range(0,len(train_loss)):
            if train_loss[i] < threshold:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return y_pred

    @staticmethod
    def validate(df: pd.DataFrame, train_path: str, feature: str, threshold: float):
        """
        This function validates the anomaly detection.
        :input: df: the dataframe with the test data, train_path: the path to the training data,
        feature: the feature to be used for the anomaly detection, threshold: the threshold for the anomaly detection
        :output: y_pred: the labels for the test data
        """
        # Returned dict:
        model_list = []

        # Create a list from the last column of the dataframe
        # And create a numpy array:
        X = df[feature].tolist()
        X = np.array(X)

        # Labels y for later use to calculate TPR:
        y = [-1 for i in range(0,len(X))]
        
        # Load the trained model:
        with open(train_path + "/MLmodels/" + feature + "autoencoder.pickle", "rb") as f:
            model = pickle.load(f)
        
        t1 = time.time()
        reconstruction = model.predict(X)
        t2 = time.time()
        loss = mae(X, reconstruction)

        # Get the y_pred:
        y_pred = anomalydl.append_labels(loss, threshold)

        # Calculate the TPR:
        accuracy = accuracy_score(y, y_pred)
        model_list.append(accuracy)
        model_list.append(t2-t1)

        return model_list


    @staticmethod
    def train(df: pd.DataFrame, train_path: str, feature: str) -> dict:
        """
        This function trains the model.
        :input: df: Dataframe, train_path: str, feature: str
        :output: trained model saved in a pickle file.
        Threshold = 2std deviations from the mean of the training loss.
        """
        # Returned dict:
        model_dict = {}
        ml_path = train_path + '/MLmodels/'

        # Checks if the directory exists:
        if not os.path.exists(ml_path):
            os.makedirs(ml_path)
        
        # Create a list from the last column of the dataframe
        # And create a numpy array:
        X = df[feature].tolist()
        X = np.array(X)

        # Labels y for later use to calculate FPR:
        y = [1 for i in range(0,len(X))]

        # Create a train-test split:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
        

        # Get the diffent variables needed to define the Autoencoder:
        input_dim = X_train.shape[1]
        layer_one = input_dim - input_dim % 10
        layer_two = round(layer_one / 2)
        layer_three = round(layer_two / 2)
        layer_four = round(layer_three / 2)
       
        hidden_layers = [layer_one, layer_two, layer_three, layer_four, layer_three, layer_two, layer_one, input_dim]
        model_dict['hidden_layers'] = hidden_layers

        # Create the Autoencoder:
        model = AnomalyDetector(input_dim, layer_one, layer_two, layer_three, layer_four)

        # Create callback to stop training if the model does not improve:
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
        model.compile(optimizer='adam', loss="mae", metrics=["mae","accuracy"])
        t1 = time.time()
        model.fit(X_train, X_train, epochs=500, batch_size=120, shuffle=True, validation_data=(X_val, X_val), callbacks=[early_stopping], verbose=0)
        t2=time.time()
        model_dict['training_time'] = t2 - t1

        # Calculate the train loss and threshold:
        threshold, train_loss = anomalydl.find_threshold(model, X_val)
        model_dict['threshhold'] = threshold

        # Get the y_pred:
        y_pred = anomalydl.append_labels(train_loss, threshold)

        # Calculate the accuracy:
        accuracy = accuracy_score(y_val, y_pred)
        model_dict['TNR'] = accuracy

        # Save the model as a pickle file:
        with open(ml_path + feature + 'autoencoder.pickle', 'wb') as f:
            pickle.dump(model, f)
        
        return model_dict

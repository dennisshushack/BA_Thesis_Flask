import os 
import sys
import pickle
import numpy as np
import pandas as pd
import time 
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Specially made for live monitoring:
class m1live:
    @staticmethod
    def clean_data(input_dirs: dict, number:int) -> pd.DataFrame:
        # Cleans the data for monitor 1:
        df = pd.DataFrame()
        timestamps = []
        for key, value in input_dirs.items():
            input_dir = value + "/m1"
            files = os.listdir(input_dir)
            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                # Get the row at index number as a dataframe:
                file_df = file_df.iloc[[number], :]
                timestamps.extend(file_df['time'])
                file_df.drop('time', axis=1, inplace=True)
                file_df.drop('seconds',axis=1, inplace=True)
                # Append the dataframe to the dataframe df
                df = df.append(file_df) 
        # Sort columns of df by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        return df, timestamps

    @staticmethod
    def load_scaler_and_standardize(df:pd.DataFrame, training_dir: str):
        """
        Loads the scaler and standardizes the dataframe
        """
        # Load the scaler from the training directory:
        with open(training_dir + "/scalers/m1_scaler.pickle", 'rb') as f:
            scaler = pickle.load(f)
        # Standardize the dataframe:
        standardized_df_numpy = scaler.transform(df)
        return standardized_df_numpy

    @staticmethod
    def preprocess_data(input_dirs: dict, training_dirs: list, number:int) -> pd.DataFrame:
        # Does data preprocessing for monitor 1: (anomaly & classification)
        return_dfs = []
        try:
            start_time = time.time()
            print(f"Preprocessing data for monitor 1... {start_time}")
            df, timestamps = m1live.clean_data(input_dirs, number)
            print("Standardizing data...")
            m1live_standardized_anomaly = m1live.load_scaler_and_standardize(df, training_dirs[0])
            m1live_standardized_classification = m1live.load_scaler_and_standardize(df, training_dirs[1])

            features_anomaly = []
            features_classification = []
            features_anomaly.append(timestamps)
            features_classification.append(timestamps)
            features_anomaly.append(m1live_standardized_anomaly)
            features_classification.append(m1live_standardized_classification)

            m1_preprocessed_anomaly = pd.DataFrame(features_anomaly).transpose()
            m1_preprocessed_classification = pd.DataFrame(features_classification).transpose()
            m1_preprocessed_anomaly.columns = ["timestamps", "m1"]
            m1_preprocessed_classification.columns = ["timestamps", "m1"]

            return_dfs.append(m1_preprocessed_anomaly)
            return_dfs.append(m1_preprocessed_classification)
            end_time = time.time()
            print(f"Finished preprocessing data for monitor 1...{end_time}")
            print(f"Time taken: {end_time - start_time}")
            return return_dfs
        except:
            return None


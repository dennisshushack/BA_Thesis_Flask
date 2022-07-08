import os 
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Specially made for live monitoring:
class m2live:
    @staticmethod
    def clean_data(input_dirs: dict, number:int) -> pd.DataFrame:
        # Cleans the data for monitor 1:
        df = pd.DataFrame()
        timestamps = []
        for key, value in input_dirs.items():
            input_dir = value + "/m2"
            files = os.listdir(input_dir)
            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                # Get the row at index number as a dataframe:
                file_df = file_df.iloc[[number], :]
                timestamps.extend(file_df['timestamp'])
                file_df.drop('timestamp', axis=1, inplace=True)
                file_df.drop('time', axis=1, inplace=True)
                file_df.drop('seconds',axis=1, inplace=True)
                file_df.drop('connectivity',axis=1, inplace=True)
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
        with open(training_dir + "/scalers/m2_scaler.pickle", 'rb') as f:
            scaler = pickle.load(f)
        # Standardize the dataframe:
        standardized_df_numpy = scaler.transform(df)
        return standardized_df_numpy

    @staticmethod
    def preprocess_data(input_dirs: dict, training_dirs: list, number:int) -> pd.DataFrame:
        # Does data preprocessing for monitor 1: (anomaly & classification)
        try:
            start_time = time.time()
            print(f"Preprocessing data for monitor 2... {start_time}")
            return_dfs = []
            df, timestamps = m2live.clean_data(input_dirs, number)
            print("Standardizing data...")
            m2live_standardized_anomaly = m2live.load_scaler_and_standardize(df, training_dirs[0])
            m2live_standardized_classification = m2live.load_scaler_and_standardize(df, training_dirs[1])

            features_anomaly = []
            features_classification = []
            features_anomaly.append(timestamps)
            features_classification.append(timestamps)
            features_anomaly.append(m2live_standardized_anomaly)
            features_classification.append(m2live_standardized_classification)

            m2_preprocessed_anomaly = pd.DataFrame(features_anomaly).transpose()
            m2_preprocessed_classification = pd.DataFrame(features_classification).transpose()
            m2_preprocessed_anomaly.columns = ["timestamps", "m2"]
            m2_preprocessed_classification.columns = ["timestamps", "m2"]

            return_dfs.append(m2_preprocessed_anomaly)
            return_dfs.append(m2_preprocessed_classification)
            end_time = time.time()
            print(f"Preprocessing data for monitor 2... Done! {end_time}")
            time_taken = end_time - start_time
            print(f"Time taken to preprocess data for monitor 2: {time_taken}")
            return return_dfs
        except:
            return None


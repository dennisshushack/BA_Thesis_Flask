import os 
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# M1 preprocess (Anomaly & Classification)
class m1: 
    @staticmethod
    def clean_data(input_dirs: dict) -> pd.DataFrame:
        # Cleans the data for monitor 1:
        df = pd.DataFrame()
        vector_behavior = []
        ids = []
        timestamps = []
        for key, value in input_dirs.items():
            input_dir = value + "/m1"
            behavior = key
            files = os.listdir(input_dir)
            for file in files:
                # Read the file:
                file_df = pd.read_csv(input_dir + "/" + file)
                # Drop all temporary features:
                timestamps.extend(file_df['time'])
                file_df.drop('time', axis=1, inplace=True)
                file_df.drop('seconds', axis=1, inplace=True)
                # Drop all rows with NaN values:
                file_df.dropna(inplace=True)
                # Drop all rows in the dataframe that contain infinity values:
                file_df.drop(file_df[file_df.values == np.inf].index, inplace=True)
                # Check if there are any duplicate rows:
                file_df.drop_duplicates(inplace=True)
                # Add the behavior to the vector_behavior list for every row:
                vector_behavior.extend([behavior] * len(file_df))
                # Add the file name to the ids list for every row:
                ids.extend([file] * len(file_df))
                # Append the dataframe to the dataframe df
                df = df.append(file_df) 
        # Sort columns of df by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        # Drop any NaN values:
        df.dropna(inplace=True)
        return df, vector_behavior, ids, timestamps
    
    @staticmethod
    def create_scaler(df:pd.DataFrame, training_dir: str):
        """
        Creates a scaler for the dataframe
        """
        if os.path.exists(training_dir + "/scalers/m1_scaler.pickle"):
            scaler = pickle.load(open(training_dir + "/scalers/m1_scaler.pickle", "rb"))
            return scaler
        scaler = StandardScaler()
        # Create a train-test split:
        X_train, X_val = train_test_split(df, test_size=.3, shuffle=True, random_state=42)         
        # Only fit to the training data:
        scaler.fit(X_train)
        # Create a directory for the scaler if it does not exist:
        output_dir = training_dir + "/scalers/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the scaler to a pickle file in the output directory:
        with open(output_dir + "m1_scaler.pickle", 'wb') as f:
            pickle.dump(scaler, f)
        return scaler
    
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
    def preprocess_data(input_dirs: dict, category: str, training_dirs: list, path:str) -> pd.DataFrame:
        """
        Main function to preprocess the data for monitor 1 (Collected data not live data)
        """
        
        if category == "training":
            # 1. Procedure for training (anomaly or classification)
            df, vector_behavior, ids, timestamps = m1.clean_data(input_dirs)
            scaler = m1.create_scaler(df, training_dirs[0])
            m1_standardized = m1.load_scaler_and_standardize(df, training_dirs[0])
            features = []
            features.append(ids)
            features.append(timestamps)
            features.append(vector_behavior)
            features.append(m1_standardized)
            m1_preprocessed = pd.DataFrame(features).transpose()
            m1_preprocessed.columns = ["id", "timestamps", "behavior", "m1"]
            m1_preprocessed.dropna(inplace=True)
            
            # output directory for the preprocessed data:
            output_dir = training_dirs[0] + "/preprocessed/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the preprocessed data to a pickle file in the output directory:
            m1_preprocessed.to_csv(output_dir + "m1_preprocessed.csv", index=False)
            

        elif category == "testing":
            df, vector_behavior, ids, timestamps = m1.clean_data(input_dirs)
           
            # Output directory for the preprocessed data:
            output_dir =  path + "/preprocessed/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            m1_standardized = m1.load_scaler_and_standardize(df, training_dirs[0])
            features = []
            features.append(ids)
            features.append(timestamps)
            features.append(vector_behavior)
            features.append(m1_standardized)
            m1_preprocessed = pd.DataFrame(features).transpose()
            m1_preprocessed.columns = ["id", "timestamps", "behavior", "m1"]
            m1_preprocessed.dropna(inplace=True)
            m1_preprocessed.to_csv(output_dir + "m1_preprocessed.csv", index=False)

        return m1_preprocessed

            


        


import os 
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
# M2 (Anomaly & Classification)
class m2:
    """
    Class for data concerning monitor 2 (Ressource Usage & HPC)
    Creates a cleaned & standardized dataframe ready for training/evaluation
    for classificaiton or anomaly detection
    """
    @staticmethod
    def clean_data(input_dirs: dict, begin:int = None, end:int = None) -> pd.DataFrame:
        """
        Cleans the data for monitor 2 HPC & Ressource Usage (one  or multiple .csv files to clean)
        :input: input_dirs: dict with the paths to the input directories where raw data (.csv) are stored with
        the corresponding behavior i.e {'path1':'normal','path2': ransom1}
        :output: a dataframe with the cleaned data, vector_behavior: vector with the behavior of the dataframe, 
        ids: vector with the ids of each file in the dataframe
        """
        # Dataframe to store the cleaned data
        df = pd.DataFrame()
        vector_behavior = []
        ids = []
        timestamps = []
        # Iterate over the dictionary input_dirs
        for key, value in input_dirs.items():
            input_dir = value + "/m2"
            behavior = key
            files = os.listdir(input_dir)
            for file in files:
                    
                file_df = pd.read_csv(input_dir + "/" + file)

                if begin is not None and end is not None:
                    file_df = file_df.loc[(file_df['timestamp'] >= begin) & (file_df['timestamp'] <= end)]

                # Drop all temporary features:
                file_df.drop('seconds', axis=1, inplace=True)
                file_df.drop('time', axis=1, inplace=True)
                timestamps.extend(file_df['timestamp'])
                
                # Keep only values with connectivity == 1 & remove connectivity
                file_df = file_df.loc[file_df['connectivity'] == 1]
                file_df.drop(['connectivity'], axis=1, inplace=True)

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

        # Sort the dataframe by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)

        # Sort by timestamp:
        df.sort_values(by=['timestamp'], inplace=True)
        df.drop('timestamp', axis=1, inplace=True)

        return df, vector_behavior, ids, timestamps
        
    @staticmethod
    def create_scaler(df:pd.DataFrame, training_dir: str):
        """
        Creates a scaler for the dataframe
        :input: df: dataframe with the data to scale, training_dir: path to the directory where the scaler is stored
        :output: scaler: scaler object
        """
        
        # Checks if the scaler already exists if so return it:
        if os.path.exists(training_dir + "/scalers/m2_scaler.pickle"):
            scaler = pickle.load(open(training_dir + "/scalers/m2_scaler.pickle", "rb"))
            return scaler

        scaler = StandardScaler()

        # Create a train-test split:
        X_train, X_val = train_test_split(df, test_size=.3, shuffle=True, random_state=42)         
        
        # Only fit to the training data:
        scaler.fit(X_train)

        # Create a directory for the scaler if it does not exist:
        output_dir = training_dir + "/scalers"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the scaler to a pickle file in the output directory:
        with open(output_dir + "/m2_scaler.pickle", 'wb') as f:
            pickle.dump(scaler, f)
        
        return scaler

    @staticmethod
    def load_scaler_and_standardize(df:pd.DataFrame, training_dir: str):
        """
        This function loads a scaler and normalizes the dataframe
        :input: df: dataframe, training_dir: path to the directory where the scaler is stored
        :output: df: dataframe with the normalized data
        """
        # Load the scaler from the training directory:
        with open(training_dir + "/scalers/m2_scaler.pickle", 'rb') as f:
            scaler = pickle.load(f)
        
        # Normalize the dataframe:
        standardized_df_numpy = scaler.transform(df)
        
        return standardized_df_numpy
        

    @staticmethod
    def preprocess_data(df: pd.DataFrame, behaviors: list, ids: list , category: str, training_dir: str, timestamps: list, testing_dir: str = None):
        """
        This function standardizes all the data in the dataframe.
        If the scaler already exists, it loads it from the input directory else it creates a new scaler and saves it.
        :input: df: The cleaned dataframe, category: category of the dataframe, training_dir: Where the scaler is saved
        or will be created, behaviors: vector with the behavior of the dataframe, ids: vector with the ids of each file in the dataframe
        :output: preprocessed dataframe ready for ML/DL models training/evaluation
        """
        # Check if the dataframe is training data:
        if category == "training":
            # Create a scaler if it does not exist:
            m2.create_scaler(df, training_dir)
            # Load the scaler from the training directory and normalize the dataframe:
            m2_standardized = m2.load_scaler_and_standardize(df, training_dir)
        else:
            m2_standardized = m2.load_scaler_and_standardize(df, training_dir)

        # Iterate through the array:
        features = []
        
        features.append(ids)
        features.append(timestamps)
        features.append(behaviors)
        features.append(m2_standardized)

        # Create a dataframe with the ids and the behavior and join it with the normalized dataframe:
        m2_preprocessed = pd.DataFrame(features).transpose()
        m2_preprocessed.columns = ["id", "timestamps", "behavior", "m2"]

           # Create directory features if it does not exist:
        if category == "training":
            output_dir = training_dir + "/preprocessed/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the dataframe as a csv file in the output directory:
            m2_preprocessed.to_csv(output_dir + "m2_preprocessed.csv", index=False)
        else:
            output_dir = testing_dir + "/preprocessed/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the dataframe as a csv file in the output directory:
            m2_preprocessed.to_csv(output_dir + "m2_preprocessed.csv", index=False)
            
        return m2_preprocessed


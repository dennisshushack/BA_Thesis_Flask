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
    """
    Cleans the data for monitor 1 (one or more .csv files to clean)
    :input: input_dir: path to the input directory where raw data (.csv) is stored,
    :output: a dataframe with the cleaned data ready for ML&DL Classification and anomaly detection
    """
    @staticmethod
    def clean_data(input_dirs: dict, begin:int = None, end: int = None ) -> pd.DataFrame:
        """
        Cleans the data for monitor 1 (one  or multiple .csv file to clean)
        :input: input_dirs: dict with the paths to the input directories where raw data (.csv) are stored with
        the corresponding behavior i.e {'path1':'normal','path2': ransom1}
        :output: a dataframe with the cleaned data
        """
        # Dataframe to store the cleaned data
        df = pd.DataFrame()
        # Numpy array to store the behavior
        vector_behavior = []
        ids = []
        # Iterate over the dictionary input_dirs
        for key, value in input_dirs.items():
            input_dir = value + "/m1"
            behavior = key
            files = os.listdir(input_dir)
            # Iterate over the files in the input directory
            for file in files:

                file_df = pd.read_csv(input_dir + "/" + file)
                
                if begin is not None and end is not None:
                    file_df = file_df.loc[(file_df['time'] >= begin) & (file_df['time'] <= end)]
                
                file_df.drop('time', axis=1, inplace=True)
                
                # Drop all temporary features:
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
                

        return df, vector_behavior, ids
    
    @staticmethod
    def create_scaler(df:pd.DataFrame, training_dir: str):
        """
        Creates a scaler for the dataframe
        :input: df: dataframe with the data to scale, training_dir: path to the directory where the scaler is stored
        :output: scaler: scaler object
        """
        
        # Checks if the scaler already exists if so return it:
        if os.path.exists(training_dir + "/scalers/m1_scaler.pickle"):
            scaler = pickle.load(open(training_dir + "/scalers/m1_scaler.pickle", "rb"))
            return scaler

        scaler = StandardScaler()

        # Create a train-test split:
        X_train, X_val = train_test_split(df, test_size=.3, shuffle=False, random_state=42)         
        
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
        This function loads a scaler and normalizes the dataframe
        :input: df: dataframe, training_dir: path to the directory where the scaler is stored
        :output: df: dataframe with the normalized data
        """
        # Load the scaler from the training directory:
        with open(training_dir + "/scalers/m1_scaler.pickle", 'rb') as f:
            scaler = pickle.load(f)
        
        # Standardize the dataframe:
        standardized_df_numpy = scaler.transform(df)
        
        return standardized_df_numpy


    @staticmethod
    def preprocess_data(df: pd.DataFrame, behaviors: list, ids: list , category: str, training_dir: str, testing_dir: str = None) -> pd.DataFrame:
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
            m1.create_scaler(df, training_dir)
            # Load the scaler from the training directory and normalize the dataframe:
            m1_standardized = m1.load_scaler_and_standardize(df, training_dir)
        else:
            m1_standardized = m1.load_scaler_and_standardize(df, training_dir)

        # Iterate through the array:
        features = []
        
        features.append(ids)
        features.append(behaviors)
        features.append(m1_standardized)

        # Create a dataframe with the ids and the behavior and join it with the normalized dataframe:
        m1_preprocessed = pd.DataFrame(features).transpose()
        m1_preprocessed.columns = ["id", "behavior", "m1"]
        
        # Create directory features if it does not exist:
        if category == "training":
            output_dir = training_dir + "/preprocessed/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the dataframe as a csv file in the output directory:
            m1_preprocessed.to_csv(output_dir + "m1_preprocessed.csv", index=False)
        else:
            output_dir = testing_dir + "/preprocessed/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the dataframe as a csv file in the output directory:
            m1_preprocessed.to_csv(output_dir + "m1_preprocessed.csv", index=False)
        
        return m1_preprocessed


import os 
import sys
import numpy as np
import pandas as pd
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# KERN (Anomaly & Classification)
class KERN:
    @staticmethod
    def preprocess_data(input_dirs: dict) -> pd.DataFrame:
        """
        Main function to preprocess the data for monitor 2 (Collected data not live data)
        """
        features = []
        df = pd.DataFrame()
        vector_behavior = []
        timestamps = []

        for key, value in input_dirs.items():
            input_dir = value + "/KERN"
            behavior = key
            files = os.listdir(input_dir)
            for file in files:
                
                # Normal pre-processing:
                file_df = pd.read_csv(input_dir + "/" + file)
                file_df = file_df[file_df.columns.sort_values()]
                file_df.dropna(inplace=True)
                file_df.drop(file_df[file_df.values == np.inf].index, inplace=True)
                file_df.drop_duplicates(inplace=True)

                # Removing temporary columns:
                timestamps.extend(file_df['timestamp'])
                file_df.drop('timestamp', axis=1, inplace=True)
                file_df.drop('seconds', axis=1, inplace=True)
                file_df.drop('time', axis=1, inplace=True)
                file_df = file_df.loc[file_df['connectivity'] == 1]
                file_df.drop('connectivity', axis=1, inplace=True)
                vector_behavior.extend([behavior] * len(file_df))

                # Feature engineering:
                file_df.drop('cs', axis=1, inplace=True)
                df = df.append(file_df) 
                # Remove df from memory:
                del file_df
                
        # Sort the dataframe by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        # Drop any NaN values:
        df.dropna(inplace=True)
        # Check, that timestamp and vector_behavior are the same length:s         
        features.append(timestamps)
        features.append(vector_behavior)
        KERN = {"KERN": df}
        features.append(KERN)
        return features
            
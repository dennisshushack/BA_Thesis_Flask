import os 
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Specially made for live monitoring:
class KERNlive:

    @staticmethod
    def preprocess_data(input_dirs: dict, number:int) -> pd.DataFrame:
        # Does data preprocessing for monitor 1: (anomaly & classification)
        features = []
        df = pd.DataFrame()
        timestamps = []
        for key, value in input_dirs.items():
            input_dir = value + "/KERN"
            files = os.listdir(input_dir)
            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                file_df = file_df.iloc[[number]]
                timestamps.extend(file_df['timestamp'])
                file_df.drop('timestamp', axis=1, inplace=True)
                file_df.drop('time', axis=1, inplace=True)
                file_df.drop('seconds',axis=1, inplace=True)
                file_df.drop('connectivity',axis=1, inplace=True)
                file_df.drop('cs', axis=1, inplace=True)
                df = df.append(file_df) 

        # Sort columns of df by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        features.append(timestamps)
        KERN = {"KERN": df}
        features.append(KERN)
        return features



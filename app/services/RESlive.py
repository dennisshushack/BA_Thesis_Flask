import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Specially made for live monitoring:
class RESlive:

    @staticmethod
    def preprocess_data(input_dirs: dict, number:int) -> pd.DataFrame:
        """
        Returns the preprocessed data for live monitoring.
        """
        features = []
        timestamps = []
        df = pd.DataFrame()

        for key, value in input_dirs.items():
            input_dir = value + "/RES"
            files = os.listdir(input_dir)
            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                file_df = file_df.iloc[[number]]
                timestamps.extend(file_df['time'])
                file_df.drop('memory', axis=1, inplace=True)
                file_df.drop('seconds', axis=1, inplace=True)
                file_df.drop('time', axis=1, inplace=True)
                df = df.append(file_df) 


        # Sort columns of df by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        features.append(timestamps)
        RES = {"RES": df}
        features.append(RES)
        return features

    
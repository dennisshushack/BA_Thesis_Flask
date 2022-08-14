import os 
import sys
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")


###############################################################################
#                              Cleaning Data                                  #
###############################################################################
class RES:
    @staticmethod
    def preprocess_data(input_dirs: dict) -> pd.DataFrame:
        """
        Main function to preprocess the data for monitor 1 (Collected data not live data)
        """
        features = []
        df = pd.DataFrame()
        vector_behavior = []
        timestamps = []
        
        for key, value in input_dirs.items():
            input_dir = value + "/RES"
            behavior = key
            files = os.listdir(input_dir)

            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                file_df = file_df[file_df.columns.sort_values()]
                
                # Normal pre-processing:
                file_df.dropna(inplace=True)
                file_df.drop(file_df[file_df.values == np.inf].index, inplace=True)
                file_df.drop_duplicates(inplace=True)

                # Dropping temporary columns:
                timestamps.extend(file_df['time'])
                vector_behavior.extend([behavior] * len(file_df))
                
                # Feature engineering:
                file_df.drop('memory', axis=1, inplace=True)
                # Drop seconds column:
                file_df.drop('seconds', axis=1, inplace=True)
                # Drop time:
                file_df.drop('time', axis=1, inplace=True)
                df = df.append(file_df) 
                # Remove df from memory:
                del file_df

        df = df.reindex(sorted(df.columns), axis=1)
        df.dropna(inplace=True)
        features.append(timestamps)
        features.append(vector_behavior)
        RES = {"RES": df}
        features.append(RES)
        return features
            
            
            
            
            
            
            
       
            

    


        


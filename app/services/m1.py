import os 
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")


###############################################################################
#                              Cleaning Data                                  #
###############################################################################
class m1:
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
            input_dir = value + "/m1"
            behavior = key
            files = os.listdir(input_dir)

            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                
                # Normal pre-processing:
                file_df.dropna(inplace=True)
                file_df.drop(file_df[file_df.values == np.inf].index, inplace=True)
                file_df.drop_duplicates(inplace=True)

                # Dropping temporary columns:
                timestamps.extend(file_df['time'])
                file_df.drop('time', axis=1, inplace=True)
                file_df.drop('seconds', axis=1, inplace=True)
                vector_behavior.extend([behavior] * len(file_df))
                
                # Feature engineering:
                file_df.drop('memory',inplace=True,axis=1)
                file_df.drop('branch-instructions',inplace=True,axis=1)
                file_df.drop('branch-misses',inplace=True,axis=1)
                file_df.drop('bus-cycles',inplace=True,axis=1)
                file_df.drop('cache-references',inplace=True,axis=1)
                file_df.drop('cpu-cycles',inplace=True,axis=1)
                file_df.drop('instructions',inplace=True,axis=1)
                file_df.drop('L1-dcache-loads',inplace=True,axis=1)
                file_df.drop('L1-dcache-stores',inplace=True,axis=1)
                file_df.drop('L1-icache-loads',inplace=True,axis=1)
                file_df.drop('branch-load-misses',inplace=True,axis=1)
                file_df.drop('branch-loads',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/br_immed_retired/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/br_mis_pred/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/br_pred/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/bus_cycles/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/cpu_cycles/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/inst_retired/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/l1d_cache/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/l1i_cache/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/mem_access/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/pc_write_retired/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/st_retired/',inplace=True,axis=1)
                file_df.drop('armv7_cortex_a7/bus_cycles/.1',inplace=True,axis=1)                
                df = df.append(file_df) 

        df = df.reindex(sorted(df.columns), axis=1)
        df.dropna(inplace=True)
        features.append(timestamps)
        features.append(vector_behavior)
        m1 = {"m1": df}
        features.append(m1)
        return features
            
            
            
            
            
            
            
       
            

    


        


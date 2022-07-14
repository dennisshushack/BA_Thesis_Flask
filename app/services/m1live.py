import os
from re import T 
import sys
import pickle
import numpy as np
import pandas as pd
import time 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Specially made for live monitoring:
class m1live:

    @staticmethod
    def preprocess_data(input_dirs: dict, number:int) -> pd.DataFrame:
        features = []
        timestamps = []
        df = pd.DataFrame()
        for key, value in input_dirs.items():
            input_dir = value + "/m1"
            files = os.listdir(input_dir)
            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                file_df = file_df.iloc[[number], :]
                timestamps.extend(file_df['time'])
                file_df.drop('time', axis=1, inplace=True)
                file_df.drop('seconds',axis=1, inplace=True)

                # Feature Engineering:
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
                # Append the dataframe to the dataframe df
                df = df.append(file_df) 

        # Sort columns of df by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        features.append(timestamps)
        m1 = {"m1": df}
        features.append(m1)
        return features

    
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
                vector_behavior.extend([behavior] * len(file_df))
                
                # Feature engineering:
                file_df.drop(['time','cpu', 'cpu-migrations','seconds','memory','timer:tick_stop','sched:sched_waking','sched:sched_stat_runtime','timer:timer_init','branch-instructions','branch-misses','branch-loads', 'bus-cycles','cache-misses','cache-references','cpu-cycles','instructions','context-switches','L1-dcache-load-misses','L1-dcache-loads','L1-dcache-store-misses','L1-dcache-stores','L1-icache-load-misses','L1-icache-loads','LLC-loads','LLC-stores','branch-load-misses','raw_syscalls:sys_enter','random:credit_entropy_bits'],axis=1,inplace=True)
                file_df.rename(columns={'armv7_cortex_a7/l2d_cache_wb/':'armv7_1','armv7_cortex_a7/l1i_tlb_refill/':'armv7_2'},inplace=True)
                file_df.drop(file_df.columns[file_df.columns.str.startswith('armv7_cortex_a7')],axis=1,inplace=True)

                df = df.append(file_df) 

        df = df.reindex(sorted(df.columns), axis=1)
        df.dropna(inplace=True)
        features.append(timestamps)
        features.append(vector_behavior)
        m1 = {"m1": df}
        features.append(m1)
        return features
            
            
            
            
            
            
            
       
            

    


        


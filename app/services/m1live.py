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
         

                # Feature Engineering:
                file_df.drop(['time','cpu', 'cpu-migrations','seconds','memory','timer:tick_stop','sched:sched_waking','sched:sched_stat_runtime','timer:timer_init','branch-instructions','branch-misses','branch-loads', 'bus-cycles','cache-misses','cache-references','cpu-cycles','instructions','context-switches','L1-dcache-load-misses','L1-dcache-loads','L1-dcache-store-misses','L1-dcache-stores','L1-icache-load-misses','L1-icache-loads','LLC-loads','LLC-stores','branch-load-misses','raw_syscalls:sys_enter','random:credit_entropy_bits'],axis=1,inplace=True)
                file_df.rename(columns={'armv7_cortex_a7/l2d_cache_wb/':'armv7_1','armv7_cortex_a7/l1i_tlb_refill/':'armv7_2'},inplace=True)
                file_df.drop(file_df.columns[file_df.columns.str.startswith('armv7_cortex_a7')],axis=1,inplace=True)
                # Append the dataframe to the dataframe df
                df = df.append(file_df) 

        # Sort columns of df by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        features.append(timestamps)
        m1 = {"m1": df}
        features.append(m1)
        return features

    
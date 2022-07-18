import os 
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Specially made for live monitoring:
class m2live:

    @staticmethod
    def preprocess_data(input_dirs: dict, number:int) -> pd.DataFrame:
        # Does data preprocessing for monitor 1: (anomaly & classification)
        features = []
        df = pd.DataFrame()
        timestamps = []
        for key, value in input_dirs.items():
            input_dir = value + "/m2"
            files = os.listdir(input_dir)
            for file in files:
                file_df = pd.read_csv(input_dir + "/" + file)
                # Get the row at index number as a dataframe:
                file_df = file_df.iloc[[number], :]
                timestamps.extend(file_df['timestamp'])
                file_df.drop('timestamp', axis=1, inplace=True)
                file_df.drop('time', axis=1, inplace=True)
                file_df.drop('seconds',axis=1, inplace=True)
                file_df.drop('connectivity',axis=1, inplace=True)

                # Feature engineering:
                file_df.drop('raw_syscalls:sys_enter', axis=1, inplace=True)
                file_df.drop('raw_syscalls:sys_exit', axis=1, inplace=True)
                file_df.drop('sched:sched_switch', axis=1, inplace=True)
                file_df.drop('sched:sched_wakeup', axis=1, inplace=True)
                file_df.drop('irq:irq_handler_entry', axis=1, inplace=True)
                file_df.drop('preemptirq:irq_enable', axis=1, inplace=True)
                file_df.drop('timer:hrtimer_start', axis=1, inplace=True)
                file_df.drop('random:mix_pool_bytes_nolock', axis=1, inplace=True)
                df = df.append(file_df) 

        # Sort columns of df by alphabetical order:
        df = df.reindex(sorted(df.columns), axis=1)
        features.append(timestamps)
        m2 = {"m2": df}
        features.append(m2)
        return features



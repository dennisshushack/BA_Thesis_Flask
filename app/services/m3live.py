import os 
import sys
import re
import pandas as pd
import numpy as np
import pickle
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")


##############################################################################################################################
#                                                   Data Cleaning
##############################################################################################################################
class m3live:
    @staticmethod
    def extract_line(line: str, real_timestamp) -> list:
        """
        This function cleans the data and gets rid of the summary in the end of a log file.
        """
        try:
            if "Summary of events:" in line:
                return "summary"
            l = re.split(r' |\( |\)', line)
            l = list(filter(lambda a: a != '', l))
            if len(l) < 6:
                return None
            timestamp = l[0]
            time_cost = l[1]
            pid = l[4]
            if l[5] == '...':
                if timestamp == '0.000':
                    return None
                else:
                    syscall = l[7].split('(')[0]
            else:
                syscall = l[5].split('(')[0]
            return [real_timestamp, pid, syscall, time_cost]
        except:
            return None

    @staticmethod
    def clean_data(input_dirs: dict, number: int) -> pd.DataFrame:
        """
        This function cleans the data for monitor 3 for one file:
        """
        # Iterate over the dictionary input_dirs
        skipped = 0
        output_name = ""
        input_dir = input_dirs + "/m3"
        inputfiles = os.listdir(input_dir)
        # Sort the files by timestamp
        inputfiles.sort(key=lambda x: float(x.split('.')[0]))
        # Get the file at the the number-th position
        inputfile = inputfiles[number]
        if inputfile.endswith('.log'):
            outputfile = inputfile.replace('.log', '.csv')
            output_path = input_dir + "/" + outputfile
            outputfile_name = inputfile.split('.')[0]
            # Read the file & create an output file:
            with open(input_dir + "/" + inputfile, 'r') as inp, open(input_dir + "/" + outputfile, 'w') as outp:
                real_timestamp = inputfile.split('.')[0]
                columnNames = ['timestamp', 'pid', 'syscall', 'time_cost']
                outp.write(','.join(columnNames) + '\n')
                for line in inp:
                    try:
                        res = m3live.extract_line(line,real_timestamp)
                    except:
                        res = None
                    if res is not None and res != 'summary':
                        [timestamp, pid, syscall, time_cost] = res
                        outp.write(timestamp + ',' + pid + ',' + syscall + ',' + time_cost + '\n')
                        outp.write('{},{},{},{}\n'.format(timestamp, pid, syscall, time_cost))
                    elif res == 'summary':
                        break
                inp.close()
                outp.close()
                # Remove the original .log file:
                os.remove(input_dir + "/" + inputfile)
        arry = [outputfile_name, output_path]
        return arry


##############################################################################################################################
#                                                  Data Standardization
##############################################################################################################################   
    @staticmethod
    def load_scaler(training_path: str, feature:str):
        """
        This function loads the scaler from the pickle file.
        """
        scaler_path = training_path + "/scalers/" + feature + "_scaler.pickle"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    @staticmethod
    def standardize(df: pd.DataFrame, training_path: str):
        """
        This function standardises the extracted features.
        input: dataframe: pd.DataFrame, category: str, input_path: str, training_path: str
        output: dataframe: pd.DataFrame scaled
        """
        features = [ 
        'frequency_1gram', 
        'tfidf_1gram',
        'hashingvectorizer_1gram',
        'sequence_1gram',
        ]

        feature_normalized_df = {}

        for feature in features:
            # Create a new dataframe with ids, behavior and the feature:
            df_feature = pd.DataFrame({'timestamps': df['id'], feature: df[feature]})
            
            if feature == 'sequence_1gram' or feature == 'hashingvectorizer_1gram':
                feature_normalized_df[feature] = df_feature
                continue
 
            # Load the scaler:
            scaler = m3live.load_scaler(training_path, feature)
            # Standardise the feature:
            data = scaler.transform(df_feature[feature].tolist())
            normaled_df = pd.DataFrame([df_feature['timestamps'].tolist(), data.tolist()]).transpose()
            normaled_df.columns = ['timestamps', feature]
            feature_normalized_df[feature] = normaled_df
            
        return feature_normalized_df

##############################################################################################################################
#                                                  Creating and Loading Vectorizers / Dictionarys
##############################################################################################################################
    @staticmethod
    def load_vectorizers(path: str):
        """
        This function loads the vectorizers from the pickle file.
        output: vectorizers: dict
        """
        vec_path = path + '/vectorizers/'
        vectorizers = {}
        # Can be adjusted:
        for n in range(1, 2):
            cvn = f'countvectorizer_ngram_1-{n}'
            tfn = f'tfidfvectorizer_ngram_1-{n}'
            hvn = f'hashingvectorizer_ngram_1-{n}'
            ngd = f'sequence_1gram'
        
            location = open(f'{vec_path}{cvn}.pickle', 'rb')
            cv = pickle.load(location)
            vectorizers[cvn] = cv

            location = open(f'{vec_path}{tfn}.pickle', 'rb')
            tf = pickle.load(location)
            vectorizers[tfn] = tf

            location = open(f'{vec_path}{ngd}.pickle', 'rb')
            hv = pickle.load(location)
            vectorizers[ngd] = hv

            location = open(f'{vec_path}{hvn}.pickle', 'rb')
            hv = pickle.load(location)
            vectorizers[hvn] = hv
        
        return vectorizers

    @staticmethod
    def apply_vectorizer(features: list, corpus_dataframe:list, corpus: list, training_path: str):
        """
        This applies the vectorizers and dictionnaries to the corpus.
        :input: features: list, corpus: list, training_path: str
        :output: dataframe: pd.DataFrame
        """
        vectorizers = m3live.load_vectorizers(training_path)
        # Can be adjusted
        for n in range(1, 2):
            cvn = f'countvectorizer_ngram_1-{n}'
            tfn = f'tfidfvectorizer_ngram_1-{n}'
            hvn = f'hashingvectorizer_ngram_1-{n}'
            ngd = f'sequence_1gram'

            # CountVectorizer:
            cv = vectorizers[cvn]
            # TfidfVectorizer:
            tf = vectorizers[tfn]
            # HashingVectorizer:
            hv = vectorizers[hvn]
            # Sequence
            ngd = vectorizers[ngd]

            # CountVectorizer:
            cv_features = cv.transform(corpus)
            cv_features = cv_features.toarray()

            # TfidfVectorizer:
            tf_features = tf.transform(corpus)
            tf_features = tf_features.toarray()

            # HashingVectorizer:
            hv_features = hv.transform(corpus)
            hv_features = hv_features.toarray()

            # Dict Sequence:
            dict_sequence_features = []
            for trace in corpus_dataframe:
                syscall_trace = m3live.replace_with_unk(trace['syscall'].to_list(), ngd)
                dict_sequence = m3live.get_dict_sequence(syscall_trace, ngd)
                # Only take the first 100 elements of the dict_sequence:
                dict_sequence = dict_sequence[:1000]
                dict_sequence_features.append(dict_sequence)
            dict_sequence_features = np.array(dict_sequence_features)

            # Append frequency features:
            features.append(cv_features)
            # Append tfidf features:
            features.append(tf_features)
            # Append hashing features:
            features.append(hv_features)
             # Append hashing features:
            features.append(dict_sequence_features)

        encoded_trace_df = pd.DataFrame(features).transpose()
        encoded_trace_df.columns = ['id', 'frequency_1gram', 'tfidf_1gram', 'hashingvectorizer_1gram', 'sequence_1gram']
        return encoded_trace_df



##############################################################################################################################
#                                                  Syscall sequence functions
##############################################################################################################################
    @staticmethod
    def replace_with_unk(syscall_trace, syscall_dict):
        """
        Replaces all values in the systemcall trace with 
        unique, if they do not appear in the systemcall dict.
        """
        for i, sc in enumerate(syscall_trace):
            if sc.lower() not in syscall_dict:
                syscall_trace[i] = 'unk'
        return syscall_trace
    
    @staticmethod
    def get_dict_sequence(trace,term_dict):
        """
        Gets the actual sequence of systemcalls
        """
        dict_sequence = []
        for syscall in trace:
            if syscall in term_dict:
                dict_sequence.append(term_dict[syscall])
            else:
                dict_sequence.append(term_dict['unk'])
        return dict_sequence


##############################################################################################################################
#                                                  Data Processing 
##############################################################################################################################
    @staticmethod
    def from_list_to_str(trace: list):
        """
        This function creates a string from a list containing the system call traces.
        :input: trace: list
        :output: string: str
        """
        tracestr = ''
        for syscall in trace:
            tracestr += syscall + ' '
        return tracestr
    
    @staticmethod
    def get_corpus(file_path: str):
        """
        This function creates a corpus (list) of all the systemcalls in the files.
        :input: path: str, files: list :output: corpus: list i.e [syscall1 sycall2 syscall3, syscall4 ...]
        """
        corpus = []
        corpus_dataframe = []
        trace = pd.read_csv(file_path)
        corpus_dataframe.append(trace)
        tr = trace['syscall'].tolist()
        longstr = m3live.from_list_to_str(tr)
        corpus.append(longstr)
        return corpus_dataframe, corpus

###############################################################################################################################
#                                                  Main function
###############################################################################################################################
    @staticmethod
    def preprocess_data(input_dirs: dict, training_dirs: list, number:int):
        # Get the value of the first element of the dictionary:
        try:
            start_time_monitor3 = time.time()
            print(f"Preprocessing data for monitor 3...{start_time_monitor3}")
            input_dirs = list(input_dirs.values())[0]
            list_of_return_dict = []
            arry= m3live.clean_data(input_dirs,number)
            file_name = arry[0]
            file_path = arry[1]
            features_anomaly = []
            features_classification = []
            file_ids = [file_name]
            features_anomaly.append(file_ids)
            features_classification.append(file_ids)
            corpus_dataframe ,corpus = m3live.get_corpus(file_path)
            # For anomaly detection & classification:
            print("Applying vectorizer to the anomaly detection & classification features...")
            df_anomaly = m3live.apply_vectorizer(features_anomaly, corpus_dataframe, corpus, training_dirs[0])
            df_classification = m3live.apply_vectorizer(features_classification, corpus_dataframe, corpus, training_dirs[1])
            print("Standardizing the features...")
            dict_df_anomaly = m3live.standardize(df_anomaly, training_dirs[0])
            dict_df_classification = m3live.standardize(df_classification, training_dirs[1])
            list_of_return_dict.append(dict_df_anomaly)
            list_of_return_dict.append(dict_df_classification)
            end_time_monitor3 = time.time()
            print(f"Preprocessing data for monitor 3...Done! {end_time_monitor3}")
            print(f"Time taken to preprocess the data for monitor 3: {end_time_monitor3 - start_time_monitor3}")
            return list_of_return_dict
        except:
            return None
        

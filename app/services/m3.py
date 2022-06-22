import os 
import sys
import re
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

warnings.simplefilter(action='ignore', category=FutureWarning)


if not sys.warnoptions:
    warnings.simplefilter("ignore")

# M3 (Anomaly & Classification)
class m3:
    """
    Class for data concerning monitor 3 (systemcalls)
    Creates a cleaned & standardized dataframe ready for training/evaluation
    for classificaiton or anomaly detection
    """
    @staticmethod
    def extract_line(line: str, real_timestamp) -> list:
        """
        This function cleans the data and gets rid of the summary in the end of a log file.
        :input: line: a line from the log file
        :output: a list with the necessary information
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
    def clean_data(input_dirs: dict, begin:int = None, end:int = None):
        """
        This function preprocesses the data.
        Gets rid of unnecessary parts of the log files 
        and returns .csv files with the necessary information.
        :input: input_dirs: dictionary with the path & behavior to the input directory where raw data (.log) files are stored,
        :output: .csv files with the cleaned data
        """
        # Iterate over the dictionary input_dirs
        skipped = 0
        for key, value in input_dirs.items():
            input_dir = value + "/m3"
            # Iterate over the files in the directory (path):
            for inputfile in os.listdir(input_dir):
                if inputfile.endswith('.log'):
                    outputfile = inputfile.replace('.log', '.csv')
                    # Read the file & create an output file:
                    try:
                        with open(input_dir + "/" + inputfile, 'r') as inp, open(input_dir + "/" + outputfile, 'w') as outp:
                            real_timestamp = inputfile.split('.')[0]
                            columnNames = ['timestamp', 'pid', 'syscall', 'time_cost']
                            outp.write(','.join(columnNames) + '\n')
                            for line in inp:
                                try:
                                    res = m3.extract_line(line,real_timestamp)
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
                    except:
                        # Remove the original .log file:
                        os.remove(input_dir + "/" + inputfile)
                        skipped += 1
                        continue
            if begin is not None and end is not None:            
                for outputfile in os.listdir(input_dir):
                    output_filename = outputfile.split('.')[0]
                    if int(output_filename) < begin or int(output_filename) > end:
                        os.remove(input_dir + "/" + outputfile)
                    

    @staticmethod
    def load_scaler(training_path: str, feature:str):
        """
        This function loads the scaler from the pickle file.
        :input: training_path: path to the directory where the scaler is stored,
        feature: the feature that is used for the scaler
        :output: the scaler
        """
        scaler_path = training_path + "/scalers/" + feature + "_scaler.pickle"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded from: " + scaler_path)
        return scaler


    @staticmethod
    def create_scaler(df: pd.DataFrame, training_path: str, feature: str):
        """
        This function creates a Standard Scaler object and saves it in a pickle file.
        :input: df: a dataframe, training_path: path to the directory where the scaler will be saved,
        :output: a pickle file with the scaler
        """

        scaler_path = training_path + '/scalers/'
        if not os.path.exists(scaler_path):
            os.makedirs(scaler_path)
        
        features_to_scale = df[feature].tolist()
        X_train, X_val = train_test_split(features_to_scale, test_size=.3, shuffle=True, random_state=42)

        # Only fit the scaler on the training data:
        scaler = StandardScaler().fit(X_train)
        # Save the scaler:
        with open(f'{scaler_path}{feature}_scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)
        
    
    @staticmethod
    def standardize(df: pd.DataFrame, category: str, training_path: str):
        """
        This function standardises the extracted features.
        input: dataframe: pd.DataFrame, category: str, input_path: str, training_path: str
        output: dataframe: pd.DataFrame scaled
        """
        features = [ 
        'frequency_1gram', 
        'tfidf_1gram',
        'hashing_1gram'
        ]

        feature_normalized_df = {}

        for feature in features:
            # Create a new dataframe with ids, behavior and the feature:
            df_feature = pd.DataFrame({'id': df['id'], 'behavior': df['behavior'], feature: df[feature]})
            if feature == 'hashing_1gram':
                feature_normalized_df[feature] = df_feature
                continue

            if category == 'training':
                m3.create_scaler(df_feature, training_path, feature)
                # Load the scaler:
                scaler = m3.load_scaler(training_path, feature)
                # Standardise the feature:
                data = scaler.transform(df_feature[feature].tolist())
                normaled_df = pd.DataFrame([df_feature['id'].tolist(), df_feature['behavior'].tolist(), data.tolist()]).transpose()
                normaled_df.columns = ['id', 'behavior', feature]
                feature_normalized_df[feature] = normaled_df
            else:
                # Load the scaler:
                scaler = m3.load_scaler(training_path, feature)
                # Standardise the feature:
                data = scaler.transform(df_feature[feature].tolist())
                normaled_df = pd.DataFrame([df_feature['id'].tolist(), df_feature['behavior'].tolist(), data.tolist()]).transpose()
                normaled_df.columns = ['id', 'behavior', feature]
                feature_normalized_df[feature] = normaled_df
            
        return feature_normalized_df

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
        
            location = open(f'{vec_path}{cvn}.pickle', 'rb')
            cv = pickle.load(location)
            vectorizers[cvn] = cv

            location = open(f'{vec_path}{tfn}.pickle', 'rb')
            tf = pickle.load(location)
            vectorizers[tfn] = tf

            location = open(f'{vec_path}{hvn}.pickle', 'rb')
            hv = pickle.load(location)
            vectorizers[hvn] = hv

        return vectorizers


    @staticmethod
    def apply_vectorizer(features: list, corpus: list, training_path: str):
        """
        This applies the vectorizers to the corpus.
        :input: features: list, corpus: list, training_path: str
        :output: dataframe: pd.DataFrame
        """
        vectorizers = m3.load_vectorizers(training_path)
        # Can be adjusted
        for n in range(1, 2):
            cvn = f'countvectorizer_ngram_1-{n}'
            tfn = f'tfidfvectorizer_ngram_1-{n}'
            hvn = f'hashingvectorizer_ngram_1-{n}'

            # CountVectorizer:
            cv = vectorizers[cvn]
            # TfidfVectorizer:
            tf = vectorizers[tfn]
            # HashVectorizer:
            hv = vectorizers[hvn]

            # CountVectorizer:
            cv_features = cv.transform(corpus)
            cv_features = cv_features.toarray()

            # TfidfVectorizer:
            tf_features = tf.transform(corpus)
            tf_features = tf_features.toarray()

            # HashVectorizer:
            hv_features = hv.transform(corpus)
            hv_features = hv_features.toarray()

            # Append frequency features:
            features.append(cv_features)

            # Append tfidf features:
            features.append(tf_features)

            # Append hashing features:
            features.append(hv_features)


        encoded_trace_df = pd.DataFrame(features).transpose()
        # Please adjust accordingly if higher ngrams are used:
        encoded_trace_df.columns = ['id', 'behavior', 'frequency_1gram', 'tfidf_1gram', 
        'hashing_1gram']

        return encoded_trace_df

    
    @staticmethod
    def create_vectorizer(corpus: list, path: str):
        """
        This function creates the vectorizers with ngram of size 1
        :input: corpus: list, path: str train_path
        :output: vectorizers: saved in a pickle file + dictionary
        """
        # Checks if directory vectorizers exists, if not, creates it:
        vec_path = path + '/vectorizers/'
        if not os.path.exists(vec_path):
            os.makedirs(vec_path)
        vectorizers = {}
        
        # Only creates 1-gram can be adjusted:
        for n in range(1, 2):
            vectorizers[f'countvectorizer_ngram_1-{n}'] = CountVectorizer(ngram_range=(1, n)).fit(corpus)
            n_gram_dict = vectorizers[f'countvectorizer_ngram_1-{n}'].vocabulary_
            vectorizers[f'tfidfvectorizer_ngram_1-{n}'] = TfidfVectorizer(ngram_range=(1, n)).fit(corpus)
            vectorizers[f'hashingvectorizer_ngram_1-{n}'] = HashingVectorizer(ngram_range=(1, n), n_features=2**10).fit(corpus)
        
        # Saves every vectorizer in a pickle file:
        for name in vectorizers:
            with open(f'{vec_path}{name}.pickle', 'wb') as f:
                pickle.dump(vectorizers[name], f)

        # Saves n_gram_dict in a pickle file:
        with open(f'{vec_path}n_gram_dict.pickle', 'wb') as f:  
            pickle.dump(n_gram_dict, f)

    
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
    def get_corpus(path: str, files: list):
        """
        This function creates a corpus (list) of all the systemcalls in the files.
        :input: path: str, files: list
        :output: corpus: list i.e [syscall1 sycall2 syscall3, syscall4 ...]
        """
        corpus = []
        for file in files:
            if '.csv' in file:
                file_path = path + "/" + file
                try:
                    trace = pd.read_csv(file_path)
                except:
                    continue
                tr = trace['syscall'].tolist()
                longstr = m3.from_list_to_str(tr)
                corpus.append(longstr)
        return corpus
    
    @staticmethod
    def get_features(input_dirs: dict, category: str, train_path: str, test_path: str = None):
        """
        This function extracts the features from the raw data using 
        1) Bag-of-words 2) Tfidf and 3) Hash vectorization 
        -----------------------------------------------------------
        :input: input_dirs dictonary with path to .csv files, with the bahaviors
                category: the category of the data (train vs test)
        :output: Creates usable features for training 
        """
       
        features = []
        file_ids, behaviors = [], []
        corpus = []
        print("Extracting features...")
        # Get's all files:
        for key, value in input_dirs.items():
            input_dir = value + "/m3"
            behavior = key
            files = os.listdir(input_dir)
            # Sort files by name:
            files.sort(key=lambda x: x.split('.')[0])
            file_ids_sub, behaviors_sub = [], []

            # Iterate over the files:
            for file in files:
                if '.csv' in file:
                    file_ids_sub.append(int(file.replace('.csv', '')))
                    behaviors_sub.append(behavior)
            
            # Get the corpus from the files in the input_dir:
            print("Creating corpus...")
            corpus_subdirectory = m3.get_corpus(input_dir, files)
            
            # Extend the corpus:
            corpus.extend(corpus_subdirectory)

            # Extend the file_ids:
            file_ids.extend(file_ids_sub)

            # Extend the behaviors:
            behaviors.extend(behaviors_sub)
        
        features.append(file_ids)
        features.append(behaviors)

        if category == 'training':
            # Create the different vectorizers:
            print("Creating vectorizers...")
            m3.create_vectorizer(corpus, train_path)
            # Apply the vectorizers to the corpus:
            df = m3.apply_vectorizer(features, corpus, train_path)
        else:
            df = m3.apply_vectorizer(features, corpus, train_path)
        
        # Drop any NaN values in the dataframe:
        df.dropna(inplace=True)

        # Apply StandardScaler:
        dict_df = m3.standardize(df, category,train_path)

        # Create directory for the features:
        if category == "training":
            feat_path = train_path + '/preprocessed/'
            if not os.path.exists(feat_path):
                os.makedirs(feat_path)
            
            # Save the dataframes in the dictionary as .csv files:
            for key, value in dict_df.items():
                value.to_csv(f'{feat_path}{key}.csv', index=False)
        else:
            feat_path = test_path + '/preprocessed/'
            if not os.path.exists(feat_path):
                os.makedirs(feat_path)

            for key, value in dict_df.items():
                value.to_csv(f'{feat_path}{key}.csv', index=False)


        return dict_df

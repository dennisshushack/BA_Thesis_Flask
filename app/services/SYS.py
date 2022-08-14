import os 
import sys
import re
import pandas as pd
import numpy as np
import pickle
import time
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

##############################################################################################################################
#                                                   Data Cleaning                                                         # 
##############################################################################################################################
class SYS:
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
    def clean_data(input_dirs: dict) -> pd.DataFrame:
        """
        This function cleans the data for monitor 3:
        """
        # Iterate over the dictionary input_dirs
        skipped = 0
        for key, value in input_dirs.items():
            input_dir = value + "/SYS"
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
                                    res = SYS.extract_line(line,real_timestamp)
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


##############################################################################################################################
#                                                  Making the vectorizer / Sequence    (Training)                            #
##############################################################################################################################
    @staticmethod
    def make_vectorizers(corpus, train, features):
        corpuses = {}

        # Creating n-gram 1 and 2:    
        for n in range(1, 2):
            print("Creating n-grams of length {}".format(n))
            start_create_cv = time.time()
            print("Create CountVectorizer")
            cv = CountVectorizer(ngram_range=(1,n)).fit(corpus)
            end_create_cv = time.time()
            vocabulary = cv.vocabulary_
            print("Create TfidfVectorizer")
            start_create_tfidf = time.time()
            tfidf = TfidfVectorizer(ngram_range=(1, n), vocabulary=vocabulary).fit(corpus)
            end_create_tfidf = time.time()
            start_create_hv = time.time()
            print("Create HashingVectorizer")
            hash = HashingVectorizer(n_features=2**10, norm='l2', ngram_range=(1, n)).fit(corpus)
            end_create_hv = time.time()

            # Saving the vectorizers:
            vec_path = train + '/vectorizers/'
            if not os.path.exists(vec_path):
                os.makedirs(vec_path)

            # Pickle file for transforming the corpus into a vector:
            pickle.dump(cv, open(vec_path + f'CountVectorizer_{n}.pkl', 'wb'))
            pickle.dump(tfidf, open(vec_path + f'TfidfVectorizer_{n}.pkl', 'wb'))
            pickle.dump(hash, open(vec_path + f'HashingVectorizer_{n}.pkl', 'wb'))


            start_apply_hv = time.time()
            print("Applying HashingVectorizer...")
            HashingVectorizer_corpus = hash.transform(corpus).toarray()
            end_apply_hv = time.time()
            corpuses[f'HashingVectorizer_{n}'] = HashingVectorizer_corpus
            del HashingVectorizer_corpus


            # Apply the vectorizers:
            start_apply_cv = time.time()
            print("Applying CountVectorizer...")
            CountVectorizer_corpus = cv.transform(corpus).toarray()
            end_apply_cv = time.time()
            corpuses[f'CountVectorizer_{n}'] = CountVectorizer_corpus
            del CountVectorizer_corpus

            start_apply_tfidf = time.time()
            print("Applying TfidfVectorizer...")
            TfidfVectorizer_corpus = tfidf.transform(corpus).toarray()
            end_apply_tfidf = time.time()
            corpuses[f'TfidfVectorizer_{n}'] = TfidfVectorizer_corpus
            del TfidfVectorizer_corpus

           

            # Delete the corpus:
            del corpus
        

            # Save metrics in .csv:
            with open('output.csv', 'a', newline='') as csvfile:
                fieldnames = ['Name', 'Start', 'End', 'Duration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Name': f'Creating CountVectorizer_{n}', 'Start': start_create_cv, 'End': end_create_cv, 'Duration': end_create_cv - start_create_cv})
                writer.writerow({'Name': f'Applying CountVectorizer_{n}', 'Start': start_apply_cv, 'End': end_apply_cv, 'Duration': end_apply_cv - start_apply_cv})
                writer.writerow({'Name': f'Creating TfidfVectorizer_{n}', 'Start': start_create_tfidf, 'End': end_create_tfidf, 'Duration': end_create_tfidf - start_create_tfidf})
                writer.writerow({'Name': f'Applying TfidfVectorizer_{n}', 'Start': start_apply_tfidf, 'End': end_apply_tfidf, 'Duration': end_apply_tfidf - start_apply_tfidf})
                writer.writerow({'Name': f'Creating HashingVectorizer_{n}', 'Start': start_create_hv, 'End': end_create_hv, 'Duration': end_create_hv - start_create_hv})
                writer.writerow({'Name': f'Applying HashingVectorizer_{n}', 'Start': start_apply_hv, 'End': end_apply_hv, 'Duration': end_apply_hv - start_apply_hv})
                csvfile.close()    

        # Return the features:
        features.append(corpuses)
        return features

##############################################################################################################################
#                                                  Apply Vectorizer (Testing)                                                #
##############################################################################################################################
    @staticmethod
    def apply_vectorizers(corpus, train, features):
        corpuses = {}
        print("Applying vectorizers...")
        vec_path = train + '/vectorizers/'
        for n in range(1, 2):
            cv = pickle.load(open(vec_path + f'CountVectorizer_{n}.pkl', 'rb'))
            tfidf = pickle.load(open(vec_path + f'TfidfVectorizer_{n}.pkl', 'rb'))
            hash = pickle.load(open(vec_path + f'HashingVectorizer_{n}.pkl', 'rb'))
            
            # Apply the vectorizers:
            CountVectorizer_corpus = cv.transform(corpus)
            CountVectorizer_corpus = CountVectorizer_corpus.toarray()
            corpuses[f'CountVectorizer_{n}'] = CountVectorizer_corpus

            TfidfVectorizer_corpus = tfidf.transform(corpus)
            TfidfVectorizer_corpus = TfidfVectorizer_corpus.toarray()
            corpuses[f'TfidfVectorizer_{n}'] = TfidfVectorizer_corpus

            HashingVectorizer_corpus = hash.transform(corpus)
            HashingVectorizer_corpus = HashingVectorizer_corpus.toarray()
            corpuses[f'HashingVectorizer_{n}'] = HashingVectorizer_corpus

        # Return the features:
        features.append(corpuses)
        return features


##############################################################################################################################
#                                                  Data Processing                                                          #
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
    def get_corpus(path: str, files: list):
        """
        This function creates a corpus (list) of all the systemcalls in the files.
        :input: path: str, files: list :output: corpus: list i.e [syscall1 sycall2 syscall3, syscall4 ...]
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
                longstr = SYS.from_list_to_str(tr)
                corpus.append(longstr)
        return corpus
    
    ###############################################################################################################################
    #                                                  Main function
    ###############################################################################################################################


    @staticmethod
    def preprocess_data(input_dirs: dict, category: str, training_dirs: list):
        """
        Preprocesses SYS: return => [timestamps, behaviors, dict_of_features]
        """

        print("Cleaning SYS's data...")
        time_start = time.time()
        SYS.clean_data(input_dirs)
        time_end = time.time()

        # Save in the output.csv:
        with open('output.csv', 'a', newline='') as csvfile:
            fieldnames = ['Name', 'Start', 'End', 'Duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Name': 'Cleaning SYS\'s data', 'Start': time_start, 'End': time_end, 'Duration': time_end - time_start})
            csvfile.close()

        print("Creating the corpus...")
        time_start = time.time()
        features = []
        file_ids, behaviors = [], []
        corpus = []

        for key, value in input_dirs.items():
            input_dir = value + "/SYS"
            behavior = key
            files = os.listdir(input_dir)
            # This is for every subirecory 
            file_ids_sub, behaviors_sub = [], []
            # Iterate over the files:
            for file in files:
                if '.csv' in file:
                    file_ids_sub.append(int(file.replace('.csv', '')))
                    behaviors_sub.append(behavior)
            corpus_subdirectory = SYS.get_corpus(input_dir, files)
            corpus.extend(corpus_subdirectory)
            file_ids.extend(file_ids_sub)
            behaviors.extend(behaviors_sub)
            del file_ids_sub, behaviors_sub, corpus_subdirectory
        time_end = time.time()
        # Save in the output.csv:
        with open('output.csv', 'a', newline='') as csvfile:
            fieldnames = ['Name', 'Start', 'End', 'Duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Name': 'Creating the corpus', 'Start': time_start, 'End': time_end, 'Duration': time_end - time_start})
            csvfile.close()
        
        # Here we have the timestamps and behaviors:
        features.append(file_ids)
        features.append(behaviors)

        if category == "training":
            return SYS.make_vectorizers(corpus, training_dirs[0], features)
        elif category == "testing":
            return SYS.apply_vectorizers(corpus, training_dirs[0], features)




      





            
    

            



        
                








      
            
     

        


    

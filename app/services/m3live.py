import os 
import sys
import re
import pandas as pd
import numpy as np
import pickle
import warnings
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
#                                                   Load Vectorizer
##############################################################################################################################
    @staticmethod
    def apply_vectorizers(corpus, train, features):
        
        corpuses = {}
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
    def preprocess_data(input_dirs: dict,  number:int, training_dir: str):
        features = []
        arry= m3live.clean_data(input_dirs,number)
        file_name = arry[0]
        file_path = arry[1]
        features.append([file_name])
        corpus_dataframe ,corpus = m3live.get_corpus(file_path)
        return m3live.apply_vectorizers(corpus, training_dir, features)



            
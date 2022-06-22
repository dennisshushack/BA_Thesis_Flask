import json
import os 
from flask import Blueprint
import threading
import sqlite3
from flask import request, jsonify
from app import create_app
from app.db import get_db, close_db, init_app
from app.services.AuthenticationService import login_required
from app.services.m3 import m3
from app.services.m2 import m2
from app.services.m1 import m1
from app.services.anomalyml import anomalyml
from app.services.anomalydl import anomalydl
from app.services.classificationml import classificationml

from app.database.dbqueries import dbqueries

bp = Blueprint('rest', __name__, url_prefix='/rest')


def identify_subdirectories(path: str, behavior: str = None) -> dict:
    """
    This function identifies the subdirectories of the given path.
    :input: path: path of the directory
    :return: dict of subdirectories
    """
    # For testing:
    if behavior != None:
        subdirectories = {behavior: path + '/' + behavior}
        return subdirectories

    # For training (classification or anomaly detection):
    else:
        subdirectories = {}
        subdirectory_names = ['normal', 'poc', 'dark','raas']
        for dirs in os.walk(path):
            for subdirectory in subdirectory_names:
                if subdirectory in dirs[1]:
                    subdirectories[subdirectory] = path + '/' + subdirectory
        return subdirectories

def get_paths(db: sqlite3.Connection, category: str, device: str, ml_type: str, path: str) -> dict:
    """
    This function returns the paths to the training and testing data (wherere the data to preprocess is stored).
    :input: db: database connection, category: str, device: str,  ml_type: str, path: str
    :return: paths: dict with the paths to the training and testing data
    """
    paths = {}
    if category == 'testing':
        # Location must be stored if the category is testing:
        training_path = dbqueries.get_training_data_location(db, device, ml_type)
        testing_path = path
    else:
        # Otherwise it will be newly created or is already stored:
        if dbqueries.get_training_data_location(db, device, ml_type) != None:
            training_path = path
            testing_path = None
        else:
            dbqueries.insert_training_data_location(db, device, ml_type, path)
            training_path = path
            testing_path = None
    
    paths['training'] = training_path
    paths['testing'] = testing_path
    return paths

def preprocess_data(monitors: list, subdirectories: dict, begin: int, end: int, category: str, training_path: str, testing_path: str) -> dict:
    """
    This function preprocesses the data for the given monitors (m1, m2, m3).
    :input: monitors: list of monitors, subdirectories: dict of subdirectories, begin: int, end: int, category: str, training_path: str, testing_path: str
    :return: data: dict of dataframes
    """
    preprocessed_data = {}
    for monitor in monitors:
        # Preprocessing for monitor m1:
        if monitor == 'm1':
            df_m1, vector_behavior_m1, ids_m1, m1_timestamps = m1.clean_data(subdirectories, begin, end)
            m1_preprocessed = m1.preprocess_data(df_m1, vector_behavior_m1, ids_m1, category, training_path, m1_timestamps, testing_path)
            preprocessed_data['m1'] = m1_preprocessed

        # Preprocessing for monitor m2:
        elif monitor == 'm2':
            df_m2, vector_behavior_m2, ids_m2, m2_timestamps = m2.clean_data(subdirectories, begin, end)
            m2_preprocessed = m2.preprocess_data(df_m2, vector_behavior_m2, ids_m2, category, training_path, m2_timestamps, testing_path)
            preprocessed_data['m2'] = m2_preprocessed

        # Preprocessing for monitor m3:
        elif monitor == "m3":
            m3.clean_data(subdirectories,begin,end)
            dict_df = m3.get_features(subdirectories, category, training_path, testing_path)
            for key, value in dict_df.items():
                preprocessed_data[key] = value

    return preprocessed_data

def train_anomaly_detection(db: sqlite3.Connection, device: str, preprocessed_data: dict, training_path: str):
    """
    This function trains the anomaly detection model (with machine and deep learning).
    :input: db: database connection, device: str, preprocessed_data: dict, training_path: str
    :return: None
    """
    # Training ML Algorithms:
    for feature, dataframe in preprocessed_data.items():
        list_anomaly_training = anomalyml.train(dataframe, training_path, feature)
        for item in list_anomaly_training:
            for key, value in item.items():
                model = key
                TNR = value[0]
                train_time = value[1]
                dbqueries.create_ml_anomaly(db, device, feature, model, TNR, train_time)

    # Training DL Algorithms:
    for feature, dataframe in preprocessed_data.items():
        dict_dl_anomaly = anomalydl.train(dataframe, training_path, feature)
        TNR = dict_dl_anomaly['TNR']
        train_time = dict_dl_anomaly['training_time']
        threshhold = dict_dl_anomaly['threshhold']
        neurons = str(dict_dl_anomaly['hidden_layers'])
        dbqueries.create_dl_anomaly(db, device, feature, "autoencoder", TNR, train_time, threshhold, neurons)
    
    return

def test_anomaly_detection(db: sqlite3.Connection, device: str, preprocessed_data: dict, training_path: str, experiment: str, behavior: str):
    """
    This function evaluates the data (testing new unseen data) for anomaly detection.
    :input: db: database connection, device: str, preprocessed_data: dict, training_path: str, experiment: str, behavior: str
    :return: None
    """
    # Testing ML Algorithms:
    for feature, dataframe in preprocessed_data.items():
        list_anomaly_testing = anomalyml.validate(dataframe, training_path, feature)
        for item in list_anomaly_testing:
            for key, value in item.items():
                model = key
                TPR = value[0]
                test_time = value[1]
                # Get the correct primary key to use, as a foreign key:
                primary_key = dbqueries.get_foreign_key_ml(db, device, feature, model)
                dbqueries.create_ml_anomaly_testing(db, device, experiment, behavior,feature, model, TPR, test_time, primary_key)
            
    # Part Anomaly Detection DL:
    for feature, dataframe in preprocessed_data.items():
        print("Testing DL for feature: " + feature)
        threshhold = dbqueries.get_threshold(db, device, feature)
        threshhold = float(threshhold)
        print(threshhold)
        model_list = anomalydl.validate(dataframe, training_path, feature, threshhold)
        print("hello")
        TPR = model_list[0]
        test_time = model_list[1]
        # Get the correct primary key to use, as a foreign key:
        primary_key = dbqueries.get_foreign_key_dl(db, device, feature, "autoencoder")
        dbqueries.create_dl_anomaly_testing(db, device, experiment, behavior, feature, "autoencoder",TPR, test_time, primary_key)
    
    return



def background(data: json):
    """
    Is executed as a seperate thread in the background.
    As it is very process heavy the user will get informed before the job starts.
    Takes as an input the data that is sent to the endpoint (see main endpoint below)
    """
    app = create_app()
    with app.app_context():
        
        # Get the data from the database:
        try:
            db = get_db()
        except:
            app.logger.debug('Database not found')
            return
        
        # Get the data from the request
        device = data['device']
        category = data['category']
        ml_type = data['ml_type']
        monitors = data['monitors']
        behavior = data['behavior']
        path = data['path']
        begin = data['begin']
        end = data['end']
        experiment = data['experiment']
        monitors = monitors.split(',')
        if begin == 'None':
            begin = None
        if end == 'None':
            end = None
        
        # The testing folder contains experiments:
        if category == "testing":
            path = path + '/' + experiment
        
         # Insert the data into the database table post_requests:
        dbService = dbqueries()
        dbService.insert_into_post_requests(db, device, category, ml_type, str(monitors), behavior, path)

        # Get the paths to the training or testing data:
        paths = get_paths(db, category, device, ml_type, path)
        training_path = paths['training']
        testing_path = paths['testing']

        # Find the subdirectories (i.e testing may have multiple experiments or classification may have multiple ransomwares):
        if category == 'testing':
            subdirectories = identify_subdirectories(path, behavior)
        else:
            subdirectories = identify_subdirectories(path)    

        # Preprocess the data:
        preprocessed_data = preprocess_data(monitors, subdirectories, begin, end, category, training_path, testing_path)
      
        # Only for Anomaly detection training:
        if ml_type == 'anomaly' and category == 'training':
            train_anomaly_detection(db, device, preprocessed_data, training_path)
            
        
        # Only for Anomaly detection testing:
        if ml_type == "anomaly" and category == "testing":
            test_anomaly_detection(db, device, preprocessed_data, training_path, experiment, behavior)
            
           
        # Only for Classification training:
        if ml_type == 'classification' and category == 'training':
            for feature, dataframe in preprocessed_data.items():
                print("Starting classification ML training...")
                classificationml.train(dataframe, training_path, feature)                
                print("Classification ML training finished.")
                
        print("Done")
        return
    
        

@bp.route('/test', methods=['GET'])
@login_required
def test():
    """
    This endpoint is used to test, if the server is up
    """
    return jsonify({'message': 'Server is up'})
    
    
@bp.route('/main', methods=['POST'])
@login_required
def main():
    """
    This endpoint is used to preprocess the data for Machine Learning and Deep Larning models and then train
    or evaluate the models.
    :input:
    1. device: The device to be used for the training (CPU id)
    2. category: The category of the data to be used for the training (training or testing)
    3. ml_type: The type of the machine learning model to be used (classification or anomaly)
    4. behavior: The behavior to be used for the training (normal, poc, dark or raas)
    5. path: The path to the data to be used for the training/evaluation
    6. begin: A timestamp to be used for the training/evaluation (preprocssing)
    7. end: A timestamp to be used for the training/evaluation (preprocessing)
    8. experiment: description of the monitoring 
    9. monitors: list of monitors to be used for the training/evaluation
    """
    
    # Checks, if the input body is in a JSON format:
    if not request.is_json:
        return jsonify({"status": "error", "message": "input error not json"})

    # Gets the data from the request:
    data = request.get_json()

    # Start the background job (prevents the server from being blocked by a request):
    threading.Thread(target=background, args=(data,)).start()
    return jsonify({"status": "ok", "message": "started"})







   
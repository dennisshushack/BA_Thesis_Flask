import json
import os
from unicodedata import category 
from flask import Blueprint
import threading
import sqlite3
import time
from flask import request, jsonify
from app import create_app
from app.db import get_db, close_db, init_app
from app.services.AuthenticationService import login_required
from app.services.m3 import m3
from app.services.m2 import m2
from app.services.m1 import m1
from app.services.m1live import m1live
from app.services.m2live import m2live
from app.services.m3live import m3live
from app.services.anomalyml import anomalyml
from app.services.anomalydl import anomalydl
from app.services.classificationml import classification
from app.database.dbqueries import dbqueries

bp = Blueprint('rest', __name__, url_prefix='/rest')

########################################################################################################################
#                                          Helper Functions                                                          
########################################################################################################################

def get_paths(db: sqlite3.Connection, category: str, device: str, ml_type: str, path: str) -> dict:
    """
    Returns the training paths where the models are saved. Returns it as a list:
    """
    if category == 'testing' and ml_type == None:
        # This is the case when live testing is performed:
        training_path_anomaly = dbqueries.get_training_data_location(db, device, "anomaly")
        training_path_classification = dbqueries.get_training_data_location(db, device, "classification")
        training_paths = [training_path_anomaly, training_path_classification]

    elif category == 'testing' and ml_type != None:
        # For testing collected data:
        training_path = dbqueries.get_training_data_location(db, device, ml_type)
        training_paths = [training_path]

    else:
        # Checks, if the training path is allready in the database:
        if dbqueries.get_training_data_location(db, device, ml_type) != None:
            training_path = path
        else:
            dbqueries.insert_training_data_location(db, device, ml_type, path)
            training_path = path
        training_paths = [training_path]
    return training_paths

def identify_subdirectories(path: str)-> dict:
    """
    This function identifies the subdirectories of the given path.
    """
    subdirectories = {}
    directories = os.listdir(path)
    for directory in directories:
        subdirectories[directory] = path + '/' + directory
    return subdirectories

def preprocess_data(monitors:list , subdirectories: dict, training_paths: list, category: str, path: str) -> dict:
    """
    This function preprocesses the data for m1, m2 and m3 
    """
    preprocessed_data = {}
    for monitor in monitors:
        if monitor == "m1":
            df = m1.preprocess_data(subdirectories, category, training_paths, path)
            preprocessed_data['m1'] = df
        elif monitor == "m2":
            df = m2.preprocess_data(subdirectories, category, training_paths, path)
            preprocessed_data['m2'] = df
        else:
            dict_df = m3.preprocess_data(subdirectories, category, training_paths, path)
            for name, df in dict_df.items():
                preprocessed_data[name] = df
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
    
    return "Done"

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
        TPR = model_list[0]
        test_time = model_list[1]
        # Get the correct primary key to use, as a foreign key:
        primary_key = dbqueries.get_foreign_key_dl(db, device, feature, "autoencoder")
        dbqueries.create_dl_anomaly_testing(db, device, experiment, behavior, feature, "autoencoder",TPR, test_time, primary_key)
    
    return

###################### Background Threads ##############################

def background_training(data: json):
    app = create_app()
    with app.app_context():
        
        # Get the data from the database:
        try:
            db = get_db()
        except:
            app.logger.debug('Database not found')
            return
            
        # General data also for testing:
        ml_type = data['ml_type']
        device = data['device']
        monitors = data['monitors'].split(',')
        path = data['path']
        experiment = data['experiment']
        category = data['category']
        behavior = data['behavior']

        # The testing folder contains experiment subdirectries:
        if category == "testing":
            path = path + '/' + experiment

        # Get the paths to the training or testing data (list of paths):
        training_paths = get_paths(db, category, device, ml_type, path)

        # Find the subdirectories of the path:
        subdirectories = identify_subdirectories(path)   

        # Preprocess the data:
        preprocessed_data = preprocess_data(monitors, subdirectories, training_paths, category, path)

        # Only for Anomaly detection training:
        if ml_type == 'anomaly' and category == 'training':
            train_anomaly_detection(db, device, preprocessed_data, training_paths[0])
            
        # Only for Anomaly detection testing:
        if ml_type == "anomaly" and category == "testing":
            test_anomaly_detection(db, device, preprocessed_data, training_paths[0], experiment, behavior)

        # Only for Classification training:
        if ml_type == 'classification' and category == 'training':
            for feature, dataframe in preprocessed_data.items():
                print("Starting classification ML training...")
                classification.train(dataframe, training_paths[0], feature)                
                print("Classification ML training finished.")
                
        print("Done")
        return

def background_live_thread(number):
    while True:
        time.sleep(number)
        print("Background thread running...")
        return

    
################################# Endpoints #####################################

@bp.route('/test', methods=['GET'])
def test():
    """
    This endpoint is used to test, if the server is up!
    """
    return jsonify({'message': 'Server is up'})
    
    
@bp.route('/training', methods=['POST'])
@login_required
def training():
    """
    This endpoint is used to preprocess the data for Machine Learning and Deep Larning models and then train
    or evaluate the models with collected data and not live data
    :input:
    1. device: The device to be used for the training (CPU id)
    2. ml_type: The type of the machine learning model to be used (classification or anomaly)
    3. behavior: The behavior to be used for the training (normal, poc, dark or raas)
    4. path: The path to the data to be used for the training/evaluation
    5. begin: A timestamp to be used for the training/evaluation (preprocssing)
    6. end: A timestamp to be used for the training/evaluation (preprocessing)
    7. experiment: description of the monitoring 
    8. monitors: list of monitors to be used for the training/evaluation
    """

    # Checks, if the input body is in a JSON format:
    if not request.is_json:
        return jsonify({"status": "error", "message": "input error not json"})

    # Gets the data from the request:
    data = request.get_json()

    # Start the background job (prevents the server from being blocked by a request):
    threading.Thread(target=background_training, args=(data,)).start()
    return jsonify({"status": "ok", "message": "started"})

@bp.route('/live', methods=['POST'])
@login_required
def live():
    """
    This endpoint is used for live testing the anomaly detection &
    classification
    """
    
    # Checks, if the input body is in a JSON format:
    if not request.is_json:
        return jsonify({"status": "error", "message": "input error not json"})

    # Gets the data from the request:
    dbService = dbqueries()
    try:
        db = get_db()
    except:
        return jsonify({"status": "error", "message": "database not found"})
    
    data = request.get_json() 
    device = data['device']
    monitors = data['monitors'].split(',')
    path = data['path']
    category = "testing"
    number = 0

    # Get the paths:
    training_paths = get_paths(db, category, device, None, path)
     
    # Find the subdirectories of the path:
    subdirectories = identify_subdirectories(path)
    
    start_time_preprocessing = time.time()
    # Preprocess the data:
    preprocessed_anomaly = {}
    preprocessed_classification = {}
    for monitor in monitors:
        if monitor == "m1":
            return_dfs = m1live.preprocess_data(subdirectories, training_paths, number)
            if return_dfs == None:
                continue
            preprocessed_anomaly['m1'] = return_dfs[0]
            preprocessed_classification['m1'] = return_dfs[1]
  
        elif monitor == "m2":
            return_dfs = m2live.preprocess_data(subdirectories, training_paths, number)
            if return_dfs == None:
                continue
            preprocessed_anomaly['m2'] = return_dfs[0]
            preprocessed_classification['m2'] = return_dfs[1]

        elif monitor == "m3":
            list_of_return_dict = m3live.preprocess_data(subdirectories, training_paths, number)
            if list_of_return_dict == None:
                print(list_of_return_dict)
                continue
            anomaly_dict = list_of_return_dict[0]
            for name, df in anomaly_dict.items():
                preprocessed_anomaly[name] = df
            classification_dict = list_of_return_dict[1]
            for name, df in classification_dict.items():
                preprocessed_classification[name] = df
    stop_time_preprocessing = time.time()
    print("Preprocessing time: " + str(stop_time_preprocessing - start_time_preprocessing))
    print("Starting evaluation...")
    
    # Evaluate anomaly detection ML:
    start_time_anomaly = time.time()
    for feature, dataframe in preprocessed_anomaly.items():
        anomalyml.validate_live(dataframe, training_paths[0], feature, db, device)
    
    # Evaluate anomaly detection DL:
    for feature, dataframe in preprocessed_anomaly.items():
        threshhold = dbqueries.get_threshold(db, device, feature)
        print(threshhold)
        threshhold = float(threshhold)
        anomalydl.validate_live(dataframe, training_paths[0], feature, db, device, threshhold)
        
    # Evaluate classification ML:
    for feature, dataframe in preprocessed_classification.items():
        classification.validate_live(dataframe, training_paths[1], feature, db, device)


    return jsonify({"status": "ok", "message": "started"})




    
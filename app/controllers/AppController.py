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
    if category == 'testing' and ml_type == None:
        training_path_anomaly = dbqueries.get_training_data_location(db, device, "anomaly")
        training_path_classification = dbqueries.get_training_data_location(db, device, "classification")
        training_paths = [training_path_anomaly, training_path_classification]
    elif category == 'testing' and ml_type != None:
        training_path = dbqueries.get_training_data_location(db, device, ml_type)
        training_paths = [training_path]
    else:
        if dbqueries.get_training_data_location(db, device, ml_type) != None:
            training_path = path
        else:
            dbqueries.insert_training_data_location(db, device, ml_type, path)
            training_path = path
        training_paths = [training_path]
    return training_paths

def identify_subdirectories(path: str)-> dict:
    subdirectories = {}
    directories = os.listdir(path)
    for directory in directories:
        subdirectories[directory] = path + '/' + directory
    return subdirectories

def preprocess_data(monitors:list , subdirectories: dict, training_paths: list, category: str) -> list:
    preprocessed_data = []
    for monitor in monitors:
        if monitor == "m1":
            list_m1 = m1.preprocess_data(subdirectories)
            preprocessed_data.append(list_m1)
        elif monitor == "m2":
            list_m2 = m2.preprocess_data(subdirectories)
            preprocessed_data.append(list_m2)
        else:
            list_m3 = m3.preprocess_data(subdirectories, category, training_paths)
            preprocessed_data.append(list_m3)
    return preprocessed_data
            
def train_anomaly_detection(db: sqlite3.Connection, device: str, preprocessed_data: dict, training_path: str):
    # Training ML Algorithms:
    for feature in preprocessed_data:
        anomalyml.train(feature, training_path, device, db)
        anomalydl.train(feature, training_path, device, db)
    return "Done"

def test_anomaly_detection(db: sqlite3.Connection, device: str, preprocessed_data: dict, training_path: str, experiment: str, behavior: str):
    for feature in preprocessed_data:
        anomalyml.validate(feature, training_path, device, db, experiment, behavior)
        anomalydl.validate(feature, training_path, device, db, experiment, behavior)
    return "Done"

def train_classification(preprocessed_data: dict, training_path: str):
    for feature in preprocessed_data:
        classification.train(feature, training_path)

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
        preprocessed_data = preprocess_data(monitors, subdirectories, training_paths, category)

        # Only for Anomaly detection training:
        if ml_type == 'anomaly' and category == 'training':
            train_anomaly_detection(db, device, preprocessed_data, training_paths[0])
            
        # Only for Anomaly detection testing:
        if ml_type == "anomaly" and category == "testing":
            test_anomaly_detection(db, device, preprocessed_data, training_paths[0], experiment, behavior)

        # Only for Classification training:
        if ml_type == 'classification' and category == 'training':
             train_classification(preprocessed_data, training_paths[0])

        close_db(db)
        return "Done"
           
                
def background_live_thread(data:json):
    dbService = dbqueries()
    db = get_db()
    data = request.get_json() 
    device = data['device']
    monitors = data['monitors'].split(',')
    path = data['path']
    category = "testing"
    number = 0

    # Get the paths and subdirectories of the path:
    preprocessed_anomaly = {}
    preprocessed_classification = {}
    training_paths = get_paths(db, category, device, None, path)
    subdirectories = identify_subdirectories(path)
    
    # Preprocess the data:
   
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

    
################################# Endpoints #####################################

@bp.route('/test', methods=['GET'])
def test():
    """
    This endpoint is used to test, if the server is up!
    """
    return jsonify({'message': 'Server is up'})
    
    
@bp.route('/main', methods=['POST'])
@login_required
def training():
    """
    Main endpoint for training the ML algorithms.
    Also for testing the ML algorithms (anomaly detection and classification).
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
    data = request.get_json()
    threading.Thread(target=background_live_thread, args=(data,)).start()
    return jsonify({"status": "ok", "message": "started"})

    



    
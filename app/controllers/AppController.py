import json
import os
from flask import Blueprint
import csv
import threading
import sqlite3
import time
from flask import request, jsonify
from app import create_app
from app.db import get_db, close_db, init_app
from app.services.AuthenticationService import login_required
from app.services.SYS import SYS
from app.services.KERN import KERN
from app.services.RES import RES
from app.services.RESlive import RESlive
from app.services.KERNlive import KERNlive
from app.services.SYSlive import SYSlive
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

        if monitor == "RES":
            start_monitor_1 = time.time()
            list_RES = RES.preprocess_data(subdirectories)
            end_monitor_1 = time.time()
            with open('output.csv', 'a', newline='') as csvfile:
                fieldnames = ['Name', 'Start', 'End', 'Duration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Name': 'Preprocess RES', 'Start': float(start_monitor_1), 'End': float(end_monitor_1), 'Duration': float(end_monitor_1 - start_monitor_1)})
                csvfile.close()    
            preprocessed_data.append(list_RES)

        elif monitor == "KERN":
            start_monitor_2 = time.time()
            list_KERN = KERN.preprocess_data(subdirectories)
            end_monitor_2 = time.time()
            with open('output.csv', 'a', newline='') as csvfile:
                fieldnames = ['Name', 'Start', 'End', 'Duration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Name': 'Preprocess KERN', 'Start': float(start_monitor_2), 'End': float(end_monitor_2), 'Duration': float(end_monitor_2 - start_monitor_2)})
                csvfile.close()
            preprocessed_data.append(list_KERN)

        else:
            start_monitor_3 = time.time()
            list_SYS = SYS.preprocess_data(subdirectories, category, training_paths)
            end_monitor_3 = time.time()
            with open('output.csv', 'a', newline='') as csvfile:
                fieldnames = ['Name', 'Start', 'End', 'Duration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Name': 'Preprocess SYS', 'Start': float(start_monitor_3), 'End': float(end_monitor_3), 'Duration': float(end_monitor_3 - start_monitor_3)})
                csvfile.close()
            preprocessed_data.append(list_SYS)


    return preprocessed_data
            
def train_anomaly_detection(db: sqlite3.Connection, device: str, preprocessed_data: dict, training_path: str):
    for feature in preprocessed_data:
        anomalyml.train(feature, training_path, device, db)
        anomalydl.train(feature, training_path, device, db)
    return "Done"

def test_anomaly_detection(db: sqlite3.Connection, device: str, preprocessed_data: dict, training_path: str, experiment: str, behavior: str, path: str):
    for feature in preprocessed_data:
        anomalyml.validate(feature, training_path, device, db, experiment, behavior, path)
        anomalydl.validate(feature, training_path, device, db, experiment, behavior)
    return "Done"

def train_classification(preprocessed_data: dict, training_path: str):
    for feature in preprocessed_data:
        classification.train(feature, training_path)

def test_classification(preprocessed_data:dict, training_path: str, path: str):
    for feature in preprocessed_data:
        classification.test(feature, training_path, path)

###################### Background Threads ##############################

def background_training(data: json):
    """
    Main background thread for offline training and evaluation.
    """
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


        # Creates a .csv if the category is training:
        if category == "training":
            with open('output.csv', 'w', newline='') as csvfile:
                # Use the name: Name, start and end as columns
                fieldnames = ['Name', 'Start', 'End', 'Duration']
                # Create a writer object
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # Write the header
                writer.writeheader()
                # Close the file
                csvfile.close()

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
            test_anomaly_detection(db, device, preprocessed_data, training_paths[0], experiment, behavior, path)

        # Only for Classification training:
        if ml_type == 'classification' and category == 'training':
             train_classification(preprocessed_data, training_paths[0])
        
        if ml_type == 'classification' and category == 'testing':
            test_classification(preprocessed_data, training_paths[0],path)

        close_db(db)
        print("Done training or evaluating")
        return "Done"
    
def background_testing(data: json):
    """
    For live evaluation of data:
    """
    app = create_app()
    with app.app_context():
        
        # Get the data from the database:
        try:
            db = get_db()
        except:
            app.logger.debug('Database not found')
            return

        # Get all the data from the request:
        device = data['device']
        monitors = data['monitors'].split(',')
        path = data['path']
        number = data['number']
        print(number)
        experiment = "live"
        category = "testing"
        path = path + '/' + experiment
        # Initialize the database:
        dbService = dbqueries()
        db = get_db()

        # Get the paths and subdirectories of the path:
        preprocessed_anomaly = []
        preprocessed_classification = []
        # Gets the training path for anomaly detection + classification:
        training_paths = get_paths(db, category, device, None, path)
        # Gets the subdirectories of the path:
        subdirectories = identify_subdirectories(path)


        # Preprocess the data:
        for monitor in monitors:
            if monitor == "RES":
                try:
                    list_RES = RESlive.preprocess_data(subdirectories, number)
                    preprocessed_anomaly.append(list_RES)
                    preprocessed_classification.append(list_RES)
                except:
                    continue
    
            elif monitor == "KERN":
                try:
                    list_KERN = KERNlive.preprocess_data(subdirectories, number)
                    preprocessed_anomaly.append(list_KERN)
                    preprocessed_classification.append(list_KERN)
                except:
                    continue

            elif monitor == "SYS":
                try:
                    list_SYS = SYSlive.preprocess_data(subdirectories, number, training_paths)
                    preprocessed_anomaly.append(list_SYS[0])
                    preprocessed_classification.append(list_SYS[1])
                except:
                    continue
        
        # Anomaly detection ML / DL:
        for feature in preprocessed_anomaly:
            try:
                anomalyml.validate_live(feature, training_paths[0], device, db)
            except:
                continue
            try:
                anomalydl.validate_live(feature, training_paths[0], device, db)
            except:
                continue

        # Classification:
        for feature in preprocessed_classification:
            try:
                classification.validate_live(feature, training_paths[1], device, db)
            except:
                continue
        db.close()
        return
        

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
    This endpoint is used for live testing anomaly detection &
    classification
    """
    # Checks, if the input body is in a JSON format:
    if not request.is_json:
        return jsonify({"status": "error", "message": "input error not json"})

    # Start the background job (prevents the server from being blocked by a request):
    threading.Thread(target=background_testing, args=(request.get_json(),)).start()
    return jsonify({"status": "ok", "message": "started"})

    # END VERSION 1.0
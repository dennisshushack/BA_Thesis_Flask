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


def identify_subdirectories(path: str, behavior: str = None):
    """
    This function identifies the subdirectories of the given path.
    This differes for training and testing. 
    :input: path: path of the directory
    :return: list of subdirectories
    """
    # For testing:
    if behavior != None:
        subdirectories = {behavior: path + '/' + behavior}
        return subdirectories

    # For training:
    else:
        subdirectories = {}
        subdirectory_names = ['normal', 'poc', 'dark','raas']
        for dirs in os.walk(path):
            for subdirectory in subdirectory_names:
                if subdirectory in dirs[1]:
                    subdirectories[subdirectory] = path + '/' + subdirectory
        return subdirectories


def background(data: json):
    """
    This is the main job, that happens, if the training endpoint is called.
    As it is very process heavy the user will get informed before the job starts.
    Takes as an input the data that is sent to the endpoint.
    """
    app = create_app()
    with app.app_context():
        
        # Get the data from the database:
        try:
            db = get_db()
        except:
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
        
         # Insert the data into the database table post_requests:
        dbService = dbqueries()
        dbService.insert_into_post_requests(db, device, category, ml_type, str(monitors), behavior, path)

        # Get the training path:
        if category == 'testing':
            training_path = dbqueries.get_training_data_location(db, device, ml_type)
            testing_path = path
        else:
           if dbqueries.get_training_data_location(db, device, ml_type) != None:
               training_path = path
               testing_path = None
           else:
               dbqueries.insert_training_data_location(db, device, ml_type, path)
               training_path = path
               testing_path = None
        
        # Find the subdirectories of the given path:
        if category == 'testing':
            subdirectories = identify_subdirectories(path, behavior)
        else:
            subdirectories = identify_subdirectories(path)    

        # Start the preprocessing (for the training & testing data) Classification & Anomaly Detection:
        print("Starting preprocessing for your {category} data...".format(category=category))
        preprocessed_data = {}
        for monitor in monitors:
            if monitor == 'm1':
                df_m1, vector_behavior_m1, ids_m1 = m1.clean_data(subdirectories, begin, end)
                m1_preprocessed = m1.preprocess_data(df_m1, vector_behavior_m1, ids_m1, category, training_path, testing_path)
                preprocessed_data['m1'] = m1_preprocessed
            elif monitor == 'm2':
                df_m2, vector_behavior_m2, ids_m2 = m2.clean_data(subdirectories, begin, end)
                m2_preprocessed = m2.preprocess_data(df_m2, vector_behavior_m2, ids_m2, category, training_path, testing_path)
                preprocessed_data['m2'] = m2_preprocessed
            elif monitor == "m3":
                m3.clean_data(subdirectories,begin,end)
                dict_df = m3.get_features(subdirectories, category, training_path, testing_path)
                for key, value in dict_df.items():
                    preprocessed_data[key] = value
        print("Preprocessing finished.")
    

        # Only for Anoanomaly detection training:
        # Part Anomaly Detection ML:
        if ml_type == 'anomaly' and category == 'training':
            print("Starting anomaly detection ML training...")
            for feature, dataframe in preprocessed_data.items():
                list_anomaly_training = anomalyml.train(dataframe, training_path, feature)
                for item in list_anomaly_training:
                    for key, value in item.items():
                        model = key
                        TNR = value[0]
                        train_time = value[1]
                        dbqueries.create_ml_anomaly(db, device, feature, model, TNR, train_time)
            print("Anomaly detection ML training finished.")

            # Part Anomaly Detection DL:
            print("Starting anomaly detection DL training...")
            for feature, dataframe in preprocessed_data.items():
                dict_dl_anomaly = anomalydl.train(dataframe, training_path, feature)
                TNR = dict_dl_anomaly['TNR']
                train_time = dict_dl_anomaly['training_time']
                threshhold = dict_dl_anomaly['threshhold']
                neurons = str(dict_dl_anomaly['hidden_layers'])
                dbqueries.create_dl_anomaly(db, device, feature, "autoencoder", TNR, train_time, threshhold, neurons)
            print("Anomaly detection DL training finished.")
            print("Done with anomaly detection training.")
            return
        
        # Only for Anomaly  Detecttion testing:
        # Part Anomaly Detection ML:
        if ml_type == "anomaly" and category == "testing":
            print("Starting anomaly detection ML testing...")
            for feature, dataframe in preprocessed_data.items():
                list_anomaly_testing = anomalyml.validate(dataframe, training_path, feature)
                for item in list_anomaly_testing:
                    for key, value in item.items():
                        model = key
                        dbqueries.create_ml_anomaly_testing(db, device, experiment, behavior,feature, model, value[0], value[1])
            print("Anomaly detection ML testing finished.")
            
            # Part Anomaly Detection DL:
            print("Starting anomaly detection DL testing...")
            for feature, dataframe in preprocessed_data.items():
                threshhold = dbqueries.get_threshold(db, device, feature)
                threshhold = float(threshhold)
                model_list = anomalydl.validate(dataframe, training_path, feature, threshhold)
                dbqueries.create_dl_anomaly_testing(db, device, experiment, behavior, feature, "autoencoder", model_list[0], model_list[1])
            print("Anomaly detection DL testing finished.")
            print("Done with anomaly detection testing.")
            return

        # Only for Classification training:
        if ml_type == 'classification' and category == 'training':
            for feature, dataframe in preprocessed_data.items():
                print("Starting classification ML training...")
                classificationml.train(dataframe, training_path, feature)                
                print("Classification ML training finished.")
                

    
        

@bp.route('/test', methods=['GET'])
@login_required
def test():
    """
    This endpoint is used to test if the server is up
    """
    return jsonify({'message': 'Server is up'})
    
    
@bp.route('/main', methods=['POST'])
@login_required
def train():
    """
    This endpoint is used to preprocess the data for Machine Learning and Deep Larning models and then train
    the models. It is also used for evaluating the models.
    """
    
    # Checks, if the input body is in a JSON format:
    if not request.is_json:
        return jsonify({"status": "error", "message": "input error not json"})

    # Gets the data from the request:
    data = request.get_json()

    # Start the background job:
    threading.Thread(target=background, args=(data,)).start()

    # Creates a thread to train the models:
    return jsonify({"status": "ok", "message": "started"})







   
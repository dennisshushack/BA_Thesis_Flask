# This is specific to the frontend:
import sqlite3
import json
from app.db import get_db, close_db, init_app
from app.services.AuthenticationService import login_required
from flask import Flask, request, jsonify, render_template, current_app, redirect, url_for, flash, Blueprint
from app.database.dbqueries import dbqueries


bp = Blueprint('admin', __name__, url_prefix='/')


@bp.route('/')
@login_required
def admin_dashboard():
    return render_template("main.html")

@bp.route('/live/')
def admin_live():
    return render_template("classification.html")

# Helpers for this endpoint:
@bp.route('/get/types', methods=['GET'])
@login_required
def get_types():
    """This method gets the types"""
    try:
        db = get_db()
    except:
        pass
    
    types = dbqueries.get_devices(db)
    close_db()
    types = [tuple(row) for row in types]
    types = json.dumps(types)
    return types

@bp.route('/get/features', methods=['GET'])
@login_required
def get_features():
    """This method gets the features"""
    try:
        db = get_db()
    except:
        pass
    
    features = dbqueries.get_features(db)
    close_db()
    features = [tuple(row) for row in features]
    features = json.dumps(features)
    return features

@bp.route('/post/classification', methods=['POST','GET'])
@login_required
def get_classification():
    """This method gets the classification"""
    # Get's the json data from the request:
    device = request.form.get("device")
    feature = request.form.get("feature")
    algorithm = request.form.get("algorithm")
    try:
        db = get_db()
    except:
        pass

    classification = dbqueries.get_live_anomaly(db, device, algorithm,feature)
    close_db()
    # Make it usable for the frontend:
    data=[]
    for row in classification:
        data.append({"timestamp":row[0],"value":row[1]})
    classification = json.dumps(data)
    return classification


@bp.route('/post/anomaly', methods=['POST','GET'])
@login_required
def get_anomaly():
    """This method gets the classification"""
    # Get's the json data from the request:
    device = request.form.get("device")
    feature = request.form.get("feature")
    algorithm = request.form.get("algorithm")
    try:
        db = get_db()
    except:
        pass

    classification = dbqueries.get_live_anomaly_detection(db, device, algorithm,feature)
    close_db()
    # Make it usable for the frontend:
    data=[]
    for row in classification:
        data.append({"timestamp":row[0],"value":row[1]})
    classification = json.dumps(data)
    return classification






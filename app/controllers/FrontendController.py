# This is specific to the frontend:
import sqlite3
from app.db import get_db, close_db, init_app
from app.services.AuthenticationService import login_required
from flask import Flask, request, jsonify, render_template, current_app, redirect, url_for, flash, Blueprint


bp = Blueprint('admin', __name__, url_prefix='/')


@bp.route('/dashboard')
def admin_dashboard():
    return render_template("dashboard.html")

@bp.route('/live')
def admin_live():
    return render_template("live.html")
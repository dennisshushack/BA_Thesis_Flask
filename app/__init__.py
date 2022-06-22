import os
from flask import Flask


# Flask Factory:
def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True, static_folder='static',
            template_folder='templates')

    app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'ml-flask.sqlite'),
    BASIC_AUTH_USER='admin',
    BASIC_AUTH_PASSWORD='admin'
)

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Important imports:
    from . import db
    from app.controllers import AppController, FrontendController
    app.register_blueprint(AppController.bp)
    app.register_blueprint(FrontendController.bp)
    db.init_app(app)
    
    return app

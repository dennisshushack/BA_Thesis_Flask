from flask import request, Response, current_app
from functools import wraps
import base64


def check(authorization_header):
    username = current_app.config['BASIC_AUTH_USER']
    password = current_app.config['BASIC_AUTH_PASSWORD']
    encoded_uname_pass = authorization_header.split()[-1].encode("utf-8")
    if encoded_uname_pass == base64.b64encode((username + ":" + password).encode("utf-8")):
        return True


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        authorization_header = request.headers.get('Authorization')
        if authorization_header and check(authorization_header):
            return f(*args, **kwargs)
        else:
            resp = Response()
            resp.headers['WWW-Authenticate'] = 'Basic'
            return resp, 401

    return decorated_function

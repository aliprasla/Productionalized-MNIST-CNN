"""
This file contains the Flask Interface used to make API requests
"""
from email import header
import logging
import os
import requests


from flask import Flask
from flask_restful import Api


from api.resources.resources_predict import PredictResource
from api.resources.resources_home import HomeResource

LOGGER = logging.getLogger(__name__)
# load environment variables
PORT = int(os.environ.get('PORT'))
DEBUG = os.environ.get('DEBUG_MODE') == "True"
HOST = os.environ.get("HOST_IP")
LOGLEVEL = os.environ.get("LOGLEVEL").upper()

APP = Flask(__name__)
API = Api(APP)
logging.basicConfig(level=LOGLEVEL)



API.add_resource(HomeResource,'/')
API.add_resource(PredictResource,'/predict')


if __name__ == "__main__":
    LOGGER.info("HOST IP: {}".format(HOST))
    LOGGER.info("PORT: {}".format(PORT))

    
    APP.run(port=PORT, host=HOST, debug=DEBUG)
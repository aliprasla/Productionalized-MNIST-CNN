"""
This file contains the Flask Interface used to make API requests
"""
import logging
import os
import requests


from flask import Flask
from flask_restful import Api


from api.resources.resources_predict import PredictResource



LOGGER = logging.getLogger(__name__)
# load environment variables
PORT = int(os.environ.get('PORT'))
DEBUG = os.environ.get('DEBUG_MODE') == "True"
HOST = os.environ.get("HOST_IP")
LOGLEVEL = os.environ.get("LOGLEVEL").upper()
API_VERSION = os.environ.get("API_VERSION")

APP = Flask(__name__)
API = Api(APP)
logging.basicConfig(level=LOGLEVEL)


API.add_resource(PredictResource,'/{}/predict'.format(API_VERSION))


if __name__ == "__main__":
    LOGGER.info("HOST IP: {}".format(HOST))
    LOGGER.info("API_VERSION: {}".format(API_VERSION))
    LOGGER.info("PORT: {}".format(PORT))

    
    APP.run(port=PORT, host=HOST, debug=DEBUG)
    # train model
    requests.post("localhost:{}/{}/train".format(PORT,API_VERSION))
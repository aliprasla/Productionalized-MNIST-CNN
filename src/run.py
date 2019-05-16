"""
This file contains the Flask Interface used to make API requests
"""
import logging
import os

from flask import Flask
from flask_restful import Api

from api.resources.resources_train import TrainResource



LOGGER = logging.getLogger(__name__)
# load environment variables
PORT = int(os.environ.get('PORT'))
DEBUG = os.environ.get('DEBUG_MODE') == "True"
HOST = os.environ.get("HOST_IP")
LOGLEVEL = os.environ.get("LOGLEVEL").upper()


APP = Flask(__name__)
API = Api(APP)
logging.basicConfig(level=LOGLEVEL)

API.add_resource(TrainResource, '/train')


if __name__ == "__main__":
    APP.run(port=PORT, host=HOST, debug=DEBUG)

"""
This file includes prediction resources
"""
import logging
import os
import json
import keras
import traceback

from flask import request
from flask_restful import Resource

import numpy as np
import app.database as db



LOGGER = logging.getLogger(__name__)
LOGLEVEL = os.environ.get("LOGLEVEL").upper()
logging.basicConfig(level=LOGLEVEL)


class PredictResource(Resource):
    """
    Prediction Endpoint for Flask API 
    """
    feature_length = int(os.environ['MNIST_FEATURE_LENGTH'])
    model_file_name = "mnist_model.pkl"
    pixel_values_key = 'pixel_values'
    def post(self):
        """
        Returns Predictions upon the trained MNIST classifier


        Expect JSON Payload:

        {
            'prediction_data':[2D array representing gray scale numbers]
        }
        Return JSON payload with specifications if successful:

        {
            'class': (predicted_class_number)
        }
        """

        request_data = json.loads(request.data)

        # TODO: add basic input validation

        # process into array
        feature_array = request_data.get(self.pixel_values_key)

        x_pred = np.array(feature_array).flatten().reshape(1, -1)

        # once this is complete, return data
        try:
            # TODO: Add more testing for this
            LOGGER.info("Attempting to Load Model")

            model = db.load_model(self.model_file_name)

            LOGGER.info("Model Loaded Successfully.")

            LOGGER.info("Beginning Prediction")

            prediction = model.predict(x_pred)

            LOGGER.info("Finished Prediction")
            
            assert prediction.shape[0] == 1, "Too much data in request"
        
        except Exception as e:
        
            LOGGER.info("Model prediction failed. Traceback: {}".format(traceback.format_exc()))
            keras.backend.clear_session()
            return {"message":"Model prediction failed. Error: {}".format(str(e))},500

        

        return {'class': prediction[0]}, 200

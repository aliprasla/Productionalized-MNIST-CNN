"""
This file includes prediction resources
"""
import logging
import os
import json
import keras
import tensorflow as tf
import traceback

from flask import request
from flask_restful import Resource

from app.models.cnn import MNISTClassifer

import numpy as np
import app.database as db



LOGGER = logging.getLogger(__name__)
LOGLEVEL = os.environ.get("LOGLEVEL").upper()
logging.basicConfig(level=LOGLEVEL)


class PredictResource(Resource):
    """
    Prediction Endpoint for Flask API 
    """
    model_file_name = "mnist_model.json"
    pixel_values_key = 'prediction_data'
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
        LOGGER.info("Beginning Model Prediction")
        
        LOGGER.info(str(request.data))
        request_data = json.loads(request.data)


        # TODO: add basic input validation

        # process into array
        feature_array = json.loads(request_data.get(self.pixel_values_key))
        # find number of records to predict
        x_pred = np.array(feature_array)
      

        # once this is complete, return data
        try:
            # TODO: Add more testing for this
            LOGGER.info("Attempting to Load Model")

            model_config = db.load_model(self.model_file_name)
            
            with tf.Graph().as_default():
                
                model = MNISTClassifer.from_json(model_config)
                
                LOGGER.info("Model Loaded Successfully.")
                
                LOGGER.info("Beginning Prediction")


                prediction = model.predict(x_pred)

            keras.backend.clear_session()


            LOGGER.info("Finished Prediction")
            
            assert prediction.shape[0] == 1, "Too much data in request"
        
        except Exception as e:
        
            LOGGER.info("Model prediction failed. Traceback: {}".format(traceback.format_exc()))
            keras.backend.clear_session()
            return {"message":"Model prediction failed. Error: {}".format(str(e))},500

        

        return {'class': prediction[0]}, 200, {'Access-Control-Allow-Origin':'*'}

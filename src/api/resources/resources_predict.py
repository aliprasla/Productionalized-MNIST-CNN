import logging
import os
import time
import json

from flask import request
from flask_restful import Resource

import app.database as db
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGLEVEL = os.environ.get("LOGLEVEL").upper()
logging.basicConfig(level=LOGLEVEL)

class PredictResource(Resource):
    """
    Prediction Endpoint for Flask API 
    """
    feature_length = int(os.environ['MNIST_FEATURE_LENGTH'])

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
        feature_array = request_data.get('pixel_values')

        num_rows_of_data = int(np.sqrt(self.feature_length))
        
        flattened_feature_array = np.array(feature_array).flatten()


        x_pred = flattened_feature_array.reshape(len(flattened_feature_array),1)
        # once this is complete, return data

        #TODO: Add more testing for this 
        model = db.load_model("mnist_model.pkl")

        prediction = model.predict(flattened_feature_array)

        assert prediction.shape[0] == 1

        return { 'class': prediction[0] }, 200


        
"""
This file contains testing code for the API

"""
import time
import os
import unittest
import numpy as np


from run_api import APP


class APIEndpointTesting(unittest.TestCase):
    """
    Send point for testing the API call and response.
    """
    # number of features for Sklearn
    feature_length = int(os.environ['MNIST_FEATURE_LENGTH'])
    max_seconds_allowed_for_prediction = float(
        os.environ['MAX_SECONDS_ALLOWED_FOR_PREDICTION'])

    def setUp(self):
        """
        Run API Locally
        """
        self.app = APP.test_client()
        # propagate the exceptions to the test client
        self.app.testing = True

    def tearDown(self):
        pass

    def test_model_training(self):
        """
        Tests model training endpoint
        """
        result = self.app.post('/train')
        assert result.status_code == 200, "Training failed. Server response: {}".format(
            result)

    def test_model_prediction(self):
        """
        Tests a dummy model prediction
        """

        # calculate size of one dimension

        rows_in_picture = int(np.sqrt(self.feature_length))
        x_pred = np.array([0]*self.feature_length)
        x_pred = x_pred.reshape(rows_in_picture, rows_in_picture).tolist()


        # create prediction data -
        request_data = {
            "pixel_values": x_pred
        }
        beg = time.time()
        result = self.app.post('/predict', json=request_data)
        total_time_for_prediction = time.time() - beg

        assert result.status_code == 200, "Normal prediction endpoint failed"

        if total_time_for_prediction > self.max_seconds_allowed_for_prediction:
            raise TimeoutError(
                "Prediction for request data, {} exceeded the max seconds allowed for prediction.".format(request_data))

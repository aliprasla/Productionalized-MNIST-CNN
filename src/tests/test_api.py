"""
This file contains testing code for the API

"""
import unittest

from run_api import APP


class APIEndpointTesting(unittest.TestCase):
    """
    Send point for testing the API call and response.
    """

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

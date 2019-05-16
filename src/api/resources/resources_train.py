import logging
import os

from flask_restful import Resource
from flask import request

from app.models.cnn import MNISTClassifer
import app.database as db


LOGGER = logging.getLogger(__name__)
LOGLEVEL = os.environ.get("LOGLEVEL").upper()
logging.basicConfig(level=LOGLEVEL)


class TrainResource(Resource):
    """
    Training Endpoint for Flask API 
    """

    def post(self):
        """
        Post request for /train. Used to re-train model
        
        """
        #Will be used later
        request_data = request

        # re-run model training endpoint here

        # load training data
        LOGGER.info("Retraining Model")

        try:
            training_data = db.load_training_data("mnist_training.pkl")

            x_train = training_data['X']
            y_train = training_data['y']

            # free up memory space
            training_data = None

            model = MNISTClassifer()

            model.fit(x_train=x_train, y_train=y_train,
                      batch_size=200,
                      epochs=5,
                      verbose=2,
                      validation_split=0.2)

            db.save_model(model, "mnist_model.pkl")

        except Exception as exception:
            return str(exception),500

        return "Successful", 200

"""
This file contains the Flask Interface used to make API requests
"""
import logging
import os

from flask import Flask
from src.app.models.cnn import MNISTClassifer
import src.app.database as db


LOGGER = logging.getLogger(__name__)
# load environment variables
PORT = int(os.environ.get('PORT'))
DEBUG = os.environ.get('DEBUG_MODE') == "True"
HOST = os.environ.get("HOST_IP")
LOGLEVEL = os.environ.get("LOGLEVEL").upper()


APP = Flask(__name__)
logging.basicConfig(level=LOGLEVEL)


@APP.route("/train", methods=["POST"])
def train_model():
    """
    Retrain model endpoint - retrain model based upon post request
    """
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
        return str(exception)

    return "Successful"


if __name__ == "__main__":
    APP.run(port=PORT, host=HOST, debug=DEBUG)

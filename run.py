"""
This file contains the Flask Interface used to make API requests
"""
import logging
import os

from flask import Flask, render_template
from src.app.models.cnn import MNIST_Classifer
import src.app.database as db





LOGGER = logging.getLogger(__name__)
# load environment variables
PORT = int(os.environ.get('PORT'))
DEBUG = True if os.environ.get('DEBUG_MODE') == "True" else False
HOST = os.environ.get("HOST_IP")
LOGLEVEL = os.environ.get("LOGLEVEL").upper()


APP = Flask(__name__)
logging.basicConfig(level=LOGLEVEL)


@APP.route("/train", methods=["POST"])
def train_model():
    # re-run model training endpoint here

    # load training data
    LOGGER.info("Retraining Model")

    try:
        training_data = db.load_training_data("mnist_training.pkl")

        x_train = training_data['X']
        y_train = training_data['y']

        # free up memory space
        training_data = None

        model = MNIST_Classifer()

        model.fit(X=x_train, y=y_train,
                  batch_size=200,
                  epochs=5,
                  verbose=2,
                  validation_split=0.2)

        db.save_model(model, "mnist_model.pkl")

    except Exception as e:
        message = str(e)
        return message

    return "Successful"


if __name__ == "__main__":
    APP.run(port=PORT, host=HOST, debug=DEBUG)

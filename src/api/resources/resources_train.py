import logging
import os
import keras
import traceback

from flask_restful import Resource

from app.models.cnn import MNISTClassifer
import app.database as db


LOGGER = logging.getLogger(__name__)
LOGLEVEL = os.environ.get("LOGLEVEL").upper()
logging.basicConfig(level=LOGLEVEL)


class TrainResource(Resource):
    """
    Training Endpoint for Flask API 
    """
    training_data_name_file_name = "mnist_training.pkl"
    model_file_name = "mnist_model.json"

    def post(self):
        """
        Post request for /train. Used to re-train model

        """
        # re-run model training endpoint here

        # load training data
        LOGGER.info("Retraining Model")

        try:
            LOGGER.info("Loading Training Data")

            training_data = db.load_training_data(
                self.training_data_name_file_name)

            LOGGER.info("Training Data Loaded Successfully")
            x_train = training_data['X']
            y_train = training_data['y']

            # free up memory space
            training_data = None

            model = MNISTClassifer()

            validation_accuracy = model.fit(x_train=x_train, y_train=y_train,
                      batch_size=200,
                      epochs=5,
                      verbose=2,
                      validation_split=0.2)
            
            db.save_model(model, self.model_file_name)
            

        except Exception as exception:

            keras.backend.clear_session()
            LOGGER.info('Training Failed. Traceback:\n {}'.format(traceback.format_exc()))
            return {"message":str(exception)}, 500
        
        keras.backend.clear_session()
        return {"message":"Training_Successful","metrics":{"validation_accuracy":validation_accuracy}}, 200

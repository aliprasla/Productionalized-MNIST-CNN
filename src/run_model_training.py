import logging
import keras
import traceback
import numpy as np


from app.models.cnn import MNISTClassifer
import app.database as db


LOGGER = logging.getLogger(__name__)
# re-run model training endpoint here

# load training data
LOGGER.info("Retraining Model")

MODEL_FILE_NAME = "mnist_model.json"
np.random.seed(10)

if __name__ == "__main__": 

    try:      
        LOGGER.info("Loading Training Data")

        x_train, y_train = db.load_training_data()

        LOGGER.info("Training Data Loaded Successfully")

        model = MNISTClassifer()

        validation_accuracy = model.fit(x_train=x_train,
                                        y_train=y_train,
                                        batch_size=200,
                                        epochs=5,
                                        verbose=2,
                                        validation_split=0.2)

        LOGGER.info("Model Training Successful. Validation Accuracy: {}".format(validation_accuracy))
        LOGGER.info("Attempting to save model to directory")
        
        db.save_model(model, MODEL_FILE_NAME)
        
        LOGGER.info("Model Saving Successful.")

    except Exception as exception:

        keras.backend.clear_session()
        LOGGER.info('Training Failed. Traceback:\n {}'.format(traceback.format_exc()))
        raise Exception("Model Training Failed.")
    
    keras.backend.clear_session()

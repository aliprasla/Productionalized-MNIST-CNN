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

if __name__ == "__main__": 

    try:      
        np.random.seed(10)

        LOGGER.info("Loading Training Data")

        x_train, y_train = db.load_training_data()

        LOGGER.info("Training Data Loaded Successfully")

        model = MNISTClassifer(
            conv_layer_one_filters=35,
            conv_layer_one_kernel_size = (3,3),
            dense_layer_one_num_units = 40)

        # process data to have x_train be binary


        validation_accuracy = model.fit(x_train=x_train,
                                        y_train=y_train,
                                        batch_size=300,
                                        epochs=7,
                                        verbose=2,
                                        validation_split=0.1)

        LOGGER.info("Model Training Successful. Validation Accuracy: {}".format(validation_accuracy))
        LOGGER.info("Attempting to save model to directory")
        
        db.save_model(model, MODEL_FILE_NAME)
        
        LOGGER.info("Model Saving Successful.")

    except Exception as exception:

        keras.backend.clear_session()
        LOGGER.info('Training Failed. Traceback:\n {}'.format(traceback.format_exc()))
        raise Exception("Model Training Failed.")
    
    keras.backend.clear_session()

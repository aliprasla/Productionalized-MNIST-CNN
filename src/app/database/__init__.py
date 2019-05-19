"""
This file is the DAL of the app. Reads and writes to directories. 

Implements an abstraction layer to minimize vendor lock.

"""

import os
import pickle
import tensorflow as tf
from pathlib import Path

from app.database.google_cloud_storage import GoogleCloudStorage

# load environment variables

DATA_LOCATION = os.environ.get('DATA_LOCATION')
GCP_DATA_LOCATION_NAME = "gcp"
LOCAL_DATA_LOCATION_NAME = "local"
def load_training_data(data_name):
    """
    Loads training data from directory
    """
    f_path = _get_training_data_directory()/data_name
    
    #load training data locally
    return _load_pickle_file(f_path,data_location_var=LOCAL_DATA_LOCATION_NAME)


def save_model(model_object, model_name: str):
    """
    Save model file to local "data" directory
    """
    f_path = _get_model_directory() / model_name

    _save_pickle_file(model_object, f_path,data_location_var=DATA_LOCATION)


def load_model(model_name: str):
    """
    Loads a pickled model file from a directory. NOTE: Change to handle multiple data sources
    """
    f_path = _get_model_directory() / model_name
    
    return _load_pickle_file(f_path,data_location_var=DATA_LOCATION)


############################
# PRIVATE METHODS
############################
def _get_training_data_directory():
    return _get_data_directory() / "training_data"


def _get_model_directory():
    return _get_data_directory() / "models"


def _get_data_directory():
    return Path("data")


def _load_pickle_file(f_path,data_location_var:str):
    """
    Load pickle pickle depending upon which data location was specified
    
    Args:
        f_path (Pathlib) : file location of the pickle object you want to load
        data_location_var (str) : Data Location variable used to identify storage choice
    """
    global graph
    graph = tf.get_default_graph()
    # handle for multi-threading in flask. See https://stackoverflow.com/questions/51127344/tensor-is-not-an-element-of-this-graph-deploying-keras-model for more details.

    with graph.as_default():
        if data_location_var == LOCAL_DATA_LOCATION_NAME:
            return pickle.load(open(f_path, 'rb'))

        elif data_location_var == GCP_DATA_LOCATION_NAME:
            storage = GoogleCloudStorage()
            return storage.load_pickle_file(f_path=f_path)

        else:
            raise NotImplementedError("DATA_LOCATION, {}, not valid. Cannot load pickle file, {}".format(DATA_LOCATION,f_path))


    


def _save_pickle_file(obj:object, f_path,data_location_var):
    """ 
    Saves pickle file to f_path at data_location_var type

    Args:
        obj (Object) : Generic object you wish to save
        f_path (Path ): pathlib path object details where you want to save the object
        data_location_var: Data Location type

    """
    global graph
    graph = tf.get_default_graph()

    with graph.as_default():
        if data_location_var == LOCAL_DATA_LOCATION_NAME:
            # local data
            pickle.dump(obj, open(f_path, 'wb'))


        elif data_location_var == GCP_DATA_LOCATION_NAME:
            # GCP save file
            storage = GoogleCloudStorage()
            return storage.save_pickle_file(obj=obj,f_path=f_path)

        else:
            raise NotImplementedError("DATA_LOCATION, {}, not valid. Cannot load model".format(DATA_LOCATION))

        

 

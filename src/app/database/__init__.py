"""
This file is the DAL of the app. Reads and writes to directories. 

Implements an abstraction layer to minimize vendor lock.

"""

import os
import numpy as np
import pickle
import json


from pathlib import Path

from tensorflow.keras.datasets import mnist

# load environment variables


def load_training_data():
    """
    Loads training data from mnist file
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.append(x_train,x_test,axis=0)
    y_train = np.append(y_train, y_test,axis=0)
    return x_train,y_train

def save_model(model_object, model_name: str):
    """
    Save model file to local "data" directory
    """
    f_path = _get_model_directory() / model_name
    
    json_file = model_object.to_json()
    
    _save_json_file(json_file, f_path)


def load_model(model_name: str):
    """
    Loads a pickled model file from a directory. NOTE: Change to handle multiple data sources
    """
    f_path = _get_model_directory() / model_name
    
    return _load_json_file(f_path)


############################
# PRIVATE METHODS
############################
def _get_training_data_directory():
    return _get_data_directory() / "training_data"


def _get_model_directory():
    return _get_data_directory() / "models"


def _get_data_directory():
    return Path("data")

def _load_json_file(f_path):
    """
    Load pickle pickle depending upon which data location was specified
    
    Args:
        f_path (Pathlib) : file location of the pickle object you want to load
        data_location_var (str) : Data Location variable used to identify storage choice
    """

    
    return json.load(open(f_path, 'r'))


def _save_json_file(obj_as_json, f_path):
    """ 
    Saves pickle file to f_path at data_location_var type

    Args:
        obj_as_json (list or dict) : Generic object you wish to save
        f_path (Path ): pathlib path object details where you want to save the object
        data_location_var: Data Location type

    """
    
    # local data
    json.dump(obj_as_json, open(f_path, 'w'))
    

 

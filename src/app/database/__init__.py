"""
This file is the DAL of the app. Reads and writes to directories
"""

import os
import pickle
from pathlib import Path


def load_training_data(data_name):
    """
    Loads training data from directory
    """
    f_path = _get_training_data_directory()/data_name

    return _load_pickle_file(f_path)


def save_model(model_object, model_name: str):
    """
    Save model file to local "data" directory
    """
    f_path = _get_model_directory() / model_name
    _save_pickle_file(model_object, f_path)


def load_model(model_name: str):
    """
    Loads a pickled model file from a directory. NOTE: Change to handle multiple data sources
    """
    f_path = _get_model_directory() / model_name
    return _load_pickle_file(f_path)


############################
# PRIVATE METHODS
############################
def _get_training_data_directory():
    return _get_data_directory() / "training_data"


def _get_model_directory():
    return _get_data_directory() / "models"


def _get_data_directory():
    return Path("data")


def _load_pickle_file(f_path):
    return pickle.load(open(f_path, 'rb'))


def _save_pickle_file(model_object, f_path):
    pickle.dump(model_object, open(f_path, 'wb'))

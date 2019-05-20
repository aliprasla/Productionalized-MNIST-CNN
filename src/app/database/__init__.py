"""
This file is the DAL of the app. Reads and writes to directories. 

Implements an abstraction layer to minimize vendor lock.

"""

import os
import pickle
import json

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
    
    json_file = model_object.to_json()
    
    _save_json_file(json_file, f_path,data_location_var=DATA_LOCATION)


def load_model(model_name: str):
    """
    Loads a pickled model file from a directory. NOTE: Change to handle multiple data sources
    """
    f_path = _get_model_directory() / model_name
    
    return _load_json_file(f_path,data_location_var=DATA_LOCATION)


############################
# PRIVATE METHODS
############################
def _get_training_data_directory():
    return _get_data_directory() / "training_data"


def _get_model_directory():
    return _get_data_directory() / "models"


def _get_data_directory():
    return Path("data")

#TODO: Make these functions more DRY.

def _load_pickle_file(f_path,data_location_var=DATA_LOCATION):
    if data_location_var == LOCAL_DATA_LOCATION_NAME:
        return pickle.load(open(f_path,'rb'))

    elif data_location_var == GCP_DATA_LOCATION_NAME:

        storage = GoogleCloudStorage()
        raw_string = storage.load_file_as_string(f_path)

        return pickle.loads(raw_string)

    else:
        raise NotImplementedError("DATA_LOCATION, {} not valid".format(data_location_var))

def _save_pickle_file(object,f_path,data_location_var=DATA_LOCATION):

    if data_location_var == LOCAL_DATA_LOCATION_NAME:
        return pickle.dump(open(f_path),'rb')

    elif data_location_var == GCP_DATA_LOCATION_NAME:

        storage = GoogleCloudStorage()
        storage.save_file_as_string(obj_as_string=pickle.dumps(obj),f_path=f_path)

        return pickle.loads(raw_string)

    else:
        raise NotImplementedError("DATA_LOCATION, {} not valid".format(data_location_var))

def _load_json_file(f_path,data_location_var=DATA_LOCATION):
    """
    Load pickle pickle depending upon which data location was specified
    
    Args:
        f_path (Pathlib) : file location of the pickle object you want to load
        data_location_var (str) : Data Location variable used to identify storage choice
    """

    if data_location_var == LOCAL_DATA_LOCATION_NAME:
        return json.load(open(f_path, 'r'))

    elif data_location_var == GCP_DATA_LOCATION_NAME:

        storage = GoogleCloudStorage()
        raw_string = storage.load_file_as_string(f_path=f_path)
        return json.loads(raw_string)

    else:
        raise NotImplementedError("DATA_LOCATION, {}, not valid. Cannot load pickle file, {}".format(DATA_LOCATION,f_path))



def _save_json_file(obj_as_json, f_path,data_location_var=DATA_LOCATION):
    """ 
    Saves pickle file to f_path at data_location_var type

    Args:
        obj_as_json (list or dict) : Generic object you wish to save
        f_path (Path ): pathlib path object details where you want to save the object
        data_location_var: Data Location type

    """
    
    if data_location_var == LOCAL_DATA_LOCATION_NAME:
        # local data
        pickle.dump(obj_as_json, open(f_path, 'wb'))


    elif data_location_var == GCP_DATA_LOCATION_NAME:

        # GCP save file
        storage = GoogleCloudStorage()
        return storage.save_file_as_string(obj_as_string=json.dumps(obj_as_json),f_path=f_path)

    else:
        raise NotImplementedError("DATA_LOCATION, {}, not valid. Cannot load model".format(DATA_LOCATION))

        

 

"""
This file contains classes for reading and writing to/from Google Cloud Storage.

For more information on the docs for this:
https://googleapis.github.io/google-cloud-python/latest/storage
"""
import os
import pickle
import json
import logging
import requests

from google.cloud import storage
from google.auth.credentials import Credentials


LOGGER = logging.getLogger(__name__)
LOGLEVEL = os.environ.get("LOGLEVEL").upper()
logging.basicConfig(level=LOGLEVEL)

class GoogleCloudStorage:
    def __init__(self):
        """
        Loads environment variables into object
        """
        self.bucket_name = os.environ.get('GCP_BUCKET_NAME')
 
    def load_file_as_string(self,f_path):
        """
        Loads a pickle file located at the f_path
        
        Args:
            f_path (Pathlib) : file location of data on the GCP bucket

        Returns:
            File downloaded stored at that location
        """
        # get gcp bucket
        bucket = self._get_bucket()

        # download data from blob location

        blob = bucket.blob(str(f_path))

        return blob.download_as_string()


    def save_file_as_string(self,obj_as_string:str,f_path):
        """
        Saves a pickled version of obj at f_path

        Args:

            obj_as_string (str) : what object you want to save on the cloud bucket. Already serialized.
            f_path (Pathlib) : where do you want to save the file

        Returns:
            None
        """
        # get bucket where you want to upload the object
        bucket = self._get_bucket()

        # get blob location upon which to upload the object
        blob = bucket.blob(str(f_path))

        # upload the file
        blob.upload_from_string(data=obj_as_string)

    def _get_bucket(self):
        """
        Returns the GCP bucket object 

        For more information on this object, check out:
        
        https://googleapis.github.io/google-cloud-python/latest/storage/buckets.html#google.cloud.storage.bucket.Bucket
        
        Args:
            None

        Returns:
            Google Cloud Bucket object
        """
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)

        return bucket


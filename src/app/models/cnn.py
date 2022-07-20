"""
This file contains Convolutional Neural Network based classes for MNIST.

For more information on why default graph declaration is important via TensorFlow, view this thread:
https://github.com/tensorflow/tensorflow/issues/14356

"""
from decimal import localcontext
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, Flatten
from keras.models import Model
from keras.utils import to_categorical

import keras
import tensorflow as tf
import numpy as np
import os

class MNISTClassifer:
    """
    Author: Ali Prasla

    This class presents a wrapper for a Keras MNIST classifer.

    It contains the CNN architecture necessary to accurately make predictions.
        This is an example and is a relatively rigid architecture

    """

    # in this example we are using the basic MNIST Dimensions
    

    # MNIST has fixed dimensions
    num_classes = 9
    num_pixel_rows_per_image = 28
    num_pixel_columns_per_image = 28

    def __init__(self,
                 conv_layer_one_filters=20,
                 conv_layer_one_kernel_size=(2, 2),
                 conv_layer_one_activation='relu',
                 max_pooling_layer_one_pool_size=(2, 2),
                 dense_layer_one_num_units=20,
                 dense_layer_one_activation='relu',
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy']):

        # define hyper parameter

        self.conv_layer_one_filters = conv_layer_one_filters
        self.conv_layer_one_kernel_size = conv_layer_one_kernel_size
        self.conv_layer_one_activation = conv_layer_one_activation

        self.max_pooling_layer_one_pool_size = max_pooling_layer_one_pool_size

        self.dense_layer_one_num_units = dense_layer_one_num_units
        self.dense_layer_one_activation = dense_layer_one_activation


        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = None
        self.training_history = None

    def fit(self, x_train, y_train, batch_size: int, **kwargs):
        """
        Fits the underlying keras model

        Args:
                x_train (Array like) : X Training Data
                y_train (Array like) : y target values
                batch_size (int) :  number of sample per gradient update

                **kwargs : additional parameters passed to the keras .fit method

        Returns:
            Final Validation Accuracy for the model

        """

        if y_train.ndim == 1:
            y_one_hot = to_categorical(y_train, num_classes=self.num_classes + 1)
        else:
            assert y_one_hot.shape[1] == self.num_classes
            y_one_hot = y_train


        self.model = self._get_untrained_model()

        self.model.compile(optimizer=self.optimizer,
                        loss=self.loss, metrics=self.metrics)
        
        # process data to have x_train be binary

        x_train = x_train > 0

        import code
        code.interact(local=locals())

        history = self.model.fit(x=x_train, y=y_one_hot,
                    batch_size=batch_size,
                    **kwargs)
        

        return history.history['val_accuracy'][-1]

    def to_json(self):
        """
        Returns a json_dictionary of this object. 
        """

        # get model config and weights
        

        # get other parameters
        out_dict = {}

        out_dict['model_config'] = self.model.get_config()
        
        out_dict['model_weights'] = [layer_weight.tolist() for layer_weight in self.model.get_weights()]



        return out_dict

    @classmethod
    def from_json(self,model_json:dict):

        model_config = model_json['model_config']
        model_weights = model_json['model_weights']
        
        new_obj = MNISTClassifer()

        for key,value in model_json.items():
            if key not in ['model_config','model_weights']:
                setattr(new_obj,key,value)

        
        # set model
        new_obj.model = Model.from_config(model_config)

        # convert weights nested list to numpy array
        model_weights = [np.array(x) for x in model_weights]

        new_obj.model.set_weights(model_weights)
        
        return new_obj

        

    def predict(self, x_predict):
        """
        Runs model prediction

        Args:
                x_predict (Array like) : Data upon which you want to make predictions
        """

        # TODO: add validation testing
        x_predict = x_predict > 0 
        
        # run prediction
        y_pred = self.model.predict(x_predict)

        # convert softmax floats into
        construct = np.argmax(y_pred, axis=1)

        return construct.astype(str)

    def _get_untrained_model(self):
        """
        Defines Model Architecture using Keras' Functional API

        """

        # input layer with dimensions

        # assuming square images for MNIST
       
        input_layer = Input(shape=(self.num_pixel_rows_per_image, self.num_pixel_columns_per_image,))

        model = Reshape((self.num_pixel_rows_per_image, self.num_pixel_columns_per_image,1))(input_layer)

        # define convolutional layers

        model = Conv2D(filters=self.conv_layer_one_filters,
                       kernel_size=self.conv_layer_one_kernel_size,
                       activation=self.conv_layer_one_activation)(model)

        model = MaxPooling2D(
            pool_size=self.max_pooling_layer_one_pool_size)(model)

        model = Flatten()(model)

        model = Dense(units=self.dense_layer_one_num_units,
                      activation=self.dense_layer_one_activation)(model)

        prediction_layer = Dense(
            # add one for one hot encoding purposes
            units=self.num_classes + 1, activation="softmax", name="output_layer")(model)

        return Model(inputs=input_layer, outputs=prediction_layer)

#!/usr/bin/python
# This code was developed using https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970
# Build the model of a logistic classifier
import os
import sys
import pdb
import gzip
import six.moves.cPickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
import tensorflow as tf

def init():
   # load model data in json
   dr_path = os.path.dirname(os.path.realpath(__file__))
   model_json_file = open(dr_path + '\model_data_file.json', 'r')
   loaded_model_json = model_json_file.read()
   model_json_file.close()
   loaded_model = model_from_json(loaded_model_json)
   # load weights
   loaded_model.load_weights(dr_path + '\DL_weights.h5')
   print(loaded_model)
   # evaluate loaded model on test data
   loaded_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
   graph = tf.get_default_graph()
   return loaded_model, graph
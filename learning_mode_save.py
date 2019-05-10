#!/usr/bin/python
# This code was developed using https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970
# Build the model of a logistic classifier
import os
import gzip
import pdb
import sys
import six.moves.cPickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

def build_logistic_model(input_shape, output_dim):
    model = Sequential()
    model.add(Conv2D(40, (2, 2), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(80, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='relu'))
    return model

batch_size = 128
nb_classes = 10
nb_epoch = 20
# input_dim = 784
img_x, img_y = 28, 28
input_shape = (img_x, img_y, 1)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(60000, input_dim)
# X_test = X_test.reshape(10000, input_dim)

X_train = X_train.reshape(X_train.shape[0], img_x,img_y,1)
X_test = X_test.reshape(X_test.shape[0], img_x,img_y,1)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = build_logistic_model(input_shape, nb_classes)

model.summary()
# compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)


print('Test score:', score[0])
print('Test accuracy:', score[1])


model_json = model.to_json()
model_json_data_file = open("model_data_file.json","w")
model_json_data_file.write(model_json)
model_json_data_file.close()
dr_path = os.path.dirname(os.path.realpath(__file__))
model.save_weights(dr_path + '\DL_weights.h5')
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
from keras import optimizers


def build_logistic_model(input_shape, output_dim):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(88, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

batch_size = 128
nb_classes = 10
nb_epoch = 15
input_dim = 784
img_x, img_y = 28, 28
input_shape = (img_x, img_y, 1)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
Train_sample_X = np.vsplit(X_train, [10000,60000])
Train_sample_Y = np.split(y_train, [10000,60000])
X_train = Train_sample_X[1]
X_train = X_train.reshape(50000, img_x, img_y, 1)
X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.reshape(10000, img_x, img_y, 1)
X_test = X_test.astype('float32')
X_test /= 255

X_val = Train_sample_X[0]
X_val = X_val.reshape(10000, img_x, img_y, 1)
X_val = X_val.astype('float32')
X_val /= 255
Y_train = Train_sample_Y[1]
Y_val = Train_sample_Y[0]

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = build_logistic_model(input_shape, nb_classes)

model.summary()
# compile the model
sgd = optimizers.SGD(lr=0.1, decay=0.0, momentum=0.1, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, Y_val))
					
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

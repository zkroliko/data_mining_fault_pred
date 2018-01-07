from __future__ import print_function

import fileinput
import os

import math
import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import RMSprop
from keras.models import model_from_json

from data_utils import load_processed_data
from transform_input import make_timeseries_instances
from warp_labels import warp_labels

import sys

WINDOW_SIZE = 40
PREDICTION_LENGTH = 1000

TEST_ROWS = 0

PCA_TARGET_SIZE = 10


# For predicting a single instance from a file
def main(argv):
    if len(argv) < 4:
        print("Correct arguments: <model> <weights> <data_file> |-<custom>|")
        exit()
    if not (os.path.exists(argv[1]) and os.path.exists(argv[2]) and os.path.exists(argv[3])):
        print("One of the specified files {}, {}, {} doesn't exist".format(argv[1], argv[2]),argv[3])
        exit()

    print("# Loading data from files {}".format(argv[3]))
    (X, y_test) = load_processed_data(argv[3], TEST_ROWS)

    print("### Loaded {} test rows".format(X.shape[0]))
    print("## X_test shape: ", X.shape)
    print("## Y_test shape: ", y_test.shape)

    # y_train = np.random.choice([0, 1], size=y_train.shape, p=[0.99, 0.01])

    # Modifying labels to time series prediction

    print("### Modifying labels")
    nonzero_test = np.count_nonzero(y_test)
    print("# Number of non-error labels: {}".format(y_test.shape[0] - nonzero_test))
    print("# Number of error labels: {}".format(nonzero_test))

    y_test = warp_labels(y_test, PREDICTION_LENGTH, WINDOW_SIZE)

    nonzero_test = np.count_nonzero(y_test)
    print("# Number of non-error labels: {}".format(y_test.shape[0] - nonzero_test))
    print("# Number of error labels: {}".format(nonzero_test))

    print("### Modified labels a to signal errors in the next {} samples.".format(PREDICTION_LENGTH))

    # Modifying x's to be 3D vectors

    X = make_timeseries_instances(X, WINDOW_SIZE)

    print("### Modified data to tensors with height {}".format(WINDOW_SIZE))

    # Something with adding the channel count

    X = np.expand_dims(X, axis=3)

    y_test = y_test[:X.shape[0]]


    print("### Loading the model from file {}".format(argv[1]))
    json_file = open(argv[1], 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    print("### Loading weights from file {}".format(argv[2]))
    model.load_weights(argv[2])
    print("### Loaded model from disk")

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print("### Evaluating the model")

    if len(argv) < 5:
        score = model.evaluate(X, y_test, verbose=1)
        print("#### Results ####")
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    else:
        while(True):
            line = sys.stdin.readline()
            print(line)
            i = int(line) - PREDICTION_LENGTH
            if 0 < i < X.shape[0]:
                prediction = model.predict(X[i,].reshape(1,WINDOW_SIZE,PCA_TARGET_SIZE,1))
                value = 0 if math.isnan(np.sum(prediction)) else np.sum(prediction)
                if value > 0:
                    print("Will fail in {}".format(PREDICTION_LENGTH))
                else:
                    print("Will not fail in {}".format(PREDICTION_LENGTH))

if __name__ == "__main__":
    main(sys.argv)

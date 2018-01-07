from __future__ import print_function
import os

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
        print("Correct arguments: <model> <weights> <data_file> |datapoint|")
        exit()
    if not (os.path.exists(argv[1]) and os.path.exists(argv[2]) and os.path.exists(argv[3])):
        print("One of the specified files {}, {}, {} doesn't exist".format(argv[1], argv[2]),argv[3])
        exit()

    print("# Loading data from files {}".format(argv[3]))
    (x_test, y_test) = load_processed_data(argv[3], TEST_ROWS)

    print("### Loaded {} test rows".format(x_test.shape[0]))
    print("## X_test shape: ", x_test.shape)
    print("## Y_test shape: ", y_test.shape)

    # y_train = np.random.choice([0, 1], size=y_train.shape, p=[0.99, 0.01])

    # Modifying labels to time series prediction

    print("### Modyfyinh labels")
    nonzero_test = np.count_nonzero(y_test)
    print("# Number of non-error labels: {}".format(y_test.shape[0] - nonzero_test))
    print("# Number of error labels: {}".format(nonzero_test))

    y_test = warp_labels(y_test, PREDICTION_LENGTH, WINDOW_SIZE)

    nonzero_test = np.count_nonzero(y_test)
    print("# Number of non-error labels: {}".format(y_test.shape[0] - nonzero_test))
    print("# Number of error labels: {}".format(nonzero_test))

    print("### Modified labels a to signal errors in the next {} samples.".format(PREDICTION_LENGTH))

    # Modifying x's to be 3D vectors

    x_test = make_timeseries_instances(x_test, WINDOW_SIZE)

    print("### Modified data to tensors with height {}".format(WINDOW_SIZE))

    # Something with adding the channel count

    x_test = np.expand_dims(x_test, axis=3)

    y_test = y_test[:x_test.shape[0]]


    print("### Loading the model from file {}".format(argv[1]))
    json_file = open(argv[1], 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    print("### Loading weights from file {}".format(argv[2]))
    model.load_weights(argv[2])
    print("### Loaded model from disk")

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print("### Evaluationg the model")

    score = model.evaluate(x_test, y_test, verbose=1)
    print("#### Results ####")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    main(sys.argv)

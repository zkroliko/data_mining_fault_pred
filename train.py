from __future__ import print_function

import datetime
import os

import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn import decomposition

from data_utils import load_raw_data, load_processed_data
from transform_input import make_timeseries_instances
from warp_labels import warp_labels

WINDOW_SIZE = 40
PREDICTION_LENGTH = 10

TRAIN_ROWS = 0
TEST_ROWS = 0

PCA_TARGET_SIZE = 10

PREPROCESSED = True

training_filename = 'training_data.csv'
test_filename = 'test_data.csv'

prep_training_filename = 'training_data_ps.csv'
prep_test_filename = 'test_data_ps.csv'

import sys

# Usually we will use pre-processed data, this is for a special case
if PREPROCESSED:
    # Normal turn of events
    print("# Loading prepared data from files {} and {}".format(prep_training_filename, prep_test_filename))
    (x_train, y_train), (x_test, y_test) = load_processed_data(prep_training_filename, TRAIN_ROWS), \
                                           load_processed_data(prep_test_filename, TEST_ROWS)
else:
    # Loading raw unprocessed data
    print("# Loading raw data from files {} and {}".format(training_filename, test_filename))

    (x_train, y_train), (x_test, y_test) = load_raw_data(training_filename, TRAIN_ROWS), \
                                           load_raw_data(test_filename, TEST_ROWS)
    # PCA dimensionality reduction
    pca = decomposition.PCA(n_components=PCA_TARGET_SIZE)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    pca.fit(x_test)
    x_test = pca.transform(x_test)
    print("# Reduced data to {} dimensions", PCA_TARGET_SIZE)

# Data is loaded, let's print some info

print("### Loaded {} training rows".format(x_train.shape[0]))
print("## X_train shape: ", x_train.shape)
print("## Y_train shape: ", y_train.shape)
print("### Loaded {} test rows".format(x_test.shape[0]))
print("## X_test shape: ", x_test.shape)
print("## Y_test shape: ", y_test.shape)

# y_train = np.random.choice([0, 1], size=y_train.shape, p=[0.99, 0.01])

# Modifying labels to time series prediction

print("### Train")
nonzero_train = np.count_nonzero(y_train)
print("# Number of non-error labels: {}".format(y_train.shape[0] - nonzero_train))
print("# Number of error labels: {}".format(nonzero_train))

y_train=warp_labels(y_train, PREDICTION_LENGTH, WINDOW_SIZE)

nonzero_train = np.count_nonzero(y_train)
print("# Number of =signal= non-error labels: {}".format(y_train.shape[0] - nonzero_train))
print("# Number of =signal= error labels: {}".format(nonzero_train))

print("### Test")
nonzero_test = np.count_nonzero(y_test)
print("# Number of non-error labels: {}".format(y_test.shape[0] - nonzero_test))
print("# Number of error labels: {}".format(nonzero_test))

y_test=warp_labels(y_test, PREDICTION_LENGTH, WINDOW_SIZE)

nonzero_test = np.count_nonzero(y_test)
print("# Number of non-error labels: {}".format(y_test.shape[0] - nonzero_test))
print("# Number of error labels: {}".format(nonzero_test))

print("### Modified labels a to signal errors in the next {} samples.".format(PREDICTION_LENGTH))

# Modifying x's to be 3D vectors

x_train, x_test = make_timeseries_instances(x_train, WINDOW_SIZE), make_timeseries_instances(x_test, WINDOW_SIZE)

print("### Modified data to tensors with height {}".format(WINDOW_SIZE))

# Something with adding the channel count

x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

# print("### Modified test to shape to: ", x_test.shape)

# "Input arrays should have the same number of samples as target arrays."
y_train = y_train[:x_train.shape[0]]
y_test = y_test[:x_test.shape[0]]

# We want to concentrate on faulty behaviour
w_zero, w_one = nonzero_train / y_train.shape[0], (y_train.shape[0] - nonzero_train) / y_train.shape[0]
print("### Weights of classes 0 and 1 are respectively: {}, {}".format(w_zero, w_one))
class_weights = {0: w_one, 1: w_zero}

# Training

batch_size = 128
epochs = 5

print("------ Starting ------")

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=[WINDOW_SIZE, PCA_TARGET_SIZE, 1]))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same', input_shape=[WINDOW_SIZE, PCA_TARGET_SIZE, 1]))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same', input_shape=[WINDOW_SIZE, PCA_TARGET_SIZE, 1]))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              loss_weights=[1.0],
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    class_weight=class_weights,
                    validation_data=(x_test, y_test))

# Saving the model
if not os.path.exists("models"):
    os.makedirs("models")
filename = "models/model" + str(datetime.datetime.now()).replace(":", "").replace(".","").replace(" ","")
model_json = model.to_json()
with open(filename, "w") as file:
    file.write(model_json)
model.save_weights(filename+".h5")

# Custom testing, won't believe these metrics
# moments = [202313, 520268, 628267, 760933, 761105, 761274, 767884, 767948, 768051, 778196, 781774, 790989, 791094,
#            913179, 1073703, 1132513, 1132676, 1140226, 1141794, 1237426, 1241905, 1387080, 1388043, 1570724, 1585962,
#            1586097]
# y = 0
# t = 0
# for m in moments:
#     i = m - PREDICTION_LENGTH
#     if 0 < i < x_train.shape[0]:
#         prediction = model.predict(x_train[i,].reshape(1,WINDOW_SIZE,PCA_TARGET_SIZE,1))
#         value = 0 if math.isnan(np.sum(prediction)) else np.sum(prediction)
#         if value > 0:
#             print("Seen {} from {} away".format(m,PREDICTION_LENGTH))
#         else:
#             print("Didn't see {} from {} away".format(m, PREDICTION_LENGTH))
#         y += value
#         t += 1
# print("Predicted {} of {}".format(y, t))

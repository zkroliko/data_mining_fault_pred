from __future__ import print_function

import datetime
import os

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import RMSprop
from sklearn import decomposition

from data_utils import load_raw_data, load_processed_data
from transform_input import make_timeseries_instances
from warp_labels import warp_labels

WINDOW_SIZE = 20
PREDICTION_LENGTH = 50

TRAIN_ROWS = 0
TEST_ROWS = 0

PCA_TARGET_SIZE = 10

PREPROCESSED = True

training_filename = 'training_data.csv'
test_filename = 'test_data.csv'

prep_training_filename = 'training_data_ps.csv'
prep_test_filename = 'test_data_ps.csv'

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

# Experiment
# x_train = np.zeros(shape=x_train.shape)
# y_train = np.zeros(shape=y_train.shape)
# x_train[5000] = np.ones(shape=x_train.shape[1])
# y_train[5000] = 1
#
# x_test = np.zeros(shape=x_test.shape)
# y_test = np.zeros(shape=y_test.shape)
# x_test[5000] = np.ones(shape=x_test.shape[1])
# y_test[5000] = 1

# y_train = np.random.choice([0, 1], size=y_train.shape, p=[0.99, 0.01])

# Modifying labels to time series prediction

print("### Train")
nonzero_train = np.count_nonzero(y_train)
print("# Number of non-error labels: {}".format(y_train.shape[0] - nonzero_train))
print("# Number of error labels: {}".format(nonzero_train))

warp_labels(y_train, PREDICTION_LENGTH)

nonzero_train = np.count_nonzero(y_train)
print("# Number of =signal= non-error labels: {}".format(y_train.shape[0] - nonzero_train))
print("# Number of =signal= error labels: {}".format(nonzero_train))

print("### Test")
nonzero_test = np.count_nonzero(y_test)
print("# Number of non-error labels: {}".format(y_test.shape[0] - nonzero_test))
print("# Number of error labels: {}".format(nonzero_test))

warp_labels(y_test, PREDICTION_LENGTH)

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
class_weights = {0: 0.0, 1: 1.0}

# Training

batch_size = 128
epochs = 1

print("------ Starting ------")

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=[WINDOW_SIZE, PCA_TARGET_SIZE, 1]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

def own_metric(y_true, y_pred):
    return keras.backend.mean(y_pred)

keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              loss_weights=[1.0],
              metrics=[keras.metrics.sparse_categorical_accuracy,own_metric])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    class_weight=class_weights,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
# score = model.evaluate(x_test, y_test,sample_weight=y_test[:,0], verbose=1)
# We have to scale the results, they are not represented well with respect to classes
print("#### Results ####")
proportion = y_test.shape[0] / (nonzero_test + 0.00001)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Saving the model
if not os.path.exists("models"):
    os.makedirs("models")
filename = "models/model" + str(datetime.datetime.now()).replace(":", "").replace(".","").replace(" ","")
model_json = model.to_json()
with open(filename, "w") as file:
    file.write(model_json)
model.save_weights(filename+".h5")

# reading: https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# Custom testing, won't believe these metrics
moments = [202313, 520268, 628267, 760933, 761105, 761274, 767884, 767948, 768051, 778196, 781774, 790989, 791094,
           913179, 1073703, 1132513, 1132676, 1140226, 1141794, 1237426, 1241905, 1387080, 1388043, 1570724, 1585962,
           1586097]
hup = 10
y = 0
for m in moments:
    prediction = model.predict(x_train[m - hup:, :, :], verbose=1)
    value = np.sum(prediction)
    print("Seen {}".format(m) if value > 0 else "Didn't see {}".format(m))
    y += value
print("Predicted {} of {}".format(y, len(moments)))

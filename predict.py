from __future__ import print_function

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import RMSprop
from sklearn import decomposition

from data_utils import load_raw_data, load_processed_data
from transform_input import make_timeseries_instances
from warp_labels import warp_labels

WINDOW_SIZE = 10
PREDICTION_LENGTH = 1

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
    print("Loading prepared data from files {} and {}".format(prep_training_filename, prep_test_filename))
    (x_train, y_train), (x_test, y_test) = load_processed_data(prep_training_filename), \
                                           load_processed_data(prep_test_filename)
else:
    # Loading raw unprocessed data
    print("Loading raw data from files {} and {}".format(training_filename, test_filename))

    (x_train, y_train), (x_test, y_test) = load_raw_data(training_filename, TRAIN_ROWS), \
                                           load_raw_data(test_filename, TEST_ROWS)
    # PCA dimensionality reduction
    pca = decomposition.PCA(n_components=PCA_TARGET_SIZE)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    pca.fit(x_test)
    x_test = pca.transform(x_test)
    print("Reduced data to {} dimensions", PCA_TARGET_SIZE)

# Data is loaded, let's print some info

print("### Loaded {} training rows".format(x_train.shape[0]))
print("## X_train shape: ", x_train.shape)
print("## Y_train shape: ", y_train.shape)
print("### Loaded {} test rows".format(x_test.shape[0]))
print("## X_test shape: ", x_test.shape)
print("## Y_test shape: ", y_test.shape)



# Modifying labels to time series prediction

warp_labels(y_train,PREDICTION_LENGTH)
warp_labels(y_test,PREDICTION_LENGTH)


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

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# We want to concentrate on faulty behaviour
class_weights = {0: 0.0, 1: 1.0}

# Training

batch_size = 128
epochs = 10

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

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              loss_weights=[1.0],
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    class_weight=class_weights,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
# score = model.evaluate(x_test, y_test,sample_weight=y_test[:,0], verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

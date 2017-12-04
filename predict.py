from __future__ import print_function

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from load import load_data
from warp_labels import warp_labels

WINDOW_SIZE = 20
LAST_COLUMN = 43

TRAIN_ROWS = 0
TEST_ROWS = 0

training_filename = 'training_data.csv'
test_filename = 'test_data.csv'

(x_train, y_train), (x_test, y_test) = load_data(training_filename, TRAIN_ROWS), load_data(test_filename, TEST_ROWS)
print("Loaded {} training rows".format(x_train.shape[0]))
print(x_train.shape)
print(y_train.shape)
print("Loaded {} test rows".format(x_test.shape[0]))
print(x_test.shape)
print(y_test.shape)

# Training

batch_size = 128
epochs = 10

# Modyfying labels to time series prediction
warp_labels(y_train)
warp_labels(y_test)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# We want to concentrate on faulty behaviour
class_weights = {0: 0.0, 1: 1.0}

print("------ Starting ------")

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(41,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
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

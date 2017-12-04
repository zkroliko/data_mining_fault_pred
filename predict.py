from __future__ import print_function

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from load import load_data

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
num_classes = 2
epochs = 1

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Rolling and setting so we'll we have prediction
# Last element :)
y_train = np.roll(y_train, -1)
y_test = np.roll(y_test, -1)
# x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# We want to concentrate on faulty behaviour
# class_weights = numpy.asarray([0.0, 1.0])
class_weights = {0: 0.0, 1: 1000000.0}

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(37,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # sample_weight=y_train[:, 1],
                    class_weight=class_weights,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
# score = model.evaluate(x_test, y_test,sample_weight=y_test[:,0], verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

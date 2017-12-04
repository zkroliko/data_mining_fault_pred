from load import load_data
import numpy as np

TRAIN_ROWS = 0
TEST_ROWS = 0

training_filename = 'training_data.csv'
test_filename = 'test_data.csv'

NORMALIZED = True

(x_train, y_train), (x_test, y_test) = load_data(training_filename, TRAIN_ROWS, normalized=NORMALIZED),\
                                       load_data(test_filename, TEST_ROWS, normalized=NORMALIZED)

data = np.vstack((x_train,x_test))
print("Minimums:")
print(data.min(axis=0))
print("Maximums:")
print(data.max(axis=0))
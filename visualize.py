import os

import math
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib

from data_utils import load_raw_data, load_processed_data

PREPROCESSED = True

observation_points = [202313, 520268, 628267, 760933, 761105, 761274, 767884,
                      767948, 768051, 778196, 781774, 790989, 791094, 913179,
                      1073703, 1132513, 1132676, 1140226, 1141794, 1237426,
                      1241905, 1387080, 1388043, 1570724, 1585962, 1586097] # all errors in training

START = 761200
WINDOW = 200

r_filename = "training_data.csv"
p_filename = "training_data_ps.csv"

if not PREPROCESSED:
    array, labels = load_raw_data(r_filename, START + WINDOW)

else:
    array, labels = load_processed_data(p_filename, START + WINDOW)

array2 = np.zeros(shape=array.shape)

array2[1:, :] = np.copy(array[:-1, :])

array3 = array[START:, :]

if not os.path.exists("images"):
    os.makedirs("images")

min = array3.min(axis=0)
max = array3.max(axis=0)
array4 = (array3 - min) / (max-min)

scipy.misc.toimage(array4, cmin=-1.0, cmax=1.0).save('images/data1.jpg')
image_data = scipy.misc.imread('images/data1.jpg').astype(np.float32)
matplotlib.rcParams.update({'font.size': 8})
plt.figure(figsize=(3, WINDOW/4))
plt.imshow(image_data)
plt.yticks(np.arange(0, WINDOW + 1.0, 1.0))
plt.xticks(np.arange(0, 10, 1.0))
plt.savefig("images/figure1.png")


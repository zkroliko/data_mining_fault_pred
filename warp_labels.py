import numpy as np

# Error prediction should be made in advance - some fixed time interval
# this time interval (for now) will be intepreted simply as number of samples

INTERVAL = 30


def warp_labels(labels, interval=INTERVAL):
    for i in range(labels.shape[0]):
        labels[i] = np.amax(labels[i:min((i + interval), labels.shape[0])])
    return labels

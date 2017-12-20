import numpy as np

# Error prediction should be made in advance - some fixed time interval
# this time interval (for now) will be intepreted simply as number of samples

INTERVAL = 30
SHIFT = 40

def warp_labels(labels, interval=INTERVAL, shift=SHIFT):
    labels = np.hstack([labels,np.zeros(shift)])
    for i in range(labels.shape[0]-shift-interval):
        labels[i] = np.amax(labels[(i+shift):min((i + interval + shift), labels.shape[0])])
    return labels

import numpy as np

# Error prediction should be made in advance - some fixed time interval
# this time interval (for now) will be intepreted simply as number of samples

INTERVAL = 30
SHIFT = 40

def warp_labels(labels, interval=INTERVAL, shift=SHIFT):
    labels = np.hstack([labels,np.zeros(shift)])
    for i in range(labels.shape[0]-shift-interval):
        if labels[i] > 0:
            labels[max(0,i-interval):i]=labels[i]
    return labels

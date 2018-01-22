# Fault prediction based on timeseries of device's sensor logs

Using Keras/Tensorflow implementation of CNN's for logistic classification with very (1:1000) unbalanced classes - rebalancing was done with class weights.

# Source files #

visualize.py - is used for displaying a choosen region of data as a color image; conifugrable in-file

data_utils.py - for loading and normalization (if needed) of data

analyze.py - originally used for normalization purposes

warp_labels.py - modifies labels as to signal errors in some immediate timeframe; function definition

transform_input.py - for transformation of timeseries into tensor representation required by CNN's; function definition

process_data.py - loads normalized data, performs PCA, and then saves it into new files; configurable in-file

train.py - contains model specification; performs all necessary data preparation (using other files) and trains the model, which is later saved; configuralbe in-file

evaluate.py - used for testing model, can be interactive; usage: "<model> <data_file> |-i|" where -i stands for interactive mode

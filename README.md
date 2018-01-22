# Fault prediction based on timeseries of device's sensor logs

Using Keras/Tensorflow implementation of CNN's with dropout and BN for logistic classification with very (1:1000) unbalanced classes - rebalancing was done with class weights.

## Data ##

Timeseries of sensor samples labeled as `1` when faulty behaviour occured. Data itself, split into training and testing volumens, is not publicly available.

Originally:

```
Timestamp - 41 numeric values - 0/1 label
```

After removing the timestamp and PCA:

```
10 numeric values - 0/1 label
```

## Source files ##

[train.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/train.py) - contains model specification; performs all necessary data preparation (using other files) and trains the model, which is later saved; configuralbe in-file

[evaluate.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/evaluate.py) - used for testing model, can be interactive; usage: "<model> <data_file> |-i|" where -i stands for interactive mode

[data_utils.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/data_utils.py) - for loading and normalization (if needed) of data

[warp_labels.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/warp_labels.py) - modifies labels as to signal errors in some immediate timeframe; function definition

[transform_input.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/transform_input.py) - for transformation of timeseries into tensor representation required by CNN's; function definition

[process_data.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/process_data.py) - loads normalized data, performs PCA, and then saves it into new files; configurable in-file

[visualize.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/visualize.py) - is used for displaying a choosen region of data as a color image; conifugrable in-file

[analyze.py](https://github.com/zkroliko/data_mining_fault_pred/blob/master/analyze.py) - originally used for obtaining parameters for normalization

## Results ##

Achievied `99.983%` accurency on test data, and over `85%` of faults (`l` labels) detected within margin of 10-50 time units.

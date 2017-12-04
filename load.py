import numpy as np
import pandas as pd

# Not used right now
def date_parser(date):
    return pd.datetime.strptime(date, ' %Y-%m-%d  %H:%M:%S ')


NORMALIZATION_MIN = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.99999982e-02,
                              0.00000000e+00, 0.00000000e+00, 4.70260013e-04, - 1.71464996e+02,
                              0.00000000e+00, 2.44144991e-04, 3.81475990e-03, - 5.14281016e+04,
                              3.25895002e-04, 3.25895002e-04, 3.01764987e-04, 3.01764987e-04,
                              3.25895002e-04, 3.25895002e-04, 3.37959995e-04, 3.37959995e-04,
                              3.01764987e-04, 3.01764987e-04, - 7.62963027e-04, 1.52747002e+01,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 5.11189000e+06, 0.00000000e+00,
                              0.00000000e+00])
NORMALIZATION_MAX = np.array([2.99643002e+01, 8.12132034e+01, 1.18266998e+02, 2.40000000e+04,
                              5.00000000e+04, 1.20000000e+04, 8.79486008e+01, 1.60000002e+00,
                              1.52935004e+00, 9.99535024e-01, 3.69810009e+00, 9.88564014e-01,
                              6.78625011e+00, 1.60000000e+01, 2.21337997e+02, 3.85669006e+02,
                              7.40024033e+01, 7.48602982e+01, 6.69985962e+01, 6.69985962e+01,
                              7.20035019e+01, 7.20035019e+01, 6.57574997e+01, 6.49996033e+01,
                              6.49996033e+01, 6.49996033e+01, 1.18927002e+02, 3.53824005e+02,
                              3.49994995e+02, 4.91786003e+01, 1.85016000e+00, 4.61799002e+00,
                              2.71313000e+01, 2.85471992e+01, 1.03747002e+02, 1.00042999e+02,
                              5.39449997e+01, 6.03083992e+01, 5.11189000e+06, 2.46617004e-01,
                              4.71614990e+03])
NORMALIZATION_INTERVAL = np.maximum(NORMALIZATION_MAX - NORMALIZATION_MIN,1.0e-06).astype(np.float32)

LAST_COLUMN = 41


def load_data(filename, n_rows=0, normalized=True):
    if n_rows is 0:
        raw = pd.read_csv(filename, header=None,)
    else:
        raw = pd.read_csv(filename, header=None, nrows=n_rows)
    data = (np.array(raw.values)[:, 1:]).astype(np.float32)
    # print(NORMALIZATION_MIN)
    # print(NORMALIZATION_MAX)
    # print(NORMALIZATION_INTERVAL)
    # Returning "x" and "y"
    if normalized:
        x = ((data[:, :LAST_COLUMN] - NORMALIZATION_MIN) / NORMALIZATION_INTERVAL)
        y = data[:, LAST_COLUMN]
    else:
        x, y = data[:, :LAST_COLUMN], data[:, LAST_COLUMN]
    return x, y

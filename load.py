import dateutil
import numpy as np
import pandas as pd


def date_parser(date):
    return pd.datetime.strptime(date, ' %Y-%m-%d  %H:%M:%S ')


print(date_parser(" 2011-02-01  00:13:00 "))

NORMALIZATION_MIN = np.array([0.00000000e+00, 2.44154000e+04, 5.14322000e+02, 7.06552000e+01,
                              6.72474000e-01, 6.49680000e-01, 7.37878000e-01, 3.00433000e+00,
                              -1.71465000e+02, 5.60581000e-03, 8.33487000e+00, 1.75940000e+02,
                              -5.14280000e+04, 4.50027000e+01, 4.50027000e+01, 4.50027000e+01,
                              4.50027000e+01, 4.60021000e+01, 4.60021000e+01, 4.60021000e+01,
                              4.60021000e+01, 4.56773000e+01, 4.54941000e+01, 5.69993000e+01,
                              1.73456000e+02, 2.20795000e+02, 7.11214000e-01, 1.93790000e-01,
                              1.93790000e-01, 7.77714000e-01, 8.93683000e-01, 8.38808000e-01,
                              8.94084000e-01, 8.30615000e-01, 7.85236000e-01, 5.11189000e+06,
                              9.49580000e-02])
NORMALIZATION_MAX = np.array([2.25529000e+04, 4.15584000e+04, 2.54226000e+03, 7.48804000e+01,
                              8.34477000e-01, 8.45958000e-01, 9.34264000e-01, 3.55705000e+00,
                              -1.65310000e+02, 5.33896000e+00, 9.86639000e+00, 1.81657000e+02,
                              -4.95815000e+04, 7.40024000e+01, 7.40024000e+01, 6.69986000e+01,
                              6.69986000e+01, 7.20035000e+01, 7.20035000e+01, 6.49996000e+01,
                              6.49996000e+01, 6.43583000e+01, 6.41751000e+01, 7.40000000e+01,
                              1.82321000e+02, 2.67673000e+02, 1.64733000e+01, 1.68674000e+00,
                              4.61799000e+00, 1.13570000e+01, 9.12259000e+00, 4.03330000e+01,
                              3.76688000e+01, 1.26443000e+01, 1.53831000e+01, 5.11189000e+06,
                              1.32367000e-01])
NORMALIZATION_INTERVAL = np.maximum(1.0e-06, NORMALIZATION_MAX - NORMALIZATION_MIN).astype(np.float32)

LAST_COLUMN = 41


def load_data(filename, n_rows=0, normalized=True):
    if n_rows is 0:
        raw = pd.read_csv(filename, header=None,
                          parse_dates=[0], date_parser=date_parser)
    else:
        raw = pd.read_csv(filename, header=None,
                          parse_dates=[0], nrows=n_rows, date_parser=date_parser)
    data = (np.array(raw.values)[:, 1:]).astype(np.float32)
    # print(NORMALIZATION_MIN)
    # print(NORMALIZATION_MAX)
    # print(NORMALIZATION_INTERVAL)
    # Returning "x" and "y"
    if normalized:
        x = ((data[:, :LAST_COLUMN] - NORMALIZATION_MIN) / NORMALIZATION_INTERVAL).astype(np.float32)
        y = data[:, LAST_COLUMN]
    else:
        x, y = data[:, :LAST_COLUMN], data[:, LAST_COLUMN]
    return x, y

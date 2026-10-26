from scipy.io import arff
import numpy as np
import pandas as pd
from multipledispatch import dispatch
from util import ARFFStream

np.random.seed(101)


def read_data(pth):
    split = pth.split('.')
    root = pth
    if split[-1] == 'arff':
        data = arff.loadarff(root)
        data = pd.DataFrame(data[0])
        data = data.dropna()
        if cat_cols := [col for col in data.columns if data[col].dtype == "O"]:
            data[cat_cols] = data[cat_cols].apply(lambda x: x.str.decode('utf8'))
            # find unique values of each categorical column
            uniq_vals = [data[col].unique() for col in cat_cols]
            # assign a number to each unique value of each categorical column
            for i in range(len(cat_cols)):
                data[cat_cols[i]] = data[cat_cols[i]].apply(lambda x: np.where(uniq_vals[i] == x)[0][0])

        # change the d_type of the last column of the data frame to int
        data.iloc[:, -1] = data.iloc[:, -1].astype(int)
    else:
        print("please provide .arff type")
        return -1
    return data


@dispatch(np.ndarray)
def prepare_data(input_data):
    data_features = dict(enumerate(input_data[:-1].flatten(), 1))
    data_labels = int(input_data[-1])
    return data_features, data_labels


@dispatch(list, np.int64)
def prepare_data(input_data, lbl):
    data_features = dict(enumerate(input_data, 1))
    data_labels = lbl
    return data_features, data_labels


def prepare_data_with_features(input_data, lbl, features):
    data_features = {}
    for i, f in enumerate(features):
        data_features[f] = input_data[i]
    data_labels = lbl
    return data_features, data_labels


def convert_to_array(data_dicts):
    """
    Converts a list of dictionaries into a NumPy array.

    Args:
        data_dicts (list): A list of dictionaries with features as keys.

    Returns:
        numpy.ndarray: A 2D NumPy array where each row is a data point.
    """
    # Assuming all dictionaries have the same keys in the same order
    if not data_dicts:
        return np.array([])  # Return an empty array if no data

    n_features = len(data_dicts[0])
    data_matrix = np.zeros((len(data_dicts), n_features))

    for i, data_dict in enumerate(data_dicts):
        # Ensure the features are read in a consistent order
        for j, key in enumerate(data_dict.keys()):
            data_matrix[i, j] = data_dict[key]

    return data_matrix


def read_data_2(pth):
    stream = ARFFStream("{}".format(pth))
    return stream

import numpy as np
from util import ARFFStream

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


def read_data(pth):
    stream = ARFFStream("{}".format(pth))
    return stream

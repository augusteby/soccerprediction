import pandas as pd

from constants import LABEL

def load_X_y(data_filepath):
    """Method that retrieve the data table from its filepath and returns X (the features) and
    y (the labels)

    :param data_filepath:
    :return:
    """
    data = pd.read_csv(data_filepath)
    y = data[LABEL].values

    data = data.drop(LABEL, 1)

    features = data.columns
    X = data.values

    return X, y, features
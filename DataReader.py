import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE

    x_train, y_train = [], []
    x_test, y_test = [], []

    for data_batch in ['1', '2', '3', '4', '5']:
        with open(os.path.join(data_dir, 'data_batch_{}'.format(data_batch)), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        x_train.append(data_dict[b'data'])
        y_train.extend(data_dict[b'labels'])

    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)

    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    x_test = data_dict[b'data']
    y_test = data_dict[b'labels']
    y_test = np.array(y_test)


    ### END CODE HERE

    return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid


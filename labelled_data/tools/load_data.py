import numpy as np
import torch
import random
from .constants import *


def split_datasets(data, labels):
    # 80 % for training, 10 % for validation, 10 % for testing
    train_index = int(len(data) * 0.8)
    valid_index = train_index + int(len(data) * 0.1)

    X_train = data[0:train_index]
    X_test = data[train_index:valid_index]
    X_valid = data[valid_index:]

    y_train = labels[0:train_index]
    y_test = labels[train_index:valid_index]
    y_valid = labels[valid_index:]

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def format_dataset(X, y):
    X = np.reshape(X, (X.size, 1)).astype(np.float64)
    y = np.reshape(y, (y.size, 1)).astype(np.int16)
    X = torch.tensor(X, dtype=torch.float)

    y_final = torch.tensor(y, dtype=torch.float)

    # shape is (batch size, sequence length, input size)
    X_final = torch.reshape(X, (X.shape[0], 1, X.shape[1]))

    return X_final, y_final


def get_chunk_files():
    insect_files = os.listdir(os.path.join(RAWDATA_LABELLED_PATH, 'insects'))
    insect_files = [os.path.join(os.path.join(RAWDATA_LABELLED_PATH, 'insects'), file) for file in insect_files]

    noise_files = os.listdir(os.path.join(RAWDATA_LABELLED_PATH, 'noise'))
    noise_files = [os.path.join(os.path.join(RAWDATA_LABELLED_PATH, 'noise'), file) for file in noise_files]
    data_files, labels_files = split_files(noise_files + insect_files)

    return data_files, labels_files


def get_raw_files():
    raw_files = os.listdir(RAWDATA_LABELLED_PATH)
    raw_files = [os.path.join(RAWDATA_LABELLED_PATH, file) for file in raw_files]
    data_files, labels_files = split_files(raw_files)
    return data_files, labels_files


def split_files(files):
    random.shuffle(files)
    data_files = []
    labels_files = []
    for file in files:
        if file.split('_')[-1] == 'data.npy':
            data_files.append(file)
            labels_files.append(file[0:-8] + 'labels.npy')

    return data_files, labels_files


def data_loader(no_files=100):
    data_files, labels_files = get_raw_files()
    if len(data_files) > no_files:
        data_files = data_files[0:no_files]
        labels_files = labels_files[0:no_files]

    X_train, X_test, X_valid, y_train, y_test, y_valid = split_datasets(data_files, labels_files)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def data_generator(data, labels, sequence_length=1):
    for d in range(0, len(data)):
        data_channel = np.load(data[d])
        label_channel = np.load(labels[d])
        data_format, label_format = format_dataset(data_channel, label_channel)

        if sequence_length > 1:
            left_over = data_format.shape[0] % sequence_length
            no_batches = int((data_format.shape[0] - left_over) / sequence_length)
            data_length = data_format.shape[0]
            data_format = torch.reshape(data_format[:data_length - left_over, :, :], (no_batches, sequence_length, 1))
            label_format = torch.reshape(label_format[:data_length - left_over, :], (no_batches * sequence_length, 1))

        yield data_format*1000, label_format

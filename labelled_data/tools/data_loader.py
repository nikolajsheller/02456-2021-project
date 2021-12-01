import numpy as np
from torch.autograd import Variable
import torch
import random
import os
from .constants import *


def split_datasets(data, labels):
    # 80 % for training, 10 % for validation, 10 % for testing
    train_index = int(len(data)*0.8)
    valid_index = train_index + int(len(data)*0.1)

    X_train = data[0:train_index]
    X_test = data[train_index:valid_index]
    X_valid = data[valid_index:]

    y_train = labels[0:train_index]
    y_test = labels[train_index:valid_index]
    y_valid = labels[valid_index:]

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def format_datasets(X_train, X_test, X_valid, y_train, y_test, y_valid):
    X_train = np.reshape(X_train,(X_train.size,1)).astype(np.int32)
    X_test = np.reshape(X_test,(X_test.size,1)).astype(np.int32)
    X_valid = np.reshape(X_valid,(X_valid.size,1)).astype(np.int32)
    y_train = np.reshape(y_train,(y_train.size,1)).astype(np.int16)
    y_test = np.reshape(y_test,(y_test.size,1)).astype(np.int16)
    y_valid = np.reshape(y_valid,(y_valid.size,1)).astype(np.int16)

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_valid_tensors = Variable(torch.Tensor(X_valid))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_valid_tensors = Variable(torch.Tensor(y_valid))
    y_test_tensors = Variable(torch.Tensor(y_test))

    # shape is (batch size, sequence length, input size)
    X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_valid_tensors_final = torch.reshape(X_valid_tensors,   (X_valid_tensors.shape[0], 1, X_valid_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    return X_train_tensors_final, X_test_tensors_final, X_valid_tensors_final, y_train_tensors, y_test_tensors, y_valid_tensors


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


def data_loader(chunks=True):
    if chunks:
        data_files, labels_files = get_chunk_files()
    else:
        data_files, labels_files = get_raw_files()

    for f in range(0, len(data_files)):
        data = np.load(data_files[f])[:,1]
        labels = np.load(labels_files[f])[:,1]

        X_train, X_test, X_valid, y_train, y_test, y_valid = split_datasets(data, labels)
        X_train_tensors_final, X_test_tensors_final, X_valid_tensors_final, y_train_tensors, y_test_tensors, y_valid_tensors = format_datasets(X_train, X_test, X_valid, y_train, y_test, y_valid)
        dict = {
            'X_train': X_train_tensors_final,
            'X_test': X_test_tensors_final,
            'X_valid': X_valid_tensors_final,
            'y_train': y_train_tensors,
            'y_test': y_test_tensors,
            'y_valid': y_valid_tensors
        }
        yield dict

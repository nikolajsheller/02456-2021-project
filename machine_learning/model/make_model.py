import torch
import torch.nn as nn
from torch.nn import Sigmoid
from torch.autograd import Variable
from labelled_data.tools.load_data import data_loader
from labelled_data.tools.load_data import data_generator
from machine_learning.model.early_stopping import EarlyStopping
import wandb

use_cuda = False
chunks = False
early_stopping = EarlyStopping(patience=5, verbose=True)
class Config(object):
    """
    Turns a dictionary into a class
    """
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def make_model(config):
    lstm = LSTM(config.num_classes, config.input_size, config.lstm_hidden_size, config.linear_hidden_size, \
                config.num_layers, config.lstm_dropbout, config.linear_dropout)  # our lstm class
    lstm = get_variable(lstm)

    criterion = torch.nn.BCELoss()  # cross validation
    optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
    return lstm, criterion, optimizer


def evaluate_result(output):
    output[output > 0.5] = 1.
    output[output < 0.5] = 0.
    return output


def train_model(lstm, criterion, optimizer, config, verbose=0):
    epoch_training_loss = []
    epoch_validation_loss = []
    epoch_training_acc = []
    epoch_validation_acc = []

    X_train, X_test, X_valid, y_train, y_test, y_valid = data_loader(chunks=chunks, no_files=config.no_files)
    print('training files', len(X_train))
    print('validation files', len(X_valid))
    print('test files', len(X_test))

    for epoch in range(config.epochs):
        training_loss = 0.
        validation_loss = 0.
        training_correct = 0.
        training_all = 0.
        validation_correct = 0.
        validation_all = 0.
        train_total_batches = 0
        valid_total_batches = 0

        train_gen = data_generator(X_train, y_train, sequence_length=config.sequence_length)
        #test_gen = data_generator(X_test, y_test, sequence_length=config.sequence_length)
        valid_gen = data_generator(X_valid, y_valid, sequence_length=config.sequence_length)

        valid_batches = 0
        for _ in range(len(X_valid)):
            i_batch = 0

            data, labels = valid_gen.__next__()
            data = get_variable(data * 1000)
            labels = get_variable(labels)

            while data.shape[0] > i_batch + config.no_batches:
                valid_batches += 1
                batch_data = Variable(data[i_batch:(i_batch + config.no_batches)])
                batch_labels = Variable(labels[(i_batch * config.sequence_length):(i_batch + config.no_batches) * config.sequence_length])

                lstm.eval()
                outputs = lstm.forward(batch_data)  # forward pass
                loss = criterion(outputs, batch_labels)
                validation_loss += loss.item()

                validation_correct += (evaluate_result(outputs) == batch_labels).float().sum()
                validation_all += len(batch_labels)
                i_batch += config.no_batches
                valid_total_batches += 1

        train_batches = 0

        for _ in range(len(X_train)):
            i_batch = 0

            data, labels = train_gen.__next__()
            data = get_variable(data * 1000)
            labels = get_variable(labels)

            if epoch == 0 and verbose == 1:
                print('Number of batch loops', data.shape[0] / config.no_batches)

            while data.shape[0] > i_batch + config.no_batches:
                train_batches += 1
                batch_data = data[i_batch:(i_batch + config.no_batches)]
                batch_labels = labels[(i_batch * config.sequence_length):(i_batch + config.no_batches) * config.sequence_length]

                lstm.train()

                train_outputs = lstm.forward(batch_data)  # forward pass
                optimizer.zero_grad()  # calculate the gradient, manually setting to 0

                # obtain the loss function
                loss = criterion(train_outputs, batch_labels)
                loss.backward(retain_graph=True)  # calculates the loss of the loss function#retain_graph=True

                training_loss += loss.data.item()

                training_correct += (evaluate_result(train_outputs) == batch_labels).float().sum()
                training_all += len(batch_labels)

                optimizer.step()  # improve from loss, i.e backprop

                i_batch += config.no_batches
                train_total_batches += 1
        if validation_all > 0:
            validation_acc = validation_correct / float(validation_all)
            validation_loss = validation_loss / valid_total_batches
        else:
            validation_acc = float('NaN')
            validation_loss = float('NaN')
        if training_all > 0:
            training_acc = training_correct / float(training_all)
            training_loss = training_loss / train_total_batches
        else:
            training_acc = float('NaN')
            training_loss = float('NaN')
        # if epoch % 10 == 0:
        if verbose == 1:
            print(
                "Epoch: %d, training loss: %1.5f, validation loss: %1.5f, training acc: %1.5f, , validation acc: %1.5f" % (
                    epoch, training_loss / train_total_batches, validation_loss / valid_total_batches, training_acc, validation_acc))

        epoch_validation_loss.append(validation_loss)
        epoch_validation_acc.append(validation_acc)
        epoch_training_loss.append(training_loss)
        epoch_training_acc.append(training_acc)
        wandb.log({
            'epoch': epoch,
            'validation_loss': validation_loss,
            'validation_acc': validation_acc,
            'training_loss': training_loss,
            'training_acc': training_acc,
        })
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(validation_loss, lstm)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            return X_test, y_test
    return X_test, y_test

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, lstm_hidden_size, linear_hidden_size, num_layers, lstm_dropbout,
                 linear_dropout):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.lstm_hidden_size = lstm_hidden_size  # hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=lstm_dropbout)  # lstm
        self.linear_1 = nn.Linear(lstm_hidden_size, linear_hidden_size)  # fully connected last layer
        self.dropout = nn.Dropout(linear_dropout)
        self.linear_out = nn.Linear(int(linear_hidden_size), num_classes)  # fully connected last layer
        self.sigmoid = Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Propagate input through LSTM
        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size, device=x.device))  # hidden state
        #c_0 = Variable(
        #    torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size, device=x.device))  # internal state

        x, (h, c) = self.lstm(x)#, (h_0, c_0))  # lstm with input, hidden, and internal state

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])  # reshaping the data for Dense layer next

        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_out(x)

        out = self.sigmoid(x)
        return out

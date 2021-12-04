#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
#use_cuda = False
print("Running GPU.") if use_cuda else print("No GPU available.")

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

# Load data
EVENTS_CACHE_PATH = os.path.expanduser("~/EventCache")
filename_data = os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData/dca6327d8fa8_20210330_raw_20210330T152242Z_ds_5_data.npy')
filename_labels = os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData/dca6327d8fa8_20210330_raw_20210330T152242Z_ds_5_labels.npy')

data = np.load(filename_data)
labels = np.load(filename_labels)

# create artificial training data
#channel = 4
#event1 = data[582400:583200,channel]
#event2 = data[909000:910500,channel]
#event3 = data[1801000:1802000,channel]
#noise = data[979000:990000,channel]
#data_chunk = np.concatenate((noise, event1, noise, event2, noise, event3, noise, event2, noise))
#labels_chunk = np.zeros(data_chunk.size)
#labels_chunk[11300:11600] = 1
#labels_chunk[23500:23800] = 1
#labels_chunk[35600:36000] = 1
#labels_chunk[47800:48400] = 1

import numpy as np
from scipy.signal import butter,filtfilt
# Filter requirements.
T = 5.0         # Sample Period
fs = 300.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

channel = 4
event1 = data[582400:583200,channel]
event2 = data[909000:910500,channel]
event3 = data[1801000:1802000,channel]
#event_fake = np.concatenate((np.linspace(1444, 5000, num=300).astype(int), np.linspace(5000, 1444, num=300).astype(int)))
event_fake = butter_lowpass_filter(event3, cutoff, fs, order)
event_fake2 = event_fake*0.5+705
noise = data[979000:979500,channel]
#data_chunk = np.concatenate((noise, event1, noise, event2, noise, event_fake, noise, event3, noise, event2, noise))
#data_chunk = np.concatenate((noise, event1, noise, event_fake, event2, noise, event_fake, noise, event3, noise, event2, noise))
data_chunk = np.concatenate((event1, event_fake, event2, event_fake2, event3, event2))

levent1 = np.zeros(event1.size)
levent1[300:500] = 1
levent2 = np.zeros(event2.size)
levent2[700:1000] = 1
levent3 = np.zeros(event3.size)
levent3[350:650] = 1
levent_fake = np.zeros(event_fake.size)
levent_fake[:] = 0
lnoise = np.zeros(noise.size)
lnoise[:] = 0
#labels_chunk = np.concatenate((lnoise, levent1, lnoise, levent_fake, levent2, lnoise, levent_fake, lnoise, levent3, lnoise, levent2, lnoise))
labels_chunk = np.concatenate((levent1, levent_fake, levent2, levent_fake, levent3, levent2))

# plot
fig, ax1 = plt.subplots()
plt.title("Training data")
color = 'tab:blue'
ax1.set_ylabel('data', color=color)
ax1.plot(data_chunk, color=color)
ax1.tick_params(axis='y', color=color)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('labelled', color=color)
ax2.plot(labels_chunk, color=color)
ax2.tick_params(axis='y', color=color)
plt.show()

# Transform and scale output
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler()

x = np.reshape(data_chunk, (-1,1))
y = np.reshape(labels_chunk, (-1,1))

X_ss = ss.fit_transform(x)
y_mm = mm.fit_transform(y)
#X_ss = x.to_numpy()
#y_mm = y.to_numpy()

print("Type", type(X_ss))
print("Training Shape", X_ss.shape, y_mm.shape)

# plot training data
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_ylabel('sensor', color=color)
ax1.plot(x, color=color)
ax1.tick_params(axis='y', color=color)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('insect', color=color)
ax2.plot(y, color=color)
ax2.tick_params(axis='y', color=color)
plt.show()

# Define training and test data
#first 200 for training
n_train = 5500

X_train = X_ss[:n_train, :]
X_test = X_ss[n_train:, :]

y_train = y_mm[:n_train, :]
y_test = y_mm[n_train:, :] 

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape) 

# Convert numpy arrays to tensors and variables
X_train_tensors = get_variable(Variable(torch.Tensor(X_train)))
X_test_tensors = get_variable(Variable(torch.Tensor(X_test)))

y_train_tensors = get_variable(Variable(torch.Tensor(y_train)))
y_test_tensors = get_variable(Variable(torch.Tensor(y_test)))

# Prepare input for LSTM

#reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

# Define model
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, n_linear=128):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.n_linear = n_linear

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bidirectional=False) #lstm
        self.lstm_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bidirectional=False) #lstm
        self.fc_1 =  nn.Linear(hidden_size, n_linear) #fully connected 1
        self.fc_2 =  nn.Linear(n_linear, n_linear) #fully connected 2
        self.fc = nn.Linear(n_linear, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = get_variable(Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))) #hidden state
        c_0 = get_variable(Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc_2(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

# Hyper parameters
num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr
input_size = 1 #number of features
hidden_size = 400 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes

#  Instantiate the class LSTM1 object
#lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class 
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1], 128).cuda()
print(lstm1)

# Loss function and optimizer
def squared_loss(t: Tensor, y: Tensor) -> Tensor:
    """ Compute squared loss using tensorflow
    t : Tensor
        output from network
    y : Tensor
        labeled data
    """
    diff: Tensor = t - y
    diff_pow2: Tensor = torch.pow(diff, 2)
    t_loss: Tensor = torch.sum(diff_pow2)
    return t_loss

criterion = torch.nn.MSELoss()    # mean-squared error for regression
#criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)
  #loss = squared_loss(outputs, y_train_tensors)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 10 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Run the model

# Convert data
#df_X_ss = ss.transform(df.iloc[:, :-1]) #old transformers
#df_y_mm = mm.transform(df.iloc[:, -1:]) #old transformers
df_X_ss = X_ss
df_y_mm = y_mm

df_X_ss = get_variable(Variable(torch.Tensor(df_X_ss))) #converting to Tensors
df_y_mm = get_variable(Variable(torch.Tensor(df_y_mm)))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

# Show predictions
train_predict = lstm1(df_X_ss)#forward pass
data_predict = get_numpy(train_predict) #numpy conversion
dataY_plot = get_numpy(df_y_mm)

#data_predict = mm.inverse_transform(data_predict) #reverse transformation
#dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=n_train, c='r', linestyle='--') #size of the training set
plt.plot(dataY_plot, label='Actual Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show()

#data_predict = mm.inverse_transform(data_predict) #reverse transformation
#dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=n_train, c='r', linestyle='--') #size of the training set
plt.plot(X_ss, label='Sensor') #actual plot
plt.plot(dataY_plot, label='Actual Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show()

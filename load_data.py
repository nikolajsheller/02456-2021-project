# %% [markdown]
# # Import
import os
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Load data

EVENTS_CACHE_PATH = os.path.expanduser("~/EventCache")
filename_data = os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData/dca6327d8fa8_20210330_raw_20210330T152242Z_ds_5_data.npy')
filename_labels = os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData/dca6327d8fa8_20210330_raw_20210330T152242Z_ds_5_labels.npy')

data = np.load(filename_data)
labels = np.load(filename_labels)

#channel = 4
#data_chunk = data[:,channel]
#labels_chunk = labels[:,channel]

# create artificial training data
channel = 4
event1 = data[582400:583200,channel]
event2 = data[909000:910500,channel]
event3 = data[1801000:1802000,channel]
noise = data[979000:990000,channel]
data_chunk = np.concatenate((noise, event1, noise, event2, noise, event3, noise, event2, noise))
labels_chunk = np.zeros(data_chunk.size)
labels_chunk[11300:11600] = 1
labels_chunk[23500:23800] = 1
labels_chunk[35600:36000] = 1
labels_chunk[47800:48400] = 1

# plot
fig, ax1 = plt.subplots()
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

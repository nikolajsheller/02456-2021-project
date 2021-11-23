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

print(data.shape)
print(labels.shape)

print(data[:,0].shape)

channel = 4
data_chunk = data[:,channel]
labels_chunk = labels[:,channel]

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Sample data vs label data')
ax1.plot(data_chunk)
ax2.plot(labels_chunk)
plt.show()

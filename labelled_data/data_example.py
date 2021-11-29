import numpy as np
import matplotlib.pyplot as plt
from .data_handling.tools.constants import *

# replace with files you have, if these don't exist..
filename_data = os.path.join(RAWDATA_LABELLED_PATH, 'dca6327d8fa8_20210330_raw_20210330T152242Z_ds_5_data.npy')
filename_labels = os.path.join(RAWDATA_LABELLED_PATH, 'dca6327d8fa8_20210330_raw_20210330T152242Z_ds_5_labels.npy')

data = np.load(filename_data)[:,1]
labels = np.load(filename_labels)[:,1]

# plot
fig, ax1 = plt.subplots()
fig.set_figwidth(25)
color = 'tab:blue'
ax1.set_ylabel('data', color=color)
ax1.plot(data[800000:900000], color=color)
ax1.tick_params(axis='y', color=color)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('labelled', color=color)
ax2.plot(labels[800000:900000], color=color)
ax2.tick_params(axis='y', color=color)
plt.show()
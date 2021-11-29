# %% [markdown]
# # Import

# %%
import fpmodules as fp
import pandas as pd
from evex_scout import *
from fpmodules.tools.constants import EVENTS_CACHE_PATH
from fpmodules.fpio.raw_data import load_data_from_files
from fpmodules.tools.dbtools import to_pdatime
from fpmodules import BlobManager
import matplotlib.pyplot as plt
import fpmodules.tools as tools

# %% [markdown]
# # Constants

# %%
RAWDATA_CACHE_PATH = os.path.join(EVENTS_CACHE_PATH, 'RawData')

SNR = 10
window_res = 10
step_interval = 1
window_size = 2
erosion_mask_length = 10
expand_dist = 600
hard_threshold = 5
saturation_threshold = 30000

# {1: "C", 2: "B", 3: "A", 4: "D"}
channels = [1, 1, 2, 2, 3, 3, 4, 4]
colors = tools.get_instrument_info()['Scout']['cmap']
tools.get_instrument_info()

# %% [markdown]
# # Get data

# %%
fp.get_session(sessionid=686)

# %%
fp.get_session(labelled=True)

# %%
measurements = fp.dbquery('select * from measurement where sessionid=686')
measurements = measurements.sort_values('TimeId')
measurements = to_pdatime(measurements, delete=False)
measurements.head()

# %%
insects = fp.get_insects(sessionid=686, all_segments=True)
insects.head()

# %% [markdown]
# 

# %% [markdown]
# # Define functions

# %%
def get_start_and_stop(data, times):
    fss = get_samplerates(times)
    fs = int(np.median(fss))
    data_matrix = data.copy() # transpose
    data_matrix, medians, stds, interp = remove_rolling_mean(data_matrix.astype(int), fs, window_res=window_res, step_interval=step_interval, window_size=window_size)
    event_mask_master = threshold(data_matrix, stds, interp, SNR=SNR, erosion_mask=np.ones(erosion_mask_length), expand_dist=expand_dist, hard_threshold=hard_threshold)
    start_inds, stop_inds = label_events(event_mask_master)

    return start_inds, stop_inds

def get_seconds(event):
    dt = event.info['Datetime']
    seconds = int(dt.strftime('%s'))
    return seconds

# %%
def create_labelled_data(blob):
    file_list = blob_mgr.download_blobs([blob], RAWDATA_CACHE_PATH, container='scouts')
    data, times = load_data_from_files(file_list, path, ds=1)
    if len(data) == 0:
        return

    raw_start = pd.Timestamp(datetime.datetime.utcfromtimestamp(int(times[0])))
    raw_end = pd.Timestamp(datetime.datetime.utcfromtimestamp(int(times[-1])))
    print('raw_start:', raw_start)
    print('raw_end:', raw_end)

    in_range = (measurements['Datetime'] >= raw_start) & (measurements['Datetime'] <= raw_end)
    _meas = measurements[in_range]

    if len(_meas) == 0:
        # no measurements found in time range
        print('Found no measurements in range, continuing...')
        return
    print(f'Found {len(_meas)} measurements')

    print('Running event extraction')
    start_inds, stop_inds = get_start_and_stop(data, times)
    labels = np.zeros_like(data)

    if (len(start_inds) == 0) & (len(stop_inds) == 0):
        return
    print('len(_meas)', len(_meas))
    for i, m_id in enumerate(_meas['Id'].tolist()):
        if start_inds[i] > stop_inds[i]:
            print('start lower than stop, continueing...')
            continue
        length =  stop_inds[i] - start_inds[i]


        filename = blob.replace('/', '_').split('.')[0] + '_ds_5'
        event_path = os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData')

        # Find insects
        if m_id in insects['MeasurementId'].tolist():
            _insects = insects[insects['MeasurementId'] == m_id]
            labels = np.zeros_like(data[start_inds[i]:stop_inds[i],:])

            for c in range(0,8):
                if channels[c] in _insects['SegmentId'].tolist():
                    labels[:,c] = 1
            # save insect
            with open(event_path + '/insects/' + filename +str(m_id)+ '_data.npy', 'wb') as f:
                np.save(f, data[start_inds[i]:stop_inds[i],:])
            with open(event_path + '/insects/' + filename + str(m_id)+'_labels.npy', 'wb') as f:
                np.save(f, labels)
        # save noise
        else:
            labels = np.zeros_like(data[start_inds[i]:stop_inds[i],:])
            with open(event_path + '/noise/' + filename + str(m_id)+'_data.npy', 'wb') as f:
                np.save(f, data[start_inds[i]:stop_inds[i],:])
            with open(event_path + '/noise/' + filename + str(m_id)+'_labels.npy', 'wb') as f:
                np.save(f, labels)

        if ((i < len(_meas)-1) and (stop_inds[i] + length < len(data[:,0]))):
            if stop_inds[i] + length < start_inds[i+1]:
                zero_data = data[stop_inds[i]:stop_inds[i]+length,:]
                zero_labels = np.zeros_like(zero_data)
                with open(event_path + '/zeros/' + filename + str(m_id)+'_data.npy', 'wb') as f:
                    np.save(f, zero_data)
                with open(event_path + '/zeros/' + filename + str(m_id)+'_labels.npy', 'wb') as f:
                    np.save(f, zero_labels)
                # save labels

    return data, times, labels, start_inds, stop_inds, _meas

# %%
path = os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData/zeros')
files = os.listdir(os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData/zeros'))
for file in files[0:6]:
    if file.split('_')[-1] == 'data.npy':
        data = np.load(os.path.join(path,file))
        plt.plot(data)
        plt.show()

# %% [markdown]
# # Find blobs

# %%
mac = 'dca6327d8fa8'
path=RAWDATA_CACHE_PATH
blob_mgr = BlobManager(configuration='rclone')

# %%
date = 20210330
blob_list = blob_mgr.list_blobs(container='scouts', subdir=f"{mac}/{str(date)}/raw/")
blob_list = [b for b in blob_list if b.endswith(".raw.gz")]
print('Number of raw data files:', len(blob_list))

# %% [markdown]
# # Test and visualize

# %%
date = 20210330

blob_list = blob_mgr.list_blobs(container='scouts', subdir=f"{mac}/{str(date)}/raw/")
blob_list = [b for b in blob_list if b.endswith(".raw.gz")]
print('Number of raw data files:', len(blob_list))

# %%
for blob in blob_list[75:76]:
    data, times, labels, start_inds, stop_inds, _meas = create_labelled_data(blob)

# %%
for c in range(0,8):
    plt.plot(data[start_inds[0]:stop_inds[0],c], label=str(channels[c]), color=colors[c])
plt.legend()

# %%
start_inds[0]

# %%
stop_inds[0]

# %%
plt.figure(figsize=(25,5))
for c in range(0,1):
    plt.plot(data[1451375:1800000,c], label=str(channels[c]), color=colors[c])
plt.legend()


# %% [markdown]
# # Run for all - THIS WILL TAKE A WHILE

# %%
if False:
    for blob in blob_list:
        create_labelled_data(blob)

# %% [markdown]
# # Run for all dates - THIS WILL TAKE FOREVER

# %%


if True:
    dates = measurements['DateId'].sort_values().unique().tolist()
    for date in dates[0:7]:
        print(date)
        path=RAWDATA_CACHE_PATH
        blob_mgr = BlobManager(configuration='rclone')
        blob_list = blob_mgr.list_blobs(container='scouts', subdir=f"{mac}/{str(date)}/raw/")
        blob_list = [b for b in blob_list if b.endswith(".raw.gz")]
        print('Number of raw data files:', len(blob_list))

        for blob in blob_list:
            text_file = os.path.join(RAWDATA_CACHE_PATH, blob.replace('/', '_').split('.')[0] + '.txt')
            if os.path.exists(text_file):
                continue
            create_labelled_data(blob)
            os.remove(os.path.join(RAWDATA_CACHE_PATH, blob.replace('/', '_').split('.')[0] + '.raw.gz'))
            with open(text_file,"w") as variable_name:
                variable_name.write('Test')



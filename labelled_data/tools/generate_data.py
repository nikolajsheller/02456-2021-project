import os.path
import pandas as pd

# Internal FaunaPhotonics module - must not be shared
from .evex_scout import *

# Internal FaunaPhotonics module - must not be shared
print('if you have fpmodules < 2.10.13, use the import line below instead - check the code')
from fpmodules.fpio.raw_data import load_data_from_files
from fpmodules import BlobManager
from fpmodules.tools.dbtools import to_pdatime
import fpmodules as fp

from .constants import *


def get_start_and_stop(data, times):
    """

    :param data: raw data
    :param times: raw data times
    :return: start and stop times for events
    """
    fss = get_samplerates(times)
    fs = int(np.median(fss))
    data_matrix = data.copy()  # transpose
    data_matrix, medians, stds, interp = remove_rolling_mean(data_matrix.astype(int), fs, window_res=window_res,
                                                             step_interval=step_interval, window_size=window_size)
    event_mask_master = threshold(data_matrix, stds, interp, SNR=SNR, erosion_mask=np.ones(erosion_mask_length),
                                  expand_dist=expand_dist, hard_threshold=hard_threshold)
    start_inds, stop_inds = label_events(event_mask_master)

    return start_inds, stop_inds


def get_seconds(event):
    """

    :param event: fp.Event
    :return: Timestamp from events in seconds
    """
    dt = event.info['Datetime']
    seconds = int(dt.strftime('%s'))
    return seconds


def get_insects_and_measurements(max_val):
    measurements = fp.dbquery('select * from measurement where sessionid=686').sort_values('TimeId')
    measurements = to_pdatime(measurements, delete=False)
    feat = fp.get_features(sessionid=686, featureid=12)
    feat = feat[(feat['WavelengthId'] == '810') & (feat['SegmentId'] == 4)]
    feat = feat[feat['max'] > max_val]
    feat['MeasurementId'] = feat['MeasurementId'].astype(int)
    insects = fp.get_insects(sessionid=686, all_segments=True)
    insects = insects[insects['MeasurementId'].isin(feat['MeasurementId'])]

    return measurements, insects


def create_labelled_data(blob, tight=True, ds=5,max_val=0,segment=4, channel=7):
    """

    :param blob: blob from storage
    :param ds: downsample
    :return:
    """
    os.makedirs(RAWDATA_LABELLED_PATH, exist_ok=True)

    # Get data from DB
    measurements, insects = get_insects_and_measurements(max_val)

    # Get files from blob storage
    blob_mgr = BlobManager(configuration='rclone')
    file_list = blob_mgr.download_blobs([blob], RAWDATA_CACHE_PATH, container='scouts')
    data, times = load_data_from_files(file_list, RAWDATA_CACHE_PATH)
    if len(data) == 0:
        return

    raw_start = pd.Timestamp(datetime.datetime.utcfromtimestamp(int(times[0])))
    raw_end = pd.Timestamp(datetime.datetime.utcfromtimestamp(int(times[-1])))
    print('Raw data start time:', raw_start)
    print('Raw data end time:', raw_end)

    in_range = (measurements['Datetime'] >= raw_start) & (measurements['Datetime'] <= raw_end)
    _meas = measurements[in_range]
    _meas = _meas.sort_values(['Datetime','Id'])

    if len(_meas) < 1:
        print('Found 0 measurements in range, continuing...')
        return
    print(f'Found {len(_meas)} measurements')

    print('Running event extraction')
    start_inds, stop_inds = get_start_and_stop(data, times)

    if (len(start_inds) == 0) & (len(stop_inds) == 0):
        return
    if len(_meas) != len(start_inds):
        return

    if tight:
        for i in range(0, len(start_inds)):
            start_inds[i], stop_inds[i] = tight_cut(data, start_inds[i], stop_inds[i])

    print('Saving files')
    filename = blob.replace('/', '_').split('.')[0] + '_ds_' + str(ds)
    save_raw_data(filename, _meas, insects, data, start_inds, stop_inds, ds=ds, segment=segment, channel=channel)

    return


def save_raw_data(filename, measurements, insects, data, start_inds, stop_inds, ds, segment, channel):
    data = data[:, channel]
    labels = np.zeros_like(data)

    for i, dt in enumerate(measurements['Datetime'].tolist()):
        insect_measurement = insects[(insects['Datetime'] == dt) & (insects['SegmentId'] == segment)]

        if len(insect_measurement['MeasurementId'].unique()) > 1:
            labels[start_inds[i]:stop_inds[i]] = 2

        if len(insect_measurement) > 0:
            labels[start_inds[i]:stop_inds[i]] = 1

    data = data/np.linalg.norm(data)
    data = data[::ds]

    mean = np.mean(data)
    epsilon = mean*1.1
    labels = labels[::ds]
    for d in range(0, len(data), 5000):
        if (d+5000) > len(data):
            slice = data[d:]
        else:
            slice = data[d:d+5000]
        if np.max(slice) < epsilon:
            labels[d:d+5000] = 2

    data = data[labels < 2]
    labels = labels[labels < 2]
    if len(data) > 0:
        save_data('', filename, data, labels)
    return


def save_data(folder, filename, data, labels):
    path = os.path.join(RAWDATA_LABELLED_PATH, folder)
    with open(os.path.join(path, filename + '_data.npy'), 'wb') as f:
        np.save(f, data)
    with open(os.path.join(path, filename + '_labels.npy'), 'wb') as f:
        np.save(f, labels)
    return


def tight_cut(data, start_inds, stop_inds):
    start_index = None
    stop_index = None
    for j in range(start_inds, stop_inds):
        if start_index is not None:
            break
        if (int(np.max(data[j:j + 100, 7])) - int(data[j, 7]) > 20) and start_index is None:
            start_index = j

    for j in range(stop_inds, start_inds, -1):
        if stop_index is not None:
            break
        if (int(np.max(data[j - 100:j, 7])) - int(data[j, 7]) > 20) and stop_index is None:
            stop_index = j
    if stop_index is None:
        stop_index = stop_inds
    if start_index is None:
        start_index = start_inds
    if np.fabs(stop_index - start_index) > 20000:
        return start_inds, stop_inds
    return start_index, stop_index

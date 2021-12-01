import os.path
import pandas as pd

from .evex_scout import *

print('if you have fpmodules < 2.10.13, use the import line below instead - check the code')
# from fpmodules.plotting import load_data_from_files

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


def create_labelled_data(blob, chunks=True, ds=5):
    """

    :param blob: blob from storage
    :param chunks: if True save as events, if False save as 10 minute files
    :param ds: downsample
    :return:
    """
    os.makedirs(RAWDATA_LABELLED_PATH, exist_ok=True)
    os.makedirs(os.path.join(RAWDATA_LABELLED_PATH, 'insects'), exist_ok=True)
    os.makedirs(os.path.join(RAWDATA_LABELLED_PATH, 'noise'), exist_ok=True)

    # Get data from DB
    measurements = fp.dbquery('select * from measurement where sessionid=1307').sort_values('TimeId')
    measurements = to_pdatime(measurements, delete=False)
    insects = fp.get_insects(sessionid=1307, all_segments=True)
    #features = fp.get_features(sessionid=1307, all_segments=True)

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

    if len(_meas) == 0:
        print('Found no measurements in range, continuing...')
        return
    print(f'Found {len(_meas)} measurements')

    print('Running event extraction')
    start_inds, stop_inds = get_start_and_stop(data, times)

    if (len(start_inds) == 0) & (len(stop_inds) == 0):
        return
    if len(_meas) != len(start_inds):
        return

    print('Saving files')
    filename = blob.replace('/', '_').split('.')[0] + '_ds_' + str(ds)
    if chunks:
        save_chunks(filename, _meas, insects, data, start_inds, stop_inds, ds=ds)
    else:
        save_raw_data(filename, _meas, insects, data, start_inds, stop_inds, ds=ds)

    return


def save_chunks(filename, measurements, insects, data, start_inds, stop_inds, ds):
    for i, m_id in enumerate(measurements['Id'].tolist()):
        if start_inds[i] > stop_inds[i]:
            print('start lower than stop, continueing...')
            continue
        event = data[start_inds[i]:stop_inds[i], :]
        labels = np.zeros_like(event)
        # process insect
        if m_id in insects['MeasurementId'].tolist():
            # Find insects
            _insects = insects[insects['MeasurementId'] == m_id]
            for c in range(0, 8):
                if channels[c] in _insects['SegmentId'].tolist():
                    labels[:, c] = 1

            # save insect
            save_data('insects', filename, event[::ds], labels[::ds])

        # save noise
        else:
            save_data('noise', filename, event[::ds], labels[::ds])


def save_raw_data(filename, measurements, insects, data, start_inds, stop_inds, ds):
    for i, m_id in enumerate(measurements['Id'].tolist()):
        if start_inds[i] > stop_inds[i]:
            print('start lower than stop, continueing...')
            continue

        labels = np.zeros_like(data)

        if m_id in insects['MeasurementId'].tolist():
            _insects = insects[insects['MeasurementId'] == m_id]

            for c in range(0, 8):
                if channels[c] in _insects['SegmentId'].tolist():
                    labels[start_inds[i]:stop_inds[i], c] = 1

    save_data('', filename, data[::ds], labels[::ds])
    return


def save_data(folder, filename, data, labels):
    path = os.path.join(RAWDATA_LABELLED_PATH, folder)
    with open(os.path.join(path, filename + '_data.npy'), 'wb') as f:
        np.save(f, data)
    with open(os.path.join(path, filename + '_labels.npy'), 'wb') as f:
        np.save(f, labels)
    return

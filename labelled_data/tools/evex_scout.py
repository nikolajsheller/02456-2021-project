#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:33:44 2020

@author: Rami El-Rashid

This is the version of the event extractor being developed to run on the Scout.
Notes: 
    - the second dimension is the number of channels.
"""

import datetime, time
import os, sys
import argparse
import gzip
import json
from itertools import product

import numpy as np

if os.getcwd() not in sys.path: # To avoid import madness
    sys.path.append(os.getcwd())
from .evex_calc import log, remove_rolling_mean, threshold, label_events, extract_events, data_type, filter_events

#%% I/O functions
def get_external_arguments():
    """Receives arguments from console"""
    parser = argparse.ArgumentParser(description="Event extractor parameters")
    parser.add_argument("infile", help="Path to the file on which to run the event extractor. Required.")
    parser.add_argument("--outdir", help="Directory to save the extracted .json files. Required.", default='C:/Data/Rami/TempEvents/R/')
    parser.add_argument("--sn", help="Serial number of the device (str)", default="VirtualTest")
    parser.add_argument("-m", type=int, help="Method of file saving, can be 0 (gzip) or 1 (json), default 0", choices=[0,1], default=0)
    parser.add_argument("-v", type=int, help="Verbose, can be 0 or 1, default 0.", choices=[0,1], default=0)
    args = vars(parser.parse_args())
    return args["infile"], args["outdir"], args["sn"], args["m"], args["v"]


def read_gzip(path, dtype=np.uint16):
    """Reads gzip file contents into 1d numpy array"""
    with gzip.open(path, "rb") as infile:
         data = np.frombuffer(infile.read(), dtype=dtype)
    return data


def load_filedata(filename, dtype):
    try:
        raw_data = read_gzip(filename, dtype=dtype)
    except Exception as err:
        log(f"Could not open file {filename}. " + str(err))
        return None
    return raw_data['times'], raw_data['data'].astype(int)
    
    
def format_channels(values, segments, wavelengths, segment_map):
    def atom(x):
        return x.tolist() if len(x) > 1 else x[0]
    
    if isinstance(values, np.ndarray):
        if len(values.shape) == 1:
            values = values.reshape(-1, len(values))
    data = []
    for i, (segment, wavelength) in enumerate(product(segments, wavelengths)):
        segment_name = segment_map[segment]
        column = dict(segmentid=segment,
                 segmentname=segment_name,
                 wavelength=wavelength,
                 value=atom(values[:,i]))
        data.append(column)
    return dict(channels=data)
    
def data_to_json(timestamp, serialnumber="VirtualTest", processingpath="unknown", **entries):
    """Parameters:
        timestamp: str or similar, necessary parameter
        data: if it is an event, the event data as np.array
        entries: all other entries as keyword arguments, going into 'data'
    """
    data_dict = {"type": processingpath}
    data_dict.update(**entries)
    final_dict = dict(UTC=timestamp, SN=serialnumber, data=data_dict)
    return final_dict
            
       
def save_json(json_data, path):
    # This is to be changed accordingly.
    ppath = json_data.get("data", "").get("type", "")
    fname = json_data.get("UTC", "NO_TIME") + "_" + ppath + ".json"
    with open(os.path.join(path, fname), 'w') as f:
        json.dump(json_data, f)
    
def send_to_iothub(*args, **kwargs):
    raise NotImplementedError
    
def get_githash():
    try:
        import subprocess
        gh = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return gh.decode("utf-8")
    except Exception as err:
        log(str(err))
        return "Unknown"

#%%          
def save_chunk(files, zfile_sizes, save_path, chunk_name, counter=0, limit=256000, singular_file_limit=256000, verbose=False):
    size = 0
    with gzip.open(os.path.join(save_path, chunk_name), "wb") as f:
        while size < limit and counter < len(files):
            if zfile_sizes[counter] > singular_file_limit:
                handle_large_file(save_path, files[counter], counter, limit)  # Maybe collect these in a list and handle afterwards?
                counter += 1
                continue
            f.write(files[counter])
            counter += 1
            f.flush()
            chunksize = os.fstat(f.fileno()).st_size
            size = chunksize + zfile_sizes[counter]
    if verbose:
        log(f"{counter}/{len(files)} events saved...({size/1000}KB)")
    return counter
    
def handle_large_file(save_path, event, counter, limit):
    #filename = "Large_event_" + str(counter) + ".gz"
    #filename = "Large_event_extracttime_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%fZ")
    filename = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%fZ") + "_large_event.gz"
    log(f"Event {counter} is larger than {limit/1000}KB. Skipping this event")
    with gzip.open(os.path.join(save_path, filename), "wb") as f:
        f.write(event)
        
def save_all_chunks(filename, save_path, msg_list, file_sizes, limit=256000, verbose=False):
    counter, chunk_counter = 0, 0
    while counter < len(msg_list):
        chunk_name = os.path.split(filename)[-1] + "_events_" + str(chunk_counter).zfill(3) + ".gz"
        counter = save_chunk(msg_list, file_sizes, save_path, 
                             chunk_name, counter=counter, limit=limit, verbose=verbose)
        chunk_counter += 1
    return True
    
#%% TIMESTAMP-RELATED
   
def get_samplerates(times):
    _, counts = np.unique(times.astype(int), return_counts=True)
    return counts

def convert_timestamp_to_dt(ts):
    dt = datetime.datetime.utcfromtimestamp(ts)
    return dt.isoformat()

def count_saturations(data, saturation_threshold=30000):
   #Counts the number of saturated samples in each channel over a ten-minute file
   saturation = data > saturation_threshold
   saturated_spikes = np.sum(saturation,axis=0)
   
   return saturated_spikes.astype(int)

def timestamp_from_filename(filename):
    tsformat="%Y%m%dT%H%M%SZ"
    now_format="%Y%m%dT%H%M%S.%f"
    path = os.path.split(filename)[-1]
    ts = path.split(".")[0]
    try:
        _ = datetime.datetime.strptime(ts, tsformat)
    except Exception as err:
        log(str(err))
        ts = datetime.datetime.now().strftime(now_format)
    return ts


#%% Rain filter
def is_skiprainfilter_on():
    FILEDIR = '/etc/fauna'
    fname = 'SKIPRAINFILTER.fptxt'
    fpath = os.path.join(FILEDIR, fname)
    return os.path.exists(fpath)


def apply_rain_filter(event_freq, event_list, mean_stds):
    if event_freq > 60 and not is_skiprainfilter_on() and len(event_list) > 0:
        included_event_list, excluded_event_list = filter_events(event_list,mean_stds)

        if len(included_event_list)/len(event_list)>0.7:
            included_event_list = event_list
            excluded_event_list = []
    else:
        included_event_list = event_list
        excluded_event_list = []
    return included_event_list, excluded_event_list


def save_events(save_method, filtered_event_list, fs, serialnumber, segments, wavelengths, segment_map, save_path, filename, filesize_limit):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    msg_list = []
    if save_method == 0:
        if verbose:
            print("Compiling and zipping events...")
        zipped_list, file_sizes = [], []
    else:
        if verbose:
            print("Compiling events...")
    for event, ts in filtered_event_list:
        mdict = dict(sampling_rate=fs)
        measurements = [dict(name=n, value=v) for n, v in mdict.items()]
        msg = data_to_json(ts, serialnumber=serialnumber, 
                           processingpath="event", 
                           measurements=measurements,
                           value=format_channels(event, segments, wavelengths, segment_map))
        if save_method == 1:
            save_json(msg, save_path)
        else:
            serial_msg = bytes(json.dumps(msg), encoding="utf-8")
            msg_list.append(serial_msg)
            zipped_msg = gzip.compress(serial_msg)
            zipped_list.append(zipped_msg)
            file_sizes = [sys.getsizeof(file) for file in zipped_list] + [0]

    if save_method == 0:
        if verbose:
            print("Saving zipped events...")
        save_all_chunks(filename, save_path, msg_list, file_sizes, verbose=verbose, limit=filesize_limit)


def save_metadata(fs, fss, githash, event_freq, medians, channel_info, stds, data_matrix, saturation_threshold, filtered_event_list, event_list, filename, start_time, serialnumber, save_path):
    mdict = dict(sampling_rate = fs,
                 sampling_rate_mean = np.mean(fss[1:-1]),
                 sampling_rate_std = np.std(fss[1:-1]),
                 git_hash = githash,
                 event_freq = event_freq,
                 noise_medians = format_channels(np.mean(medians, axis=0), *channel_info),
                 noise_stds = format_channels(np.mean(stds, axis=0), *channel_info),
                 saturation_counts = format_channels(count_saturations(data_matrix, saturation_threshold).astype(float), *channel_info),
                 num_filtered_events = len(filtered_event_list),
                 num_events = len(event_list))
    measurements = [dict(name=n, value=v, SensorName="Event Extractor") for n, v in mdict.items()]
    entries = dict(filename=filename,
                   run_time=np.round(time.time()-start_time, 1),
                   num_events=len(event_list),
                   measurements=measurements)
    msg = data_to_json(timestamp_from_filename(filename),
                       serialnumber=serialnumber,
                       processingpath="environmental",
                       **entries)
    save_json(msg, save_path)


#%% RUN
def main():
    
    # Calculation parameters
    SNR = 10
    window_res = 10
    step_interval = 1
    window_size = 2
    erosion_mask_length = 10
    expand_dist = 600
    hard_threshold = 5
    saturation_threshold = 30000
    
    # File parameters (constants)
    segments = [1, 2, 3, 4]
    wavelengths = [970, 810]
    segment_map = {1: "C", 2: "B", 3: "A", 4: "D"}
    channel_info = (segments, wavelengths, segment_map)
    filesize_limit = 128000
       
    # File
    global verbose
    filename, save_path, serialnumber, save_method, verbose = get_external_arguments()
    githash = get_githash()
    start_time = time.time()
    if verbose:
        print("Opening file...")
    times, data_matrix = load_filedata(filename, dtype=data_type)
    
    # Calculations    
    if verbose:
        print("Extracting events...")
    data_matrix = data_matrix.astype(int) # Careful with this! Maybe keep it in int16
    fss = get_samplerates(times)
    fs = int(np.median(fss))
    data_matrix, medians, stds, interp = remove_rolling_mean(data_matrix, fs, window_res=window_res, step_interval=step_interval, window_size=window_size)
    event_mask_master = threshold(data_matrix, stds, interp, SNR=SNR, erosion_mask=np.ones(erosion_mask_length), expand_dist=expand_dist, hard_threshold=hard_threshold)
    start_inds, stop_inds = label_events(event_mask_master)
    event_list = extract_events(data_matrix, start_inds, stop_inds)
    print(event_list)
    times_list = [convert_timestamp_to_dt(ts)
            for ts in times[start_inds]]
    print('times_list', times_list)
    events_and_times = list(zip(event_list, times_list))

    # Calculating frequency
    file_length = 10 # 10 minute file length hardcoded
    event_freq = len(events_and_times) / file_length

    # Calculate file std
    mean_stds = np.mean(stds, axis=0)

    # Filter events
    included_events, excluded_events = apply_rain_filter(event_freq, events_and_times, mean_stds)

    # Saving events
    save_events(save_method, included_events, fs, serialnumber, segments, wavelengths, segment_map, save_path, filename, filesize_limit)
    save_events(save_method, excluded_events, fs, serialnumber, segments, wavelengths, segment_map, os.path.join(save_path, 'excluded'), filename, filesize_limit)

    # Saving metadata
    save_metadata(fs, fss, githash, event_freq, medians, channel_info, stds, data_matrix, saturation_threshold, included_events, events_and_times, filename, start_time, serialnumber, save_path)

    now = datetime.datetime.now().isoformat()
    log(f"""{filename} finished processing at {now} in {np.round(time.time()-start_time, 1)} seconds. \n
        {len(events_and_times)} events found""")

    # debug info for prod/ install webpage
    fhdebug = open("/tmp/fauna_debug_eventcount","w")
    fhdebug.write("%d" % len(events_and_times))
    fhdebug.close()

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:41:42 2020

@author: Rami El-Rashid

List of functions to be imported for Scout event extraction.
Make sure no modules other than numpy are imported.

"""

import numpy as np
import os
import os.path
data_type = np.dtype([('times', '<f8'), ('read_counter', '<u2'), ('data', '<u2', (8,))]) 
# Consider moving this

def log(message, params={}):
    #print(message.format(**params))
    print(message)
    
        
def remove_rolling_mean(data_matrix, fs, step_interval=1, window_size=2,
                        window_res=10, n_channels=8):

    """

    Inputs
    ----------
    data_matrix : np.array
        The file
    step_interval : int
        Time between each evaluation point
    window_size : int
        The width of the rolling mean window
    win_res : int
        downsampling level, how many points to skip, 1=all points used

    Outputs:
    ----------
    --tba

    """
    n_frames = len(data_matrix)

    # Width of window
    window = int(fs*window_size)                # Compute number of samples in a 2 second window
    n_steps = int(n_frames/(step_interval*fs))  # Compute number of 1-second steps to traverse our recording

    # Compute which indices correspond to 1-second steps through our recording
    steps = np.linspace(0, n_frames-1, n_steps, dtype=int)  # X coordinates of the sample points

    # Init zeroed structures to contain means and std-deviations, for 1-second intervals
    means = np.zeros([n_steps, n_channels])
    stds = np.zeros([n_steps, n_channels])

    # Compute std and means for 1-second intervals, looking at 1 second in the past and 1 second in the future
    for j in range(n_steps):

        start = steps[j] - int(window/2)
        stop = steps[j] + int(window/2)

        if start < 0:
            start = 0

        elif j == n_steps:
            stop = steps[-1]
            start = stop - window

        # Select values from raw data
        array = data_matrix[start:stop:window_res, :]

        # Calculate median
        m_vals = np.median(array, axis=0)
        # Calculate std
        std_bel_mean = np.zeros(n_channels)
        for k in range(n_channels):
            std_bel_mean[k] = np.abs(np.std(array[array[:, k] < m_vals[k], k]))

        means[j, :] = m_vals
        stds[j, :] = std_bel_mean

#    print('interpolating')
    x = np.linspace(0, n_frames/fs, n_frames)
    xp = x[steps]
    for j in range(n_channels):
        data_matrix[:, j] = data_matrix[:, j] - np.interp(x, xp, means[:, j])

    return data_matrix, means, stds, [x, xp]


def threshold(data_matrix, stds, interp, SNR=10, erosion_mask=np.ones(10),
              expand_dist=20000/10, hard_threshold=0):

    expand_dist = int(expand_dist)
    x = interp[0]
    xp = interp[1]

    n_frames = np.shape(data_matrix)[0]
    n_channels = np.shape(data_matrix)[1]
    event_mask_master = np.zeros(n_frames, dtype=bool)
    for j in range(n_channels):
#        print(j, 'Creating threshold')
        th = np.interp(x, xp, np.abs(stds[:, j]))*SNR
#        print(j, 'thresholding')
        ind0 = data_matrix[:, j] > th + hard_threshold

#        print(j, 'eroding')
        conv = np.convolve(ind0, erosion_mask, 'same')
        ind_eroded = (conv >= erosion_mask.sum())

#        print(j, 'expanding')
        diffs = np.where(np.diff(ind_eroded) == 1)[0]
        start_inds = diffs[::2].astype(int)
        stop_inds = diffs[1::2].astype(int)

        # Make absolutely sure nothings 'Inside out'
        if len(start_inds) != len(stop_inds):
            log(f"Inside-out anomaly: {len(start_inds)} start and {len(stop_inds)} stop indices")
            return
        #assert(len(start_inds) == len(stop_inds))

        # Expand and add to event mask
        for k in range(len(start_inds)):
            start = start_inds[k] - expand_dist
            stop = stop_inds[k] + expand_dist
            event_mask_master[start:stop] = 1


    return event_mask_master

#for hth in (0,1,2,5,10,20,50):
#    j=0
#    ind0 = data_matrix[:, j] > th + hth
#    print(f"{hth}: {np.sum(ind0)}")


def label_events(event_mask_master):
    event_mask_master[0] = 0
    
    starts = np.where(event_mask_master == False)[0]
    start_inds = starts[np.where(np.diff(starts) > 1)]

    event_mask_master[-1] = 1

    stops = np.where(event_mask_master)[0]


    stop_inds = stops[np.where(np.diff(stops) > 1)]
    return start_inds, stop_inds


def extract_events(data_matrix, start_inds, stop_inds):
    event_list = []
    for j in range(np.min([len(start_inds), len(stop_inds)])):
        start_ind = start_inds[j]
        stop_ind = stop_inds[j]
        if start_ind < stop_ind:
            event = data_matrix[start_ind:stop_ind, :]
        else:
            log(f"Event stop index ({stop_ind}) lower than start index ({start_ind})")
            #event = data_matrix[start_ind:stop_ind, :]
            continue
        event_list.append(event)
    #print(len(event_list), 'events found') # Is this needed
    return event_list

def calc_width(event, stds, is_active):
    #signal: length M data array with the event data (summed or single channel)
    #std: mean standard deviation of the data file of the noise for each channel
    summed_signal = np.sum(event,axis=1)
    max_ind = np.argmax(summed_signal)
    thresh_low = np.where(summed_signal[:max_ind] < np.max(stds))[0]
    thresh_high = np.where(summed_signal[max_ind:] < np.max(stds))[0]
    
    #find beginning of insect/drop
    if len(thresh_low)==0:
        raindrop_start = 0
    else:
        raindrop_start = thresh_low[-1]
    
    #find end of insect/drop
    if len(thresh_high)==0:
        raindrop_stop = len(event[:,0])
    else:
        raindrop_stop = thresh_high[0]+max_ind
    
    channel_width = []
    for c in range(len(event[0,:])):
        if not is_active[c]:
            continue
        signal = event[:,c]
        max_ind_c = np.argmax(signal[raindrop_start:raindrop_stop])+raindrop_start
        thresh_low_c = np.where(signal[:max_ind_c] < stds[c])[0]
        thresh_high_c = np.where(signal[max_ind_c:] < stds[c])[0]

        #find beginning of insect/drop
        if len(thresh_low_c)==0:
            raindrop_start_c = 0
        else:
            raindrop_start_c = thresh_low_c[-1]
        
        #find end of insect/drop
        if len(thresh_high_c)==0:
            raindrop_stop_c = len(signal)
        else:
            raindrop_stop_c = thresh_high_c[0]+max_ind_c
        
        width_c = raindrop_stop_c-raindrop_start_c
        channel_width.append(width_c)

    mean_width = np.mean(channel_width)
    
    return raindrop_start, raindrop_stop, mean_width


def direction_going_down(event_data, is_active):
    #event_data: 8xM data array with the event data for all 8 channels
    #std: 8x1 vector with the mean standard deviation of the data file of the noise for each channel

    ibot_left = 1
    itop_left = 3
    itop_right = 5
    ibot_right = 7
       
    #Direction of left sections
    if (is_active[ibot_left]>0) & (is_active[itop_left]>0):
        bottom_left_data = event_data[:,ibot_left]
        top_left_data = event_data[:,itop_left]
        corr_left = np.correlate(bottom_left_data,top_left_data,mode='same')
        dir_left = np.argmax(corr_left) - len(top_left_data)//2
    else:
        dir_left = 0
    
    #Direction of right sections
    if (is_active[ibot_right]>0) & (is_active[itop_right]>0):
        top_right_data = event_data[:,itop_right]
        bottom_right_data = event_data[:,ibot_right]
        corr_right = np.correlate(bottom_right_data,top_right_data,mode='same')
        dir_right = np.argmax(corr_right) - len(top_right_data)//2
    else:
        dir_right = 0
    
    #Check direction
    if (dir_left >0) or (dir_right >0):
        down = True
    else:
        down = False
    return down


def get_rh():
    rh = None
    hum_file = "/tmp/fauna_debug_RH"
    if os.path.exists(hum_file):
        if not os.stat(hum_file).st_size == 0:
            with open(hum_file) as f:
                rh = f.read()
    return rh

def filter_events(event_list,mean_stds):
    included_events = []
    excluded_events = []

    rh = get_rh()
    if isinstance(rh, str):
        rh = float(rh)

    h = 65
    wl = 40
    wh = 500
    l = 1800
    m = 200
    
    for event in event_list:
        e = event[0]
        if e is None:
            continue
        if len(e[:,0]) < l and np.max(e) < m: # and rh > h:

            is_active = np.zeros(len(e[0,:]))
            for c in range(len(e[0,:])):
                if np.max(e[:,c])>3.5*mean_stds[c]:
                    is_active[c] = 1
                
            raindrop_start,raindrop_stop, mean_width = calc_width(e,mean_stds,is_active)
            if (mean_width > wl) & (mean_width < wh):
                if direction_going_down(e[raindrop_start:raindrop_stop,:],is_active):
                    if rh is not None:
                        if rh > h:
                            excluded_events.append(event)
                            continue
                    else:
                        excluded_events.append(event)
                        continue
        included_events.append(event)
    return included_events, excluded_events


def die():
    pass
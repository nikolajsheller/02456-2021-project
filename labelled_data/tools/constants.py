import os
import fpmodules.tools as tools
from fpmodules.tools.constants import EVENTS_CACHE_PATH
RAWDATA_CACHE_PATH = os.path.join(EVENTS_CACHE_PATH, 'RawData')
RAWDATA_LABELLED_PATH = os.path.join(EVENTS_CACHE_PATH, 'RawLabelledData')

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

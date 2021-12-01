from tools.constants import *
from fpmodules import BlobManager
from tools.data_generator import create_labelled_data
import fpmodules as fp

mac = 'dca6324634b1'

measurements = fp.dbquery('select * from measurement where sessionid=1307')
dates = measurements['DateId'].sort_values().unique().tolist()
delete_files = True

for date in dates[1:2]:
    blob_mgr = BlobManager(configuration='rclone')
    blob_list = blob_mgr.list_blobs(container='scouts', subdir=f"{mac}/{str(date)}/raw/")
    blob_list = [b for b in blob_list if b.endswith(".raw.gz")]
    print('Number of raw data files:', len(blob_list))

    for blob in blob_list:

        text_file = os.path.join(RAWDATA_CACHE_PATH, blob.replace('/', '_').split('.')[0] + '.txt')
        # if we ran over the file before, don't do it again. If you want to run it anyway, delete the text file.
        if os.path.exists(text_file) and delete_files:
            continue

        create_labelled_data(blob, chunks=True)
        create_labelled_data(blob, chunks=False)

        if delete_files:
            os.remove(os.path.join(RAWDATA_CACHE_PATH, blob.replace('/', '_').split('.')[0] + '.raw.gz'))
            with open(text_file,"w") as variable_name:
                variable_name.write('Test')

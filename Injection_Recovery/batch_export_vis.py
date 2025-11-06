import os
import sys

# add the path to your exoALMA source code
sys.path.append('./exoALMA_source_code')

# import the reduction utilities
from reduction_utils_exoalma import export_MS

# directory containing measurement sets
ms_dir = '/mnt/d/exoALMA_disk_data/data'

for fname in os.listdir(ms_dir):
    if fname.endswith('.ms') and os.path.isdir(os.path.join(ms_dir, fname)):
        ms_path = os.path.join(ms_dir, fname)
        print(f"Exporting {ms_path}...")
        export_MS(ms_path)
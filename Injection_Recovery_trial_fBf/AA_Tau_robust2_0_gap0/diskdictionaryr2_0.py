# Load the disk dictionaries for robust2.0 

import pickle

filename = "../all_disk_dicts_r2_0.pkl"  # Change to the filename you want to load
with open(filename, "rb") as f:
    disk = pickle.load(f)


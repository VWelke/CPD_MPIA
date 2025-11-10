# This file will be run by CASA to complete a full injection recovery imaging loop for one flux bin
# It will set as a basis for a SLURM job submission script to run multiple flux bins in parallel on a cluster

import os

# For injectloop and imageloop
target = 'AA_Tau'
gap_ix = '0'
subsuf = '0'


# For recover_loop


# Create pause file to enable controlled execution
with open('PAUSE_AFTER_FLUX_BIN', 'w') as f:
    pass  # Creates empty file

print("Starting injection loop...")
execfile('AA_Tau_robust2_0_gap0/AA_Tau_robust2_0_gap0_injectloop.py', globals())  

# Specify which flux bin to process for imaging
flux_bin_uJy = how to set?  # Change this to the desired flux bin (in μJy)

print(f"Starting imaging loop for {flux_bin_uJy} μJy flux bin...")
execfile('AA_Tau_robust2_0_gap0/AA_Tau_robust2_0_gap0_imageloop.py', globals())  

print(f"Starting recovery analysis for {flux_bin_uJy} μJy flux bin...")
execfile('AA_Tau_robust2_0_gap0/recover_loop_perflux.py', globals())

# Clean up pause file
if os.path.exists('PAUSE_AFTER_FLUX_BIN'):
    os.remove('PAUSE_AFTER_FLUX_BIN')
    print("Removed PAUSE_AFTER_FLUX_BIN file")

print("Pipeline completed successfully!") 
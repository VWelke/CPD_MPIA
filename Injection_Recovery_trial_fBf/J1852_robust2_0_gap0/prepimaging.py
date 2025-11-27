import os, sys, time
import numpy as np

import sys, os
sys.path.append(os.getcwd())

from custom_mask import custom_mask

import diskdictionaryr2_0 as disk

# specify target disk, gap, and mock file index
target, gap_ix, subsuf = 'AA_Tau', '0', '0'

# # # # # # #

# package up information to pass to CASA
f = open('whichdisk.txt', 'w')
f.write(target + '\n' + gap_ix + '\n' + subsuf)
f.close()

# preliminary CASA imaging to set up for image loop
os.system('/usr/local/bin/CASA/casa-6.6.1-17-pipeline-2024.1.0.8/bin/casa --nogui --nologger --nologfile -c prepimaging_casa.py')

# make a custom mask for the gap of interest
custom_mask(target, int(gap_ix), target+'_gap'+gap_ix+'.'+subsuf,
            buffer_factor=1.5)

# make a script to convert custom mask into CASA format
os.system('/usr/local/bin/CASA/casa-6.6.1-17-pipeline-2024.1.0.8/bin/casa --nogui --nologger --nologfile -c mask_to_casa.py')
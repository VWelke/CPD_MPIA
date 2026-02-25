import os, sys, time
import numpy as np

import sys, os
sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/')

from custom_mask import custom_mask

import diskdictionaryr2_0 as disk

# specify target disk, gap, and mock file index
target, gap_ix, subsuf = 'J1852', '0', '0'

# # # # # # #

# package up information to pass to CASA
f = open('whichdisk.txt', 'w')
f.write(target + '\n' + gap_ix + '\n' + subsuf)
f.close()

from custom_mask import custom_mask

import diskdictionaryr2_0 as disk
# preliminary CASA imaging to set up for image loop
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile < prepimaging_casa.py')

# make a custom mask for the gap of interest
custom_mask(target, int(gap_ix), target+'_gap'+gap_ix+'.'+subsuf,
            buffer_factor=1.5)

# make a script to convert custom mask into CASA format
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile < mask_to_casa.py')
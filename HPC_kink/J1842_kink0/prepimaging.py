import os, sys, time
import numpy as np

import sys, os
sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/')

from custom_mask import custom_mask

import diskdictionaryr2_0 as disk

# specify target disk, kink, and mock file index
target, kink_ix, subsuf = 'J1842', '0', '0'

# # # # # # #

# package up information to pass to CASA
f = open('whichdisk.txt', 'w')
f.write(target + '\n' + kink_ix + '\n' + subsuf)
f.close()

# preliminary CASA imaging to set up for image loop
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < prepimaging_casa.py')

# make a custom mask for the kink of interest
custom_mask(target, int(kink_ix), target+'_kink'+kink_ix+'.'+subsuf,
            buffer_factor=1.5, feature='kink')

# make a script to convert custom mask into CASA format
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < mask_to_casa.py')

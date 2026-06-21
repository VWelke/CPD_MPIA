import os, sys, time
import numpy as np

sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/')
from custom_mask import custom_mask

sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/')
import diskdictionary_eqC1 as disk

# specify target disk, gap, and mock file index
target, gap_ix, subsuf = 'SY_Cha', '0', '0'

# package up information to pass to CASA
f = open('whichdisk.txt', 'w')
f.write(target + '\n' + gap_ix + '\n' + subsuf)
f.close()

# preliminary CASA imaging to set up for image loop
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < prepimaging_casa.py')

# make a custom mask for the gap of interest
custom_mask(target, int(gap_ix), target+'_gap'+gap_ix+'.'+subsuf,
            buffer_factor=1.5)

# make a script to convert custom mask into CASA format
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < mask_to_casa.py')

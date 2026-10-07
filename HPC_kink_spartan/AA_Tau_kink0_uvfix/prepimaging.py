import os, sys, time
import numpy as np

sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/')
from custom_mask import custom_mask

sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/diskdictionary_r90/')
import diskdictionary_eqC1 as disk

# specify target disk, kink candidate, and mock file index
target, gap_ix, subsuf = 'AA_Tau', '2', '0'

# package up information to pass to CASA
f = open('whichdisk.txt', 'w')
f.write(target + '\n' + gap_ix + '\n' + subsuf)
f.close()

# preliminary CASA imaging to set up for image loop
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < prepimaging_casa.py')

# make a custom mask for the kink candidate of interest
custom_mask(target, 0, target+'_gap'+gap_ix+'.'+subsuf,
            buffer_factor=1.5, feature='kink')   # annulus rkink[0] +/- 1.5*wkink[0]

# make a script to convert custom mask into CASA format
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < mask_to_casa.py')

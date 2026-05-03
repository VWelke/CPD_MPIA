import os, sys, time
import numpy as np

import sys, os
sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/')

from custom_mask import custom_mask

sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/')
import diskdictionaryr0_5 as disk

target, gap_ix, subsuf = 'HD_143006', '1', '0'

f = open('whichdisk.txt', 'w')
f.write(target + '\n' + gap_ix + '\n' + subsuf)
f.close()

os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < prepimaging_casa.py')

custom_mask(target, int(gap_ix), target+'_gap'+gap_ix+'.'+subsuf, buffer_factor=1.5)

os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < mask_to_casa.py')

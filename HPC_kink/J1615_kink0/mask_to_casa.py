import os, sys
import numpy as np

import sys, os
sys.path.append(os.getcwd())

target, kink_ix, subsuf = np.loadtxt('whichdisk.txt', dtype=str)

importfits(target+'_kink'+kink_ix+'.'+subsuf+'.custom_mask.fits',
           target+'_kink'+kink_ix+'.'+subsuf+'.custom.mask', overwrite=True)

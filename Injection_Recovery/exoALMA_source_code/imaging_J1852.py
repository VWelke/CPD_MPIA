"""
J1852
exoALMA Fiducial Imaging Script

Note to run this you must be in the exoALMA virtual environment, which at NRAO is:
> source /lustre/cv/projects/exoALMA/casa_modular_6.2.1/bin/activate

"""

import os
import sys
import numpy as np

# Define the path to the imaging_scripts directory you cloned from GitHub.
sys.path.append(os.path.expanduser('~/software/exoALMA/imaging_scripts/'))

# Import the necessary exoALMA functions
from imaging_script_release import iterative_tclean, get_nchan_vstart

ms_dir = '/home/phyji/SLOW_SAFE/exoALMA/data/ms_tables_v4/' # Path to the measurement sets

vlsr = 5.461  # Source vlsr [km/s]

vrng = 15.0   # Velocity range to image [km/s]

# Begin the imaging...

# 12CO

ms = 'J1852-3700_SBLB_no_ave_selfcal_time_ave_12CO.ms'
restfreq = '345.7959899GHz'

for continuum in ['', '.contsub']:	
    for width in [28.0, 100.0]:		# [m/s]
        vstart, nchan = get_nchan_vstart(vlsr,vrng,width)
        for beam in [0.15, 0.30]:	# [arcsec]
                iterative_tclean(ms_path=ms_dir + ms + continuum,
                             restfreq=restfreq,
                             spw='',
                             start='{:.3f}km/s'.format(vstart),
                             nchan=nchan,
                             width='{:.0f}m/s'.format(width),
                             save_path='J1852_12CO' + continuum,
                             beam=beam,
                             threshold=[4,3])

# 13CO

ms = 'J1852-3700_SBLB_no_ave_selfcal_time_ave_13CO.ms'
restfreq = '330.58796530GHz'

for continuum in ['', '.contsub']:
    for width in [100.0, 200.0]:	
        vstart, nchan = get_nchan_vstart(vlsr,vrng,width)
        for beam in [0.15, 0.30]:	
            iterative_tclean(ms_path=ms_dir + ms + continuum,
                             restfreq=restfreq,
                             spw='',
                             start='{:.3f}km/s'.format(vstart),
                             nchan=nchan,
                             width='{:.0f}m/s'.format(width),
                             save_path='J1852_13CO' + continuum,
                             beam=beam,
                             threshold=[4,3])

# CS

ms = 'J1852-3700_SBLB_no_ave_selfcal_time_ave_CS.ms'
restfreq = '342.88285030GHz'

for continuum in ['', '.contsub']:
    for width in [100.0, 200.0]:	
        vstart, nchan = get_nchan_vstart(vlsr,vrng,width)
        for beam in [0.15, 0.30]:	
            iterative_tclean(ms_path=ms_dir + ms + continuum,
                             restfreq=restfreq,
                             spw='',
                             start='{:.3f}km/s'.format(vstart),
                             nchan=nchan,
                             width='{:.0f}m/s'.format(width),
                             save_path='J1852_CS' + continuum,
                             beam=beam,
                             threshold=[4,3])





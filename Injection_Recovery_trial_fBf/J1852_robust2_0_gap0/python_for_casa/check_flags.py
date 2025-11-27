# Add this to a new cell in your notebook or run in CASA
from casatools import table

def check_flags(ms_file):
    tb = table()
    tb.open(ms_file)
    
    # Get FLAG column (boolean array)
    flags = tb.getcol('FLAG')
    tb.close()
    
    total_vis = flags.size
    flagged_vis = np.sum(flags)
    flag_fraction = flagged_vis / total_vis * 100
    
    print(f"File: {ms_file}")
    print(f"Total visibilities: {total_vis:,}")
    print(f"Flagged visibilities: {flagged_vis:,}")
    print(f"Flag fraction: {flag_fraction:.2f}%")
    print("-" * 50)
    
    return flag_fraction > 0

# Check a few of your measurement sets
import numpy as np
ms_dir = "/mnt/d/exoALMA_disk_data/measurement_set/"

test_targets = ['V4046_Sgr', 'AA_Tau', 'CQ_Tau', 'DM_Tau', 'HD_14300']  # Sample a few
for target in test_targets:
    ms_file = ms_dir + target + "_time_ave_continuum_spavg.ms"
    has_flags = check_flags(ms_file)
    if not has_flags:
        print(f"✅ {target}: No flagged data - keepflags setting doesn't matter")
    else:
        print(f"🚩 {target}: Has flagged data - keepflags=False will make a difference")
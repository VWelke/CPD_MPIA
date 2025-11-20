"""
Convert spectrally-averaged MS to .npz format using uvplot (corrects for the u,v unit)
"""
import numpy as np
import os

from casatools import table
tb = table()

import subprocess
import sys
import numpy as np
import os

# Ensure uvplot is available
try:
    from uvplot import export_uvtable
except ImportError:
    print("Installing uvplot...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'uvplot'])
    
    # Force refresh the module path
    import site
    if site.getusersitepackages() not in sys.path:
        sys.path.append(site.getusersitepackages())
    
    from uvplot import export_uvtable
    print("uvplot installed and imported successfully!")

# Directories  
ms_dir = "/mnt/d/exoALMA_disk_data/measurement_set_spavg/"
vis_txt_dir = "/mnt/d/exoALMA_disk_data/measurement_set_spavg/vis_txt/"
output_dir = "/mnt/d/exoALMA_disk_data//measurement_set_spavg/npz/"
os.makedirs(output_dir, exist_ok=True)

targets = ['AA_Tau', 'CQ_Tau', 'DM_Tau', 'HD_14300', 'HD_34282', 'HD_135344B', 
           'J1604', 'J1615', 'J1852-3700', 'LkCa_15', 'MWC_758', 'PDS_66', 
           'RXJ1842-3532', 'SY_Cha', 'V4046_Sgr']

for target in targets:
    ms_file = ms_dir + target + "_time_ave_continuum_spavg.ms"
    temp_txt = vis_txt_dir + f"{target}_time_ave_continuum_spavg_vis.txt"
    npz_file = output_dir + target + "_time_ave_continuum_spavg_lambda.vis.npz"
    
    print(f"Processing {target}...")
        
    # Load and convert to NPZ
    u, v, vis_real, vis_imag, weight = np.loadtxt(temp_txt, unpack=True)

    # compute spatial frequencies in lambda units
     # Read wavelength from header line
    with open(temp_txt) as f:
        for line in f:
            if "wavelength[m]" in line:
                wle = float(line.split('=')[1])
                break
            
    u_lambda = u / wle
    v_lambda = v / wle


    vis = vis_real + 1j * vis_imag
        
    # Save as NPZ
    np.savez(npz_file, u=u_lambda, v=v_lambda, Vis=vis, Wgt=weight)
        

        

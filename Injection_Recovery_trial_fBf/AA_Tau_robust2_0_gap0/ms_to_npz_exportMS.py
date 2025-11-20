
"""
Convert spectrally-averaged MS to .npz format using export_ms from exoALMA 
"""
import numpy as np
import os

from casatools import table
tb = table()

import subprocess
import sys
import numpy as np
import os

# Directories  
ms_dir = "/mnt/d/exoALMA_disk_data/measurement_set_spavg/"
output_dir = "/mnt/d/exoALMA_disk_data//measurement_set_spavg/npz/"
os.makedirs(output_dir, exist_ok=True)

targets = ['AA_Tau', 'CQ_Tau', 'DM_Tau', 'HD_14300', 'HD_34282', 'HD_135344B', 
           'J1604', 'J1615', 'J1852-3700', 'LkCa_15', 'MWC_758', 'PDS_66', 
           'RXJ1842-3532', 'SY_Cha', 'V4046_Sgr']

for target in targets:
    MS_filename = ms_dir + target + "_time_ave_continuum_spavg_copy.ms"
    temp_txt = f"{target}_time_ave_continuum_spavg_vis.txt"
    npz_file = output_dir + target + "_time_ave_continuum_spavg_exportMS.vis.npz"
    
    print(f"Processing {target}...")




    # get the data tables
    tb.open(MS_filename)  
    data   = np.squeeze(tb.getcol("DATA")) 
    flag   = np.squeeze(tb.getcol("FLAG"))
    uvw    = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid  = tb.getcol("DATA_DESC_ID")
    tb.close()

    # get frequency information
    tb.open(MS_filename+'/SPECTRAL_WINDOW')
    freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
    tb.close()

    # get rid of any flagged columns
    good   = np.squeeze(np.any(flag, axis=0)==False)
    data   = data[:,good]
    weight = weight[:,good]
    uvw    = uvw[:,good]
    spwid = spwid[good]

    if (freqlist.size==1): freqlist = np.asarray([freqlist])

    # compute spatial frequencies in lambda units
    get_freq = lambda ispw: freqlist[ispw]
    freqs = get_freq(spwid) #get spectral frequency corresponding to each datapoint
    u = uvw[0,:] * freqs / 2.9979e8
    v = uvw[1,:] * freqs / 2.9979e8

    #average the polarizations
    Re  = np.sum(data.real*weight, axis=0) / np.sum(weight, axis=0)
    Im  = np.sum(data.imag*weight, axis=0) / np.sum(weight, axis=0)
    Vis = Re + 1j*Im
    Wgt = np.sum(weight, axis=0)

    #output to npz file and delete intermediate measurement set

    np.savez(npz_file, u=u, v=v, Vis=Vis, Wgt=Wgt)
    print("#Measurement set exported to %s" % (npz_file,))

        # Save as ASCII .txt file
    txt_file = output_dir + target + "_time_ave_continuum_spavg_exportMS.vis.txt"
    np.savetxt(
        txt_file,
        np.column_stack([u, v, Vis.real, Vis.imag, Wgt]),
        fmt="%.8e",
        header="u[lambda] v[lambda] Re(V)[Jy] Im(V)[Jy] Weight[Jy^-2]"
    )
    print("#Measurement set also exported to %s" % (txt_file,))
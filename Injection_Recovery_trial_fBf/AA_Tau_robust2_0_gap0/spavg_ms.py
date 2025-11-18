# This file will be run by CASA to split the exoALMA measurement set and spectrally average it 

ms_dir = "d:/exoALMA_disk_data/measurement_set/"

from casatools import table
tb = table()
from casatasks import split

targets = ['V4046_Sgr', 'AA_Tau', 'CQ_Tau', 'DM_Tau', 'HD_14300', 'HD_34282', 'HD_135344B', 'J1604', 'J1615', 'J1852-3700', 'LkCa_15', 'MWC_758', 'PDS_66', 'RXJ1842-3532', 'SY_Cha']
for target in targets:
    input_ms = ms_dir + target + "_time_ave_continuum.ms"
    output_ms = ms_dir + target + "_time_ave_continuum_spavg.ms"
    
    tb.open(input_ms+'.ms/SPECTRAL_WINDOW')
    num_chan = tb.getcol('NUM_CHAN').tolist()
    tb.close()
    
    print(f"Processing {target}...")
    split(vis=input_ms, width=num_chan, datacolumn='data', outputvis=output_ms)
    print(f"Completed: {output_ms}") 
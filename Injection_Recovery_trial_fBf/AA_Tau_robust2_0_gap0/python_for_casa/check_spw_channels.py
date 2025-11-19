from casatools import table
import numpy as np

def check_spw_channels(ms_dir, targets):
    """Check if all SPWs have 1 channel (spectrally averaged)"""
    tb = table()
    
    print("Checking spectrally-averaged measurement sets:")
    print("=" * 60)
    
    for target in targets:
        ms_file = ms_dir + target + "_time_ave_continuum_spavg.ms"
        
        try:
            tb.open(ms_file + "/SPECTRAL_WINDOW")
            nchan = tb.getcol("NUM_CHAN")
            tb.close()
            
            # Check if all channels = 1
            all_single_channel = np.all(nchan == 1)
            max_chan = np.max(nchan)
            num_spw = len(nchan)
            
            status = "✅ GOOD" if all_single_channel else f"❌ ISSUE (max: {max_chan})"
            
            print(f"{target:12s} | {num_spw:2d} SPWs | {status}")
            
        except Exception as e:
            print(f"{target:12s} | ERROR: {str(e)[:30]}...")
    
    print("=" * 60)

# Check your processed measurement sets
ms_dir = "/mnt/d/exoALMA_disk_data/measurement_set/"
targets = ['AA_Tau', 'CQ_Tau', 'HD_34282', 'HD_135344B', 'J1604', 
           'J1615', 'J1852-3700', 'LkCa_15', 'MWC_758', 'PDS_66', 
           'RXJ1842-3532', 'SY_Cha', 'V4046_Sgr', 'DM_Tau', 'HD_14300']

check_spw_channels(ms_dir, targets)
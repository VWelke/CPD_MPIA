# Save, for one target, what prove_single_wavelength.ipynb needs from the MS (needs casatools):
#   u_m      -- u in metres (UVW column) for the rows the npz holds (no flagged polarisation, MS order, as ImportMS)
#   spw_row  -- spectral window of each of those rows
#   spwf     -- frequency of each spectral window [Hz]
# Run with CASA's python, e.g. from WSL:
#   /usr/local/bin/CASA/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/bin/python3 ms_rowinfo.py LkCa_15
import sys
import numpy as np
from casatools import table
tb = table()

t = sys.argv[1]
ms = '/mnt/d/exoALMA_disk_data/measurement_set_spavg/' + t + '_time_ave_continuum_spavg.ms'
tb.open(ms + '/SPECTRAL_WINDOW')
spwf = np.array([np.mean(tb.getcell('CHAN_FREQ', i)) for i in range(tb.nrows())])
tb.close()
tb.open(ms + '/DATA_DESCRIPTION')
dd2spw = tb.getcol('SPECTRAL_WINDOW_ID')
tb.close()
tb.open(ms)
uvw = tb.getcol('UVW')
dd = tb.getcol('DATA_DESC_ID')
flag = tb.getcol('FLAG')
tb.close()
unflagged = np.squeeze(np.any(flag, axis=0) == False)
np.savez(t + '_ms_rowinfo.npz', u_m=uvw[0][unflagged], spw_row=dd2spw[dd][unflagged], spwf=spwf)
print('saved', t + '_ms_rowinfo.npz', unflagged.sum(), 'rows')

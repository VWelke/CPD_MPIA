# Proof that the *_lambda.vis.npz files convert u, v with ONE wavelength, although the MS has many.
#
# The MS stores u, v in METRES (UVW column). Converting a row to wavelengths needs THAT row's frequency:
#     u_lambda = u_m * nu_row / c
# so for every row, u_npz / u_m should equal nu_row / c, which differs between spectral windows.
# If instead u_npz / u_m is the SAME number for every row, a single wavelength was used for all of them.
#
# Run with CASA's python (needs casatools), e.g. from WSL:
#   /usr/local/bin/CASA/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/bin/python3 prove_single_wavelength.py
import numpy as np
from casatools import table
tb = table()

c = 2.99792458e8
t = 'LkCa_15'
ms = '/mnt/d/exoALMA_disk_data/measurement_set_spavg/' + t + '_time_ave_continuum_spavg.ms'
npz = '/mnt/d/exoALMA_disk_data/measurement_set_spavg/npz/' + t + '_time_ave_continuum_spavg_lambda.vis.npz'

# 1. what the MS says: one frequency per spectral window
tb.open(ms + '/SPECTRAL_WINDOW')
spwf = np.array([np.mean(tb.getcell('CHAN_FREQ', i)) for i in range(tb.nrows())])
tb.close()
tb.open(ms + '/DATA_DESCRIPTION')
dd2spw = tb.getcol('SPECTRAL_WINDOW_ID')
tb.close()
tb.open(ms)
uvw = tb.getcol('UVW')            # metres, shape (3, nrow)
dd = tb.getcol('DATA_DESC_ID')
flag = tb.getcol('FLAG')
tb.close()
unflagged = np.squeeze(np.any(flag, axis=0) == False)   # the rows the npz holds (same selection as ImportMS)
spw_row = dd2spw[dd][unflagged]
nu_row = spwf[spw_row]
u_m = uvw[0][unflagged]

print('MS spectral windows:', len(spwf))
for s in np.unique(spw_row):
    print('  spw %2d  nu = %.4f GHz  lambda = %.5f mm  rows = %d' % (s, spwf[s] / 1e9, c / spwf[s] * 1e3, np.sum(spw_row == s)))

# 2. what the npz used: u_npz / u_m for every row
u_npz = np.load(npz)['u']
good = np.abs(u_m) > 1                              # avoid dividing by ~0
ratio = u_npz[good] / u_m[good]                     # = 1 / lambda_used  [1/m]
lam_used = 1 / ratio
print('\n_lambda npz: implied wavelength per row  min %.6f mm  max %.6f mm  (spread %.1e)'
      % (lam_used.min() * 1e3, lam_used.max() * 1e3, lam_used.std() / lam_used.mean()))
print('             -> one wavelength for ALL rows: %.6f mm = %.4f GHz' % (np.median(lam_used) * 1e3, c / np.median(lam_used) / 1e9))

# 3. what it should have been, and the resulting error per spectral window
print('\nper spw: correct u_lambda / npz u_lambda = nu_row / nu_used  (an injection lands at r * nu_used / nu_row)')
nu_used = c / np.median(lam_used)
for s in np.unique(spw_row):
    print('  spw %2d  nu = %.3f GHz  u scale error = %+.2f %%   -> position error = %+.2f %%'
          % (s, spwf[s] / 1e9, 100 * (spwf[s] / nu_used - 1), 100 * (nu_used / spwf[s] - 1)))
w = np.sum(np.squeeze(np.load(npz)['Wgt']).reshape(len(u_npz), -1), axis=1) if np.load(npz)['Wgt'].ndim > 1 else np.load(npz)['Wgt']
print('weight-averaged position error = %+.2f %%' % (100 * (np.sum(w * nu_used / nu_row) / np.sum(w) - 1)))

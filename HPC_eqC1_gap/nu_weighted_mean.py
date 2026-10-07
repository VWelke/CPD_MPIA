# Data-weighted mean frequency of each spavg MS, and the uv scale factor this implies
# for the *_lambda.vis.npz files (u, v divided by ONE wavelength, uvplot's 0.8877 mm = 337.7 GHz).
# An injection at radius r then lands at r * nu_ref / nu_row, so scale = nu_ref / nu_weighted_mean.
# Run with CASA's python (needs casatools), e.g. from WSL:
#   /usr/local/bin/CASA/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/bin/python3 nu_weighted_mean.py
import numpy as np
from casatools import table
tb = table()

ms_dir = '/mnt/d/exoALMA_disk_data/measurement_set_spavg/'
txt_dir = ms_dir + 'vis_txt/'
targets = ['AA_Tau', 'CQ_Tau', 'DM_Tau', 'HD_135344B', 'HD_143006', 'HD_34282', 'J1604', 'J1615',
           'J1852', 'LkCa_15', 'MWC_758', 'PDS_66', 'RXJ1842-3532', 'SY_Cha', 'V4046_Sgr']

rows = []
for t in targets:
    ms = ms_dir + t + '_time_ave_continuum_spavg.ms'
    tb.open(ms + '/SPECTRAL_WINDOW')
    spwf = np.array([np.mean(tb.getcell('CHAN_FREQ', i)) for i in range(tb.nrows())])
    tb.close()
    tb.open(ms + '/DATA_DESCRIPTION')
    dd2spw = tb.getcol('SPECTRAL_WINDOW_ID')
    tb.close()
    tb.open(ms)
    dd = tb.getcol('DATA_DESC_ID')
    weights = tb.getcol('WEIGHT')
    flag = tb.getcol('FLAG')
    tb.close()

    nu_row = spwf[dd2spw[dd]]
    w_row = weights.sum(axis=0) * ~np.all(flag, axis=(0, 1))   # flagged rows get zero weight
    nu_w = np.sum(nu_row * w_row) / np.sum(w_row)

    # the single wavelength the _lambda npz was made with (uvplot header)
    txt = txt_dir + {'HD_143006': 'HD_14300', 'J1852': 'J1852-3700'}.get(t, t) + '_time_ave_continuum_spavg_vis.txt'
    wle = float([l for l in open(txt) if 'wavelength[m]' in l][0].split('=')[1])
    nu_ref = 2.99792458e8 / wle
    rows.append((t, nu_ref / 1e9, nu_w / 1e9, nu_row.min() / 1e9, nu_row.max() / 1e9, nu_ref / nu_w))
    print('%-14s nu_ref=%.4f  nu_wmean=%.4f  range=%.3f-%.3f GHz  scale=%.5f' % rows[-1])

with open('nu_weighted_mean.txt', 'w') as f:
    f.write('# target  nu_ref_GHz  nu_wmean_GHz  nu_min_GHz  nu_max_GHz  scale=nu_ref/nu_wmean\n')
    for r in rows:
        f.write('%-14s %.5f %.5f %.4f %.4f %.6f\n' % r)

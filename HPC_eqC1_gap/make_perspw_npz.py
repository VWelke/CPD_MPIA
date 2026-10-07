# Make {target}_time_ave_continuum_spavg_lambda_perspw.vis.npz from the single-wavelength
# {target}_..._lambda.vis.npz: u, v rescaled row by row to that row's own spectral-window
# frequency (u_perspw = u_lambda * nu_row / nu_ref). Vis and Wgt are unchanged.
# Rows are the MS rows with no flagged polarisation, in MS order -- the same rows ImportMS writes back.
# First rebuilds an existing _perspw file as a check, then writes the requested targets.
# Run with CASA's python (needs casatools), e.g. from WSL:
#   /usr/local/bin/CASA/casa-6.6.1-17-pipeline-2024.1.0.8/lib/py/bin/python3 make_perspw_npz.py
import numpy as np
from casatools import table
tb = table()

ms_dir = '/mnt/d/exoALMA_disk_data/measurement_set_spavg/'
npz_dir = ms_dir + 'npz/'
check = 'LkCa_15'                 # has an existing _perspw file to compare against
targets = ['DM_Tau']              # files to write


def perspw(t):
    ms = ms_dir + t + '_time_ave_continuum_spavg.ms'
    tb.open(ms + '/SPECTRAL_WINDOW')
    spwf = np.array([np.mean(tb.getcell('CHAN_FREQ', i)) for i in range(tb.nrows())])
    tb.close()
    tb.open(ms + '/DATA_DESCRIPTION')
    dd2spw = tb.getcol('SPECTRAL_WINDOW_ID')
    tb.close()
    tb.open(ms)
    dd = tb.getcol('DATA_DESC_ID')
    flag = tb.getcol('FLAG')
    tb.close()
    unflagged = np.squeeze(np.any(flag, axis=0) == False)
    nu_row = spwf[dd2spw[dd]][unflagged]

    txt = ms_dir + 'vis_txt/' + t + '_time_ave_continuum_spavg_vis.txt'
    nu_ref = 2.99792458e8 / float([l for l in open(txt) if 'wavelength[m]' in l][0].split('=')[1])
    d = np.load(npz_dir + t + '_time_ave_continuum_spavg_lambda.vis.npz')
    print(t, ': npz rows', len(d['u']), ' unflagged MS rows', len(nu_row), ' nu_ref %.4f GHz' % (nu_ref / 1e9))
    return d['u'] * nu_row / nu_ref, d['v'] * nu_row / nu_ref, d['Vis'], d['Wgt']


u, v, Vis, Wgt = perspw(check)
e = np.load(npz_dir + check + '_time_ave_continuum_spavg_lambda_perspw.vis.npz')
print('check vs existing', check, 'perspw: max |du/u| = %.2e, max |dv/v| = %.2e' %
      (np.max(np.abs(u - e['u']) / np.abs(e['u']).clip(1)), np.max(np.abs(v - e['v']) / np.abs(e['v']).clip(1))))

for t in targets:
    u, v, Vis, Wgt = perspw(t)
    np.savez(npz_dir + t + '_time_ave_continuum_spavg_lambda_perspw.vis.npz', u=u, v=v, Vis=Vis, Wgt=Wgt)
    print('wrote', t)

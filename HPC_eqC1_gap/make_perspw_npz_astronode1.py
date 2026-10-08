# astronode1 version of make_perspw_npz.py (no vis_txt files needed), for every disk with an spavg MS.
# u, v taken straight from the MS UVW column (metres) and converted with each row's own
# spectral-window frequency: u_perspw = u_m * nu_row / c  (== u_lambda * nu_row / nu_ref).
# Vis and Wgt copied from the existing _lambda.vis.npz. Rows = unflagged MS rows, in MS order.
# Disks that already have a _perspw file are only compared against it, not overwritten.
# Run through casa with the same config the imageloop uses (bare python3 fails on ~/.casa/data):
#   /nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin/casa --pipeline --configfile /nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py --nogui --nologger --nologfile -c make_perspw_npz_astronode1.py
import os, glob
import numpy as np
from casatools import table
tb = table()

c = 2.99792458e8
ms_dir = '/nexus/posix0/MIA-astro-env/myben/vawelke/exoALMA_disk_data/measurement_set_spavg/'
npz_dir = ms_dir + 'npz/'
targets = sorted(os.path.basename(f).replace('_time_ave_continuum_spavg_lambda.vis.npz', '')
                 for f in glob.glob(npz_dir + '*_time_ave_continuum_spavg_lambda.vis.npz'))
ms_name = {'HD_14300': 'HD_143006'}   # npz name -> MS name where they differ
print('targets:', targets)

for t in targets:
    ms = ms_dir + ms_name.get(t, t) + '_time_ave_continuum_spavg.ms'
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
    nu_row = spwf[dd2spw[dd]][unflagged]
    um, vm = uvw[0][unflagged], uvw[1][unflagged]
    u, v = um * nu_row / c, vm * nu_row / c

    d = np.load(npz_dir + t + '_time_ave_continuum_spavg_lambda.vis.npz')
    nu_ref = c * d['u'] / um
    print('\n%s : npz rows %d  unflagged MS rows %d' % (t, len(d['u']), len(nu_row)))
    print('  spw freqs [GHz]:', np.round(np.unique(nu_row) / 1e9, 4))
    print('  single nu_ref used by _lambda npz [GHz]: %.4f (spread %.1e)' %
          (np.nanmedian(nu_ref) / 1e9, np.nanstd(nu_ref) / np.nanmedian(nu_ref)))

    out = npz_dir + t + '_time_ave_continuum_spavg_lambda_perspw.vis.npz'
    if os.path.exists(out):
        e = np.load(out)
        print('  existing perspw kept; max |du/u| = %.2e, max |dv/v| = %.2e' %
              (np.max(np.abs(u - e['u']) / np.abs(e['u']).clip(1)), np.max(np.abs(v - e['v']) / np.abs(e['v']).clip(1))))
    else:
        np.savez(out, u=u, v=v, Vis=d['Vis'], Wgt=d['Wgt'])
        print('  wrote', out)

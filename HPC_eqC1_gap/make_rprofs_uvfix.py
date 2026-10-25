# Run on Node-01, from the eqC1_gap folder, with the frank_env python:
#   cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap && python make_rprofs_uvfix.py
# On Spartan (DM_Tau_gap1 / LkCa_15_gap0 uvfix):
#   cd /data/gpfs/projects/punim3000/MPIA/HPC_eqC1_gap_spartan && python make_rprofs_uvfix.py /data/gpfs/projects/punim3000/MPIA
#
# For every base uvfix gap/cavity folder (gap/*_gapN_uvfix, inner_cavity/*_gapN_uvfix):
#   1. combine the base run + _rep1-9 into recoveries/{name}_recoveries.combined.txt
#      (a run counts once its pipeline has finished, i.e. recoveries/*.resid.fits exists)
#   2. compute recovery fractions exactly as the gap assess_recovery.ipynb does
#      (criteria A and B, 2-pixel floor, annulus false-positive estimate) and write
#      recoveries/{name}_rprofs.combined.txt  (Fcpd, frec_A, efrec_A, false_A, frec_B, efrec_B, false_B)
# This needs the _lambda npz for the maximum baseline, which only exists on the cluster.

import glob, os, sys
import numpy as np
from astropy.io import fits
from scipy.special import erfc

# cluster base path: astronode1 by default; on Spartan pass /data/gpfs/projects/punim3000/MPIA
# (same Source_codes/ and exoALMA_disk_data/ layout underneath)
N = sys.argv[1] if len(sys.argv) > 1 else '/nexus/posix0/MIA-astro-env/myben/vawelke'
sys.path.append(N + '/Source_codes/diskdictionary_r90')
import diskdictionary_eqC1 as disk

npix_tol, ast_tol = 2, 2


def f50(F, f):
    if f.max() < 0.5:
        return np.nan
    i = np.argmax(f >= 0.5)
    if i == 0:
        return F[0]
    return F[i - 1] + (0.5 - f[i - 1]) * (F[i] - F[i - 1]) / (f[i] - f[i - 1])


for base in sorted(glob.glob('gap/*_gap[0-9]_uvfix') + glob.glob('inner_cavity/*_gap[0-9]_uvfix')):
    name = os.path.basename(base)[:-len('_uvfix')]              # e.g. AA_Tau_gap0
    target, gap_ix = name.rsplit('_gap', 1)
    gap_ix = int(gap_ix)

    runs = [d for d in [base] + sorted(glob.glob(base + '_rep[1-9]')) if glob.glob(d + '/recoveries/*.resid.fits')]
    if base not in runs:
        print(f'{name:22s} base run not finished -- skipped')
        continue

    # 1. combine
    rec_files = [f'{d}/recoveries/{name}_recoveries.0.txt' for d in runs]
    open(f'{base}/recoveries/{name}_recoveries.combined.txt', 'w').write(''.join(open(f).read() for f in rec_files))
    Fi, Fr, mdl, ri, rr, azi, azr, xr, yr, mu, rms = np.loadtxt(f'{base}/recoveries/{name}_recoveries.combined.txt').T

    # 2. injected position (disk-polar -> sky) and distance to the recovered peak
    azir = np.radians(azi)
    inclr, PAr = np.radians(disk.disk[target]['incl']), np.radians(disk.disk[target]['PA'])
    xi = ri * np.sin(azir) * np.sin(PAr) + ri * np.cos(azir) * np.cos(inclr) * np.cos(PAr)
    yi = ri * np.sin(azir) * np.cos(PAr) - ri * np.cos(azir) * np.cos(inclr) * np.sin(PAr)
    d_ir = np.sqrt((xi - xr)**2 + (yi - yr)**2)

    hd = fits.getheader(sorted(glob.glob(base + '/recoveries/*.resid.fits'))[0])
    beam_fwhm = np.sqrt(3600**2 * hd['BMAJ'] * hd['BMIN'])
    beam_area = 3600**2 * np.pi * hd['BMAJ'] * hd['BMIN'] / (4 * np.log(2))
    pix_size = np.abs(3600 * hd['CDELT2'])

    # single-wavelength npz: wave(CRVAL3) * u gives the true baseline (same as assess_recovery.ipynb)
    vdat = np.load(N + '/exoALMA_disk_data/measurement_set_spavg/npz/' + target + '_time_ave_continuum_spavg_lambda.vis.npz')
    freq, wave = hd['CRVAL3'], 2.99792e5 / hd['CRVAL3']
    Bmax_km = (wave * np.sqrt(vdat['u']**2 + vdat['v']**2)).max()

    sigma_A = (4 / np.pi)**0.25 * beam_fwhm * rms / (np.sqrt(8 * np.log(2)) * Fr)
    sigma_B = 70 / ((freq / 1e9) * Bmax_km * Fr / rms)
    dlim_A = np.maximum(ast_tol * sigma_A, npix_tol * pix_size)
    dlim_B = np.maximum(ast_tol * sigma_B, npix_tol * pix_size)
    rec_A, rec_B = d_ir <= dlim_A, d_ir <= dlim_B

    # recovery fractions, Poisson errors and crude false-positive fractions per flux bin
    rgap, wgap = disk.disk[target]['rgap'][gap_ix], disk.disk[target]['wgap'][gap_ix]
    Fcpd = np.unique(Fi)
    rows, nper = [], []
    for F in Fcpd:
        m = Fi == F
        n = m.sum()
        nper.append(n)
        pf_spike = 0.5 * erfc(F / (2 * np.average(rms[m])))
        pfA = max(np.pi * np.average(dlim_A[m])**2 / beam_area, 1) / (4 * np.pi * rgap * wgap / beam_area)
        pfB = max(np.pi * np.average(dlim_B[m])**2 / beam_area, 1) / (4 * np.pi * rgap * wgap / beam_area)
        rows.append([F, (rec_A & m).sum() / n, np.sqrt((rec_A & m).sum()) / n, pfA * pf_spike,
                        (rec_B & m).sum() / n, np.sqrt((rec_B & m).sum()) / n, pfB * pf_spike])
    rows = np.array(rows)
    np.savetxt(f'{base}/recoveries/{name}_rprofs.combined.txt', rows, fmt='%.4f')

    print(f'{name:22s} {len(runs):2d} runs   mocks/bin {min(nper)}-{max(nper)}   F50_B = {f50(Fcpd, rows[:, 4]):6.1f} uJy')

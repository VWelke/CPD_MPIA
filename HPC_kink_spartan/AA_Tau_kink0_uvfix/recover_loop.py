import os, sys, time
import numpy as np
from astropy.io import fits

sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/')
sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/diskdictionary_r90/')
import diskdictionary_eqC1 as disk

# target disk / kink candidate; iteration
target = 'AA_Tau'
gap    = 2          # file-name label only, the old rgap index (matches injectloop.py)
ix     = '0'

# fixed candidate position -- MUST match injectloop.py's CANDIDATE_R/CANDIDATE_AZ
CANDIDATE_R  = 0.5926   # r_planet = 80 au (Pinte+2025 Table 2) / 135 pc
CANDIDATE_AZ = -142.37   # Figure 5 blue dot, recover_loop az convention (verify_figure5_blue_dot.ipynb)

# load the injection file data
inj_file = 'injections/'+target+'_gap'+str(gap)+'_mpars.'+ix+'.txt'
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T
Fcpd, mdl, rcpd, azcpd  = np.loadtxt(inj_file).T

# bookkeeping
recov_file = 'recoveries/'+target+'_gap'+str(gap)+'_recoveries.'+ix+'.txt'
os.system('rm -rf ' + recov_file)

# loop through injections
for i in range(len(Fstr)):

    im_file = target + '_gap' + str(gap) + '.F' + Fstr[i] + 'uJy_' + mstr[i]
    hdu = fits.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'resid_images', im_file + '.resid.fits'))
    img = 1e6 * np.squeeze(hdu[0].data)    # microJy/beam
    hd  = hdu[0].header
    hdu.close()

    # Cartesian sky-plane coordinate system
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    xs, ys = np.meshgrid(RAo  - disk.disk[target]['dx'],
                         DECo - disk.disk[target]['dy'])

    # Cartesian disk-plane coordinate system
    inclr = np.radians(disk.disk[target]['incl'])
    PAr   = np.radians(disk.disk[target]['PA'])
    xd = (xs * np.cos(PAr) - ys * np.sin(PAr)) / np.cos(inclr)
    yd = (xs * np.sin(PAr) + ys * np.cos(PAr))

    # Polar disk-plane coordinate system
    rd  = np.sqrt(xd**2 + yd**2)
    azd = np.degrees(np.arctan2(yd, xd))

    # beam FWHM computed per-image from its own header (this script runs
    # after imaging, unlike injectloop.py, so it doesn't need a hardcoded value)
    beam_fwhm = np.sqrt(3600**2 * hd['BMAJ'] * hd['BMIN'])

    # search box: 1 beam LARGER than the injection zone on each side --
    # i.e. +/-2 beams radially, and the 15-deg wedge padded by the beam's
    # arc-length in degrees at this radius on each side (same "injection
    # zone + 1 beam" padding convention as the HD_135344B point candidates)
    search_r_halfwidth  = 2 * beam_fwhm
    search_az_halfwidth = 7.5 + np.degrees(beam_fwhm / CANDIDATE_R)

    bndi = (rd >= (CANDIDATE_R - search_r_halfwidth))
    bndo = (rd <= (CANDIDATE_R + search_r_halfwidth))
    daz = ((azd - CANDIDATE_AZ + 180) % 360) - 180   # wrapped azimuthal offset
    bnda = (np.abs(daz) <= search_az_halfwidth)
    mask = bndi & bndo & bnda

    g_img, g_xs, g_ys = img[mask], xs[mask], ys[mask]
    g_rd, g_azd = rd[mask], azd[mask]

    # Locate and measure the peak within the search box
    peak = (g_img == g_img.max())
    pk_xs, pk_ys = g_xs[peak][0], g_ys[peak][0]
    pk_r,  pk_az = g_rd[peak][0], g_azd[peak][0]
    pk_SB = g_img.max()

    # local background/noise: same radial band, but the FULL azimuthal
    # range (minus a small exclusion right around the peak), so the noise
    # estimate isn't limited to the tiny search box itself
    dist_peak = np.sqrt((xs - pk_xs)**2 + (ys - pk_ys)**2)
    bg_mask = bndi & bndo & (dist_peak > (5 * hd['CDELT2'] * 3600))
    emean, estd = np.mean(img[bg_mask]), np.std(img[bg_mask])

    with open(recov_file, 'a') as f:
        f.write('%.0f  %.0f  %s  %.4f  %.4f  %.3f  %.3f  %.5f  %.5f  %.0f  %.0f\n' %
                (Fcpd[i], pk_SB, mstr[i], rcpd[i], pk_r, azcpd[i], pk_az,
                 pk_xs, pk_ys, emean, estd))

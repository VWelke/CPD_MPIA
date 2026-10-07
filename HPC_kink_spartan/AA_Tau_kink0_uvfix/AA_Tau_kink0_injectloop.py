import os, sys, time
import numpy as np

sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/')
from inject_CPD import inject_CPD

from frank.geometry import FixedGeometry
from frank.radial_fitters import FrankFitter
from frank.io import save_fit
sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/diskdictionary_r90/')
import diskdictionary_eqC1 as disk


# specify target disk and kink candidate
target  = 'AA_Tau'    # CSD name
gap_ix  = 2              # file-name label only (the old rgap index) --
                           # the clean mask now uses rkink[0]/wkink[0] (diskdictionary_eqC1.py),
                           # NOT for drawing a wide injection zone
subsuf  = '0'         # suffix to attach to records (if partial work)

# fixed kink-candidate position (disk-plane r, az -- deprojected from the
# sky-plane point picked in kink.ipynb using this disk's incl/PA/dx/dy)
CANDIDATE_R  = 0.5926   # r_planet = 80 au (Pinte+2025 Table 2) / 135 pc
CANDIDATE_AZ = -142.37   # Figure 5 blue dot, recover_loop az convention (verify_figure5_blue_dot.ipynb)

# beam FWHM for this disk (robust=-0.5), read from an already-imaged residual
# header -- injectloop.py runs before any imaging exists, so this can't be
# computed on the fly the way recover_loop.py does it
BEAM_FWHM = 0.0632

# injection zone: +/-1 beam radially, and a FIXED 15-deg wedge (+/-7.5 deg)
# centered on the candidate azimuth -- NOT the wide rgap+/-0.5*wgap zone used
# by the gap pipelines, and NOT the beam-arc-derived wedge used for the
# HD_135344B point candidates; this is a kink, so the azimuthal window is
# fixed by request rather than scaled by beam size at this radius
INJ_R_HALFWIDTH  = BEAM_FWHM
INJ_AZ_HALFWIDTH = 7.5   # degrees


# specify mock parameters
F_cpd = np.arange(1*disk.disk[target]['RMS']/1000,
                  15*disk.disk[target]['RMS']/1000,
                  1*disk.disk[target]['RMS']/1000)   # in mJy
n_mocks_per_F = 50              # number of mocks per flux bin


# -------


# fixed geometric parameters of CSD
incl, PA = disk.disk[target]['incl'], disk.disk[target]['PA']
offRA, offDEC = disk.disk[target]['dx'], disk.disk[target]['dy']
geom = FixedGeometry(incl, PA, dRA=offRA, dDec=offDEC)

# frank setup -- use whichever gives the larger Rmax (rout can be unreliable)
rout = disk.disk[target].get('rout', 0)
r90  = disk.disk[target].get('R90',  0)
Rmax = 2 * max(rout, r90)
Ncoll = disk.disk[target]['hyp-Ncoll']
alpha, wsmth = disk.disk[target]['hyp-alpha'], disk.disk[target]['hyp-wsmth']
FF = FrankFitter(Rmax=Rmax, N=Ncoll, geometry=geom, alpha=alpha,
                 weights_smooth=wsmth)

# load the visibility data
dat = np.load('/data/gpfs/projects/punim3000/MPIA/exoALMA_disk_data/measurement_set_spavg/npz/'
              + target + '_time_ave_continuum_spavg_lambda_perspw.vis.npz')
u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']


# loop through mock injection and modeling
os.system('rm -rf '+target+'_gap'+str(gap_ix)+'_mpars.'+subsuf+'.txt')

t0 = time.time()
for i in range(len(F_cpd)):

    # random position within +/-1 beam (radially) and +/-7.5 deg (azimuthally)
    # of the exact kink-candidate position
    r_cpd  = np.random.uniform(CANDIDATE_R - INJ_R_HALFWIDTH,
                               CANDIDATE_R + INJ_R_HALFWIDTH, n_mocks_per_F)
    az_cpd = np.random.uniform(CANDIDATE_AZ - INJ_AZ_HALFWIDTH,
                               CANDIDATE_AZ + INJ_AZ_HALFWIDTH, n_mocks_per_F)

    for j in range(n_mocks_per_F):

        file_suffix = '_F'+str(int(np.round(1e3*F_cpd[i])))+'uJy_'+str(j).zfill(4)

        vis_cpd = inject_CPD((u, v, vis, wgt),
                             (F_cpd[i], r_cpd[j], az_cpd[j]),
                             incl=incl, PA=PA, offRA=offRA, offDEC=offDEC)

        sol = FF.fit(u, v, vis_cpd, wgt)

        save_fit(u, v, vis_cpd, wgt, sol,
                 prefix=target+'_gap'+str(gap_ix)+file_suffix,
                 save_vis_fit=False, save_solution=False)

        os.system('mv '+target+'_gap'+str(gap_ix)+file_suffix
                  +'_frank_uv_resid.npz resid_vis/')
        os.system('mv '+target+'_gap'+str(gap_ix)+file_suffix
                  +'_frank_profile_fit.txt mprofiles/')
        os.system('rm '+target+'_gap'+str(gap_ix)+file_suffix+'_frank*')

        with open(target+'_gap'+str(gap_ix)+'_mpars.'+subsuf+'.txt', 'a') as f:
            f.write('%i    %s    %.4f    %.3f\n' %
                    (int(np.round(1e3*F_cpd[i])), str(j).zfill(4),
                     r_cpd[j], az_cpd[j]))

print(time.time() - t0)

# move mpars file into injections/
os.makedirs("injections", exist_ok=True)
mpars_name = f"{target}_gap{gap_ix}_mpars.{subsuf}.txt"
os.replace(mpars_name, os.path.join("injections", mpars_name))

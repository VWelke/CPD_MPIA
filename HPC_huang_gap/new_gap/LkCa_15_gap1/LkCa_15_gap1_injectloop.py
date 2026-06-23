import os, sys, time
import numpy as np

sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/')
from inject_CPD import inject_CPD

from frank.geometry import FixedGeometry
from frank.radial_fitters import FrankFitter
from frank.io import save_fit
sys.path.append('/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/')
import diskdictionary_newgap as disk


# specify target disk and gap
target = 'LkCa_15'	# CSD name
gap_ix = 1		# which gap CPD is in (based on dict list)
subsuf = '0'		# suffix to attach to records (if partial work)


# specify mock parameters
F_cpd = np.arange(2*disk.disk[target]['RMS']/1000, 15*disk.disk[target]['RMS']/1000, 1*disk.disk[target]['RMS']/1000)        # in mJy
n_mocks_per_F = 50 			    # number of mocks per flux bin


# -------


# fixed geometric parameters of CSD
incl, PA = disk.disk[target]['incl'], disk.disk[target]['PA']
offRA, offDEC = disk.disk[target]['dx'], disk.disk[target]['dy']
geom = FixedGeometry(incl, PA, dRA=offRA, dDec=offDEC)

# frank setup
try:
    Rmax = 2 * disk.disk[target]['rout']
except KeyError:
    Rmax = 2 * disk.disk[target]['R90']
Ncoll = disk.disk[target]['hyp-Ncoll']
alpha, wsmth = disk.disk[target]['hyp-alpha'], disk.disk[target]['hyp-wsmth']
FF = FrankFitter(Rmax=Rmax, N=Ncoll, geometry=geom, alpha=alpha,
                 weights_smooth=wsmth)

# load the visibility data
dat = np.load('/nexus/posix0/MIA-astro-env/myben/vawelke/exoALMA_disk_data/measurement_set_spavg/npz/' + target + '_time_ave_continuum_spavg_lambda.vis.npz')
u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']


# loop through mock injection and modeling
os.system('rm -rf '+target+'_gap'+str(gap_ix)+'_mpars.'+subsuf+'.txt')

t0 = time.time()
for i in range(len(F_cpd)):

    rgap_cen = disk.disk[target]['rgap'][gap_ix]
    gap_span = 0.5 * disk.disk[target]['wgap'][gap_ix]
    r_cpd = np.random.uniform(rgap_cen - gap_span, rgap_cen + gap_span, n_mocks_per_F)
    az_cpd = np.random.randint(-180, 180, n_mocks_per_F)

    for j in range(n_mocks_per_F):

        file_suffix = '_F'+str(int(np.round(1e3*F_cpd[i])))+'uJy_'+str(j).zfill(4)

        vis_cpd = inject_CPD((u, v, vis, wgt), (F_cpd[i], r_cpd[j], az_cpd[j]),
                             incl=incl, PA=PA, offRA=offRA, offDEC=offDEC)

        sol = FF.fit(u, v, vis_cpd, wgt)

        save_fit(u, v, vis_cpd, wgt, sol,
                 prefix=target+'_gap'+str(gap_ix)+file_suffix,
                 save_vis_fit=False, save_solution=False)

        os.system('mv '+target+'_gap'+str(gap_ix)+file_suffix+'_frank_uv_resid.npz resid_vis/')
        os.system('mv '+target+'_gap'+str(gap_ix)+file_suffix+'_frank_profile_fit.txt mprofiles/')
        os.system('rm '+target+'_gap'+str(gap_ix)+file_suffix+'_frank*')

        with open(target+'_gap'+str(gap_ix)+'_mpars.'+subsuf+'.txt', 'a') as f:
            f.write('%i    %s    %.3f    %i\n' % \
                    (int(np.round(1e3*F_cpd[i])), str(j).zfill(4), r_cpd[j], az_cpd[j]))

print(time.time() - t0)

os.makedirs("injections", exist_ok=True)
mpars_name = f"{target}_gap{gap_ix}_mpars.{subsuf}.txt"
os.replace(mpars_name, os.path.join("injections", mpars_name))

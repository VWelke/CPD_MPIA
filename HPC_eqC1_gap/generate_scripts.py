"""
generate_scripts.py
Generates per-disk/gap pipeline folders and scripts under HPC_eqC1_gap/gap/
and HPC_eqC1_gap/inner_cavity/.

Scripts are adapted from new_gap/ but import diskdictionary_eqC1 instead of
diskdictionary_newgap.  Run once locally; then rsync the whole HPC_eqC1_gap/
tree to the cluster.
"""

import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
HPC_BASE    = "/nexus/posix0/MIA-astro-env/myben/vawelke/HPC_eqC1_gap"
SRC_PATH    = "/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes"
DICT_SUBDIR = "diskdictionary_r90"   # place diskdictionary_eqC1.py here on HPC

# (disk_name, gap_ix, subfolder)
TARGETS = [
    # --- ring gaps ---
    ("HD_135344B", 0, "gap"),
    ("J1615",      0, "gap"),
    ("LkCa_15",    0, "gap"),
    ("MWC_758",    1, "gap"),
    ("V4046_Sgr",  1, "gap"),
    ("AA_Tau",     0, "gap"),
    ("AA_Tau",     1, "gap"),   # NOTE: dgap hit upper bound (200), treat with care
    # --- inner cavities ---
    ("DM_Tau",       0, "inner_cavity"),
    ("HD_135344B",   1, "inner_cavity"),
    ("J1604",        0, "inner_cavity"),
    ("J1615",        1, "inner_cavity"),
    ("J1852",        0, "inner_cavity"),
    ("LkCa_15",      1, "inner_cavity"),
    ("MWC_758",      0, "inner_cavity"),
    ("SY_Cha",       0, "inner_cavity"),
    ("V4046_Sgr",    0, "inner_cavity"),
    ("CQ_Tau",       0, "inner_cavity"),
]

SUBSUF = "0"   # file-suffix for partial-run bookkeeping (matches new_gap convention)


# ---------------------------------------------------------------------------
# Template functions — each returns the file content as a string
# ---------------------------------------------------------------------------

def injectloop(target, gap_ix):
    return f"""\
import os, sys, time
import numpy as np

sys.path.append('{SRC_PATH}/')
from inject_CPD import inject_CPD

from frank.geometry import FixedGeometry
from frank.radial_fitters import FrankFitter
from frank.io import save_fit
sys.path.append('{SRC_PATH}/{DICT_SUBDIR}/')
import diskdictionary_eqC1 as disk


# specify target disk and gap
target  = '{target}'    # CSD name
gap_ix  = {gap_ix}           # which gap CPD is in (based on dict list)
subsuf  = '{SUBSUF}'         # suffix to attach to records (if partial work)


# specify mock parameters
F_cpd = np.arange(2*disk.disk[target]['RMS']/1000,
                  15*disk.disk[target]['RMS']/1000,
                  1*disk.disk[target]['RMS']/1000)   # in mJy
n_mocks_per_F = 50              # number of mocks per flux bin


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
dat = np.load('{SRC_PATH.replace("Source_codes", "exoALMA_disk_data")}/measurement_set_spavg/npz/'
              + target + '_time_ave_continuum_spavg_lambda.vis.npz')
u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']


# loop through mock injection and modeling
os.system('rm -rf '+target+'_gap'+str(gap_ix)+'_mpars.'+subsuf+'.txt')

t0 = time.time()
for i in range(len(F_cpd)):

    # assign random radii and azimuths for mock CPDs (in disk plane)
    rgap_cen = disk.disk[target]['rgap'][gap_ix]
    gap_span  = 0.5 * disk.disk[target]['wgap'][gap_ix]   # wgap = sigma_gap
    r_cpd  = np.random.uniform(rgap_cen - gap_span,
                               rgap_cen + gap_span, n_mocks_per_F)
    az_cpd = np.random.randint(-180, 180, n_mocks_per_F)

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
            f.write('%i    %s    %.3f    %i\\n' %
                    (int(np.round(1e3*F_cpd[i])), str(j).zfill(4),
                     r_cpd[j], az_cpd[j]))

print(time.time() - t0)

# move mpars file into injections/
os.makedirs("injections", exist_ok=True)
mpars_name = f"{{target}}_gap{{gap_ix}}_mpars.{{subsuf}}.txt"
os.replace(mpars_name, os.path.join("injections", mpars_name))
"""


def prepimaging(target, gap_ix):
    return f"""\
import os, sys, time
import numpy as np

sys.path.append('{SRC_PATH}/')
from custom_mask import custom_mask

sys.path.append('{SRC_PATH}/{DICT_SUBDIR}/')
import diskdictionary_eqC1 as disk

# specify target disk, gap, and mock file index
target, gap_ix, subsuf = '{target}', '{gap_ix}', '{SUBSUF}'

# package up information to pass to CASA
f = open('whichdisk.txt', 'w')
f.write(target + '\\n' + gap_ix + '\\n' + subsuf)
f.close()

# preliminary CASA imaging to set up for image loop
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < prepimaging_casa.py')

# make a custom mask for the gap of interest
custom_mask(target, int(gap_ix), target+'_gap'+gap_ix+'.'+subsuf,
            buffer_factor=1.5)

# make a script to convert custom mask into CASA format
os.system('casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile --noconfirm < mask_to_casa.py')
"""


def prepimaging_casa(target, gap_ix):
    spavg_ms = (f"{SRC_PATH.replace('Source_codes','exoALMA_disk_data')}"
                "/measurement_set_spavg")
    return f"""\
import os, sys, time
import numpy as np

sys.path.append('{SRC_PATH}/')
execfile('{SRC_PATH}/JvM_correction_brief.py', globals())
execfile('{SRC_PATH}/ImportMS.py', globals())

sys.path.append('{SRC_PATH}/{DICT_SUBDIR}/')
import diskdictionary_eqC1 as disk

# specify target disk and gap
target, gap_ix, subsuf = np.loadtxt('whichdisk.txt', dtype=str)

# load mock injection parameters file data (as strings)
inj_file = 'injections/'+target+'_gap'+gap_ix+'_mpars.'+subsuf+'.txt'
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T

t0 = time.time()

# create a MS from the first residual file to build psf/sumwt
rfile = 'resid_vis/' + target + '_gap' + gap_ix + \\
        '_F' + Fstr[0] + 'uJy_' + mstr[0] + '_frank_uv_resid'
resid_suffix = 'gap'+gap_ix+'.F'+Fstr[0]+'uJy_'+mstr[0]+'.resid'
os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')

ImportMS('{spavg_ms}/'+target+'_time_ave_continuum_spavg.ms',
         rfile, suffix=resid_suffix, make_resid=False)

# make images as basis for imageloop
im_outfile = target+'_'+resid_suffix
for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
    os.system('rm -rf '+im_outfile+ext)

tclean(vis='{spavg_ms}/'+target+'_time_ave_continuum_spavg.'+resid_suffix+'.ms',
       imagename=im_outfile, specmode='mfs', deconvolver='multiscale',
       imsize=1024, cell='.006arcsec',
       scales=disk.disk[target]['cscale'], mask=disk.disk[target]['cmask'],
       gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
       weighting='briggs', robust=disk.disk[target]['crobust'],
       uvtaper=[], savemodel='none',
       threshold=disk.disk[target]['cthresh'], interactive=False)

# export the mask to a FITS file
exportfits(im_outfile+'.mask',
           target+'_gap'+gap_ix+'.'+subsuf+'.mask.fits', overwrite=True)

# clean up
for ext in ['.image', '.mask', '.model', '.pb', '.residual', '.JvMcorr.image']:
    os.system('rm -rf '+im_outfile+ext)
os.system('rm -rf {spavg_ms}/'+target+'_time_ave_continuum_spavg.'+resid_suffix+'.ms*')
os.system('rm -rf '+target+'_gap'+gap_ix+'.'+subsuf+'.psf')
os.system('mv '+im_outfile+'.psf '+target+'_gap'+gap_ix+'.'+subsuf+'.psf')
os.system('rm -rf '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt')
os.system('mv '+im_outfile+'.sumwt '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt')

print(time.time()-t0)
"""


def imageloop(target, gap_ix):
    spavg_ms = (f"{SRC_PATH.replace('Source_codes','exoALMA_disk_data')}"
                "/measurement_set_spavg")
    return f"""\
import os, sys, time
import numpy as np

sys.path.append('{SRC_PATH}/')
exec(compile(open('{SRC_PATH}/reduction_utils.py', "rb").read(), 'reduction_utils.py', 'exec'))
exec(compile(open('{SRC_PATH}/JvM_correction_brief.py', "rb").read(), 'JvM_correction_brief.py', 'exec'))
exec(compile(open('{SRC_PATH}/ImportMS.py', "rb").read(), 'ImportMS.py', 'exec'))

sys.path.append('{SRC_PATH}/{DICT_SUBDIR}/')
import diskdictionary_eqC1 as disk

# Create output directory for residual images
os.makedirs('resid_images', exist_ok=True)

# specify target disk and gap
target, gap_ix, subsuf = "{target}", "{gap_ix}", "{SUBSUF}"

# load mock injection parameters file data
inj_file = 'injections/'+target+'_gap'+gap_ix+'_mpars.'+subsuf+'.txt'
print(f"Loading: {{inj_file}}")
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T

# loop through each set of residuals and make an image
for i in range(len(Fstr)):

    t0 = time.time()

    rfile = 'resid_vis/'+target+'_gap'+gap_ix+ \\
            '_F'+Fstr[i]+'uJy_'+mstr[i]+'_frank_uv_resid'
    resid_suffix = 'gap'+gap_ix+'.F'+Fstr[i]+'uJy_'+mstr[i]+'.resid'

    if os.path.exists('resid_images/'+target+'_'+resid_suffix+'.fits'):
        continue

    os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')
    ImportMS('{spavg_ms}/'+target+'_time_ave_continuum_spavg.ms',
             rfile, suffix=resid_suffix, make_resid=False)

    im_outfile = target+'_'+resid_suffix

    for ext in ['.image', '.model', '.pb', '.residual', '.mask']:
        os.system('rm -rf '+im_outfile+ext)
    os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.psf   '+im_outfile+'.psf')
    os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt '+im_outfile+'.sumwt')

    tclean(vis='{spavg_ms}/'+target+'_time_ave_continuum_spavg.'+resid_suffix+'.ms',
           imagename=im_outfile, specmode='mfs', deconvolver='multiscale',
           imsize=1024, cell='.006arcsec', scales=disk.disk[target]['gscales'],
           mask=target+'_gap'+gap_ix+'.'+subsuf+'.custom.mask',
           gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
           weighting='briggs', robust=disk.disk[target]['crobust'],
           uvtaper=[], savemodel='none',
           threshold=disk.disk[target]['gthresh'], interactive=False,
           calcpsf=False)

    exportfits(im_outfile+'.image',
               'resid_images/'+im_outfile+'.fits', overwrite=True)

    for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual',
                '.sumwt', '.JvMcorr.image']:
        os.system('rm -rf '+im_outfile+ext)
    os.system('rm -rf {spavg_ms}/'+target+'_time_ave_continuum_spavg.'+resid_suffix+'.ms*')

    print(time.time()-t0)
"""


def recover_loop(target, gap_ix):
    return f"""\
import os, sys, time
import numpy as np
from astropy.io import fits

sys.path.append('{SRC_PATH}/')
sys.path.append('{SRC_PATH}/{DICT_SUBDIR}/')
import diskdictionary_eqC1 as disk

# target disk/gap; iteration
target = '{target}'
gap    = {gap_ix}
ix     = '{SUBSUF}'

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

    # Load gap properties (wgap = sigma_gap from eqC1 fit)
    rgap = disk.disk[target]['rgap'][gap]
    wgap = disk.disk[target]['wgap'][gap]

    # Boolean mask: search annulus = rgap +/- wgap (= +/- 1 sigma)
    mask = np.zeros_like(img, dtype='bool')
    bndi, bndo = (rd >= (rgap - wgap)), (rd <= (rgap + wgap))
    mask[np.logical_and(bndi, bndo)] = 1
    g_img, g_xs, g_ys = img[mask], xs[mask], ys[mask]
    g_rd, g_azd = rd[mask], azd[mask]

    # Locate and measure the peak
    peak = (g_img == g_img.max())
    pk_xs, pk_ys = g_xs[peak][0], g_ys[peak][0]
    pk_r,  pk_az = g_rd[peak][0], g_azd[peak][0]
    pk_SB = g_img.max()

    # Quantify the pixel distribution in the gap region
    dist_peak = np.sqrt((xs-pk_xs)**2 + (ys-pk_ys)**2)
    mask[(dist_peak <= (5 * hd['CDELT2'] * 3600))] = 0
    emean, estd = np.mean(img[mask]), np.std(img[mask])

    with open(recov_file, 'a') as f:
        f.write('%.0f  %.0f  %s  %.3f  %.3f  %4i  %4i  %.5f  %.5f  %.0f  %.0f\\n' %
                (Fcpd[i], pk_SB, mstr[i], rcpd[i], pk_r, azcpd[i], pk_az,
                 pk_xs, pk_ys, emean, estd))
"""


def run_pipeline_sh(target, gap_ix, subfolder):
    folder_name = f"{target}_gap{gap_ix}"
    workdir     = f"{HPC_BASE}/{subfolder}/{folder_name}"
    venv        = ("/nexus/posix0/MIA-astro-env/myben/vawelke"
                   "/venvs/frank_env/bin/activate")
    casa_bin    = ("/nexus/posix0/MIA-astro-env/myben/vawelke"
                   "/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin")
    injectloop_script = f"{folder_name}_injectloop.py"
    imageloop_script  = f"{target}_robust0_5_gap{gap_ix}_imageloop.py"
    mpars_file = f"injections/{target}_gap{gap_ix}_mpars.{SUBSUF}.txt"
    return f"""\
#!/usr/bin/env bash

WORKDIR="{workdir}"
VENV="{venv}"
CASA_BIN="{casa_bin}"
export CASA_CONFIG="/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py"

cd "$WORKDIR"
mkdir -p resid_vis mprofiles recoveries resid_images injections
source "$VENV"
export PATH="${{CASA_BIN}}:$PATH"

if [ -f "{mpars_file}" ]; then
    echo "=== [{folder_name}] Inject already done, skipping ==="
else
    echo "=== [{folder_name}] Inject start: $(date) ==="
    OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 \\
    OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
    python -u {injectloop_script} > inject.log 2>&1
    echo "=== Inject finished: $(date) ===" >> inject.log
fi

if compgen -G "*.custom.mask" > /dev/null 2>&1; then
    echo "=== Prepimaging already done, skipping ==="
else
    echo "=== [{folder_name}] Prepimaging start: $(date) ==="
    CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \\
    MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
    python -u prepimaging.py > prepimaging.log 2>&1
    echo "=== Prepimaging finished: $(date) ===" >> prepimaging.log
fi

echo "=== [{folder_name}] Imageloop start: $(date) ==="
CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \\
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \\
  -c {imageloop_script} > imageloop.log 2>&1
echo "=== Imageloop finished: $(date) ===" >> imageloop.log

echo "=== [{folder_name}] Recovery start: $(date) ==="
python -u recover_loop.py > recover_loop.log 2>&1
echo "=== Recovery finished: $(date) ===" >> recover_loop.log

deactivate

cd "$WORKDIR/resid_images"
cp $(ls -1t *.fits | head -n 2) ../recoveries/

echo "=== [{folder_name}] Pipeline complete: $(date) ==="
"""


MASK_TO_CASA = """\
import os, sys
import numpy as np

sys.path.append(os.getcwd())

target, gap_ix, subsuf = np.loadtxt('whichdisk.txt', dtype=str)

importfits(target+'_gap'+gap_ix+'.'+subsuf+'.custom_mask.fits',
           target+'_gap'+gap_ix+'.'+subsuf+'.custom.mask', overwrite=True)
"""


# ---------------------------------------------------------------------------
# Main: create folders and write files
# ---------------------------------------------------------------------------

def main():
    created = []
    for target, gap_ix, subfolder in TARGETS:
        folder_name = f"{target}_gap{gap_ix}"
        folder_path = os.path.join(BASE_DIR, subfolder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        files = {
            f"{folder_name}_injectloop.py":                    injectloop(target, gap_ix),
            "prepimaging.py":                                  prepimaging(target, gap_ix),
            "prepimaging_casa.py":                             prepimaging_casa(target, gap_ix),
            f"{target}_robust0_5_gap{gap_ix}_imageloop.py":   imageloop(target, gap_ix),
            "recover_loop.py":                                 recover_loop(target, gap_ix),
            "run_pipeline.sh":                                 run_pipeline_sh(target, gap_ix, subfolder),
            "mask_to_casa.py":                                 MASK_TO_CASA,
        }

        for fname, content in files.items():
            fpath = os.path.join(folder_path, fname)
            with open(fpath, 'w') as fh:
                fh.write(content)
            created.append(os.path.relpath(fpath, BASE_DIR))

    print(f"Created {len(created)} files in {len(TARGETS)} folders.")
    for p in created:
        print(" ", p)


if __name__ == "__main__":
    main()

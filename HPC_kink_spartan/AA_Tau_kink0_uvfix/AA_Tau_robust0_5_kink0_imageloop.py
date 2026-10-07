import os, sys, time
import numpy as np

sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/')
exec(compile(open('/data/gpfs/projects/punim3000/MPIA/Source_codes/reduction_utils.py', "rb").read(), 'reduction_utils.py', 'exec'))
exec(compile(open('/data/gpfs/projects/punim3000/MPIA/Source_codes/JvM_correction_brief.py', "rb").read(), 'JvM_correction_brief.py', 'exec'))
exec(compile(open('/data/gpfs/projects/punim3000/MPIA/Source_codes/ImportMS.py', "rb").read(), 'ImportMS.py', 'exec'))

sys.path.append('/data/gpfs/projects/punim3000/MPIA/Source_codes/diskdictionary_r90/')
import diskdictionary_eqC1 as disk

# Create output directory for residual images
os.makedirs('resid_images', exist_ok=True)

# specify target disk and kink candidate
target, gap_ix, subsuf = "AA_Tau", "2", "0"

# load mock injection parameters file data
inj_file = 'injections/'+target+'_gap'+gap_ix+'_mpars.'+subsuf+'.txt'
print(f"Loading: {inj_file}")
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T

# loop through each set of residuals and make an image
for i in range(len(Fstr)):

    t0 = time.time()

    rfile = 'resid_vis/'+target+'_gap'+gap_ix+ \
            '_F'+Fstr[i]+'uJy_'+mstr[i]+'_frank_uv_resid'
    resid_suffix = 'gap'+gap_ix+'.F'+Fstr[i]+'uJy_'+mstr[i]+'.resid'

    if os.path.exists('resid_images/'+target+'_'+resid_suffix+'.fits'):
        continue

    os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')
    ImportMS('/data/gpfs/projects/punim3000/MPIA/exoALMA_disk_data/measurement_set_spavg/'+target+'_time_ave_continuum_spavg.ms',
             rfile, suffix=resid_suffix, make_resid=False)

    im_outfile = target+'_'+resid_suffix

    for ext in ['.image', '.model', '.pb', '.residual', '.mask']:
        os.system('rm -rf '+im_outfile+ext)
    os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.psf   '+im_outfile+'.psf')
    os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt '+im_outfile+'.sumwt')

    tclean(vis='/data/gpfs/projects/punim3000/MPIA/exoALMA_disk_data/measurement_set_spavg/'+target+'_time_ave_continuum_spavg.'+resid_suffix+'.ms',
           imagename=im_outfile, specmode='mfs', deconvolver='multiscale',
           imsize=1024, cell='.006arcsec', scales=disk.disk[target]['gscales'],
           mask=target+'_gap'+gap_ix+'.'+subsuf+'.custom.mask',
           gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
           weighting='briggs', robust=disk.disk[target]['crobust'],
           uvtaper=[], savemodel='none',
           # clean to 1x RMS (not the disk's usual gthresh) -- same deeper
           # cleaning convention used for the HD_135344B point candidates
           threshold='%.5fmJy' % (disk.disk[target]['RMS']/1000), interactive=False,
           calcpsf=False)

    exportfits(im_outfile+'.image',
               'resid_images/'+im_outfile+'.fits', overwrite=True)

    for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual',
                '.sumwt', '.JvMcorr.image']:
        os.system('rm -rf '+im_outfile+ext)
    os.system('rm -rf /data/gpfs/projects/punim3000/MPIA/exoALMA_disk_data/measurement_set_spavg/'+target+'_time_ave_continuum_spavg.'+resid_suffix+'.ms*')

    print(time.time()-t0)

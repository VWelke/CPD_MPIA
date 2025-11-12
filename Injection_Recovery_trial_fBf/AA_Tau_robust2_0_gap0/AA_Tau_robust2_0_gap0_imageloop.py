import os, sys, time
import numpy as np
sys.path.append('../DSHARP_source_code')
exec(compile(open('../DSHARP_source_code/reduction_utils.py', "rb").read(), '../DSHARP_source_code/reduction_utils.py', 'exec'))
exec(compile(open('../DSHARP_source_code/JvM_correction_brief.py', "rb").read(), '../DSHARP_source_code/JvM_correction_brief.py', 'exec'))
exec(compile(open('../DSHARP_source_code/ImportMS_copy.py', "rb").read(), '../DSHARP_source_code/ImportMS.py', 'exec')) 

sys.path.append('.')  # Add current directory where diskdictionaryr2_0.py is located
import diskdictionaryr2_0 as disk

# Create output directory for residual images
os.makedirs('resid_images', exist_ok=True)

# specify target disk and gap
target, gap_ix, subsuf = "AA_Tau", "0", "0"

# Specify which flux bin to process (can be set from pipeline script)
# Example: flux_bin_uJy = 250  # Process 250 μJy flux bin
# If not specified, default to looking for the old combined file

flux_bin_uJy = 24  # Set by pipeline script
inj_file = f'injections/{target}_gap{gap_ix}_F{flux_bin_uJy}uJy_mpars.{subsuf}.txt'
print(f"Processing specific flux bin: {flux_bin_uJy} μJy")


print(f"Loading: {inj_file}")
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T


# loop through each set of residuals and make an image
for i in range(min(5, len(Fstr))):

    t0 = time.time()

    # create a MS from the residual visibilities
    rfile = 'resid_vis/'+target+'_gap'+gap_ix+ \
            '_F'+Fstr[i]+'uJy_'+mstr[i]+'_frank_uv_resid'
    resid_suffix = 'gap'+gap_ix+'.F'+Fstr[i]+'uJy_'+mstr[i]+'.resid'
    #resid_ms = target + '_data.' + resid_suffix + '.ms'
    resid_ms = f"/mnt/d/exoALMA_disk_data/data/{target}_time_ave_continuum.{resid_suffix}.ms"
    if not os.path.exists(resid_ms):
        print(f"Residual MS not found: {resid_ms}")
        print("Creating new residual MS via ImportMS() ...")
        ImportMS('/mnt/d/exoALMA_disk_data/data/'+target+'_time_ave_continuum.ms',
                 rfile, suffix=resid_suffix, make_resid=False)
        print("✅ ImportMS completed successfully.")
    else:
        print(f"✅ Using existing residual MS: {resid_ms}")

    
    

    # prepare for imaging
    im_outfile = target+'_'+resid_suffix

    for ext in ['.image', '.model', '.pb', '.residual', '.mask']:
        os.system('rm -rf '+im_outfile+ext)
    #os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.psf ' + \
    #          im_outfile+'.psf')
    #os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt ' + \
    #          im_outfile+'.sumwt')

    # clean
    tclean(vis='/mnt/d/exoALMA_disk_data/data/'+target+'_time_ave_continuum.'+resid_suffix+'.ms',
       imagename=im_outfile, specmode='mfs', deconvolver='multiscale',
       imsize=1024, cell='.006arcsec', scales=disk.disk[target]['gscales'],
       mask=target+'_gap'+gap_ix+'.'+subsuf+'.custom.mask',
       gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
       weighting='briggs', robust=disk.disk[target]['crobust'],
       uvtaper=[], savemodel='none',
       threshold=disk.disk[target]['gthresh'][int(gap_ix)], interactive=False,
       calcpsf=True)

    # perform the JvM correction
    eps = do_JvM_correction_and_get_epsilon(im_outfile)

    # export the resulting images to FITS files
    exportfits(im_outfile+'.JvMcorr.image', 
               'resid_images/'+im_outfile+'.JvMcorr.fits', overwrite=True)
    exportfits(im_outfile+'.image', 'resid_images/'+im_outfile+'.fits',
               overwrite=True)
               
    # clean up
    #for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', 
    #            '.sumwt', '.JvMcorr.image']:
    #    os.system('rm -rf '+im_outfile+ext)
    #os.system('rm -rf data/'+target+'_data.'+resid_suffix+'.ms*')

    print((time.time()-t0))

# optional cleanup
import shutil, os
local_ms = '/home/vrice/exoALMA_disk_data/data/' + target + '_time_ave_continuum.ms'
if os.path.exists(local_ms):
    print(f"Cleaning up local MS copy at {local_ms} ...")
    #shutil.rmtree(local_ms)
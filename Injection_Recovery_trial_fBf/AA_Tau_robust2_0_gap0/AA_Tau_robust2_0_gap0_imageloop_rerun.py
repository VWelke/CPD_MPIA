import os, sys, time
import numpy as np

sys.path.append(os.getcwd())

exec(compile(open('reduction_utils.py', "rb").read(), 'reduction_utils.py', 'exec'))
exec(compile(open('JvM_correction_brief.py', "rb").read(), 'JvM_correction_brief.py', 'exec'))
exec(compile(open('ImportMS.py', "rb").read(), 'ImportMS.py', 'exec'))

import diskdictionaryr2_0 as disk

os.makedirs('resid_images', exist_ok=True)

target, gap_ix, subsuf = "AA_Tau", "0", "0"

inj_file = f"injections/{target}_gap{gap_ix}_mpars.{subsuf}.txt"
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T

print(f"Loading: {inj_file}")


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
for i in range(len(Fstr)):

    t0 = time.time()

    # ----- MS creation part -----
    resid_suffix = f"gap{gap_ix}.F{Fstr[i]}uJy_{mstr[i]}.resid"
    rfile = f"resid_vis/{target}_gap{gap_ix}_F{Fstr[i]}uJy_{mstr[i]}_frank_uv_resid"

    resid_ms_path = f"/mnt/d/exoALMA_disk_data/measurement_set_spavg/{target}_time_ave_continuum_spavg.{resid_suffix}.ms"

    # SKIP MS CREATION IF EXISTS
    if not os.path.isdir(resid_ms_path):
        print("MS does not exist — creating:", resid_ms_path)
        ImportMS(f"/mnt/d/exoALMA_disk_data/measurement_set_spavg/{target}_time_ave_continuum_spavg.ms",
                 rfile, suffix=resid_suffix, make_resid=False)
    else:
        print("MS already exists — skipping creation:", resid_ms_path)


    # ----- Imaging output filenames -----
    im_outfile = f"{target}_{resid_suffix}"
    corr_fits = f"resid_images/{im_outfile}.JvMcorr.fits"
    raw_fits  = f"resid_images/{im_outfile}.fits"

    # SKIP IMAGING + JvM IF FITS ALREADY EXISTS
    if os.path.isfile(corr_fits) and os.path.isfile(raw_fits):
        print("FITS outputs already exist — skipping imaging:", corr_fits)
        continue

    print("Running CLEAN + JvM for:", im_outfile)


    # ----- Clean old temp image products -----
    for ext in ['.image', '.model', '.pb', '.residual', '.mask']:
        os.system('rm -rf '+im_outfile+ext)

    os.system(f"cp -r {target}_gap{gap_ix}.{subsuf}.psf    {im_outfile}.psf")
    os.system(f"cp -r {target}_gap{gap_ix}.{subsuf}.sumwt  {im_outfile}.sumwt")


    # ----- Run tclean -----
    tclean(vis=f'/mnt/d/exoALMA_disk_data/measurement_set_spavg/{target}_time_ave_continuum_spavg.{resid_suffix}.ms',
           imagename=im_outfile, specmode='mfs', deconvolver='multiscale',
           imsize=1024, cell='.006arcsec', scales=disk.disk[target]['gscales'],
           mask=f"{target}_gap{gap_ix}.{subsuf}.custom.mask",
           gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
           weighting='briggs', robust=disk.disk[target]['crobust'],
           uvtaper=[], savemodel='none',
           threshold=disk.disk[target]['gthresh'], interactive=False,
           calcpsf=False)

    # ----- JvM correction -----
    eps = do_JvM_correction_and_get_epsilon(im_outfile)

    # ----- Export to FITS -----
    exportfits(im_outfile+'.JvMcorr.image', corr_fits, overwrite=True)
    exportfits(im_outfile+'.image', raw_fits, overwrite=True)

    # ----- Clean temporary CASA image products -----
    for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual',
                '.sumwt', '.JvMcorr.image']:
        os.system('rm -rf '+im_outfile+ext)

    print(f"Finished {i+1}/{len(Fstr)} in {time.time()-t0:.1f}s")


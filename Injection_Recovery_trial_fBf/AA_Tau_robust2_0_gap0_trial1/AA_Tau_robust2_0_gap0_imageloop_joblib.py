import os, sys, time
import numpy as np
from joblib import Parallel, delayed

sys.path.append('../DSHARP_source_code')
execfile('../DSHARP_source_code/reduction_utils.py')
execfile('../DSHARP_source_code/JvM_correction_brief.py')
execfile('../DSHARP_source_code/ImportMS.py')
sys.path.append('../')
import diskdictionaryr2_0 as disk

# Create output directory for residual images
os.makedirs('resid_images', exist_ok=True)

# specify target disk and gap
target, gap_ix, subsuf = "AA_Tau", 0, "0"

# Parallel processing configuration
N_CORES_IMAGING = 4  # Adjust based on memory (each tclean uses ~2-4GB RAM)

# Specify which flux bin to process
flux_bin_uJy = 24  # Set by pipeline script
inj_file = f'injections/{target}_gap{gap_ix}_F{flux_bin_uJy}uJy_mpars.{subsuf}.txt'
print(f"Processing specific flux bin: {flux_bin_uJy} μJy using {N_CORES_IMAGING} cores")

print(f"Loading: {inj_file}")
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T

def process_single_mock(i, Fstr_i, mstr_i):
    """Process a single mock for imaging - this function runs in parallel"""
    
    try:
        t0 = time.time()
        
        print(f"Starting mock {i+1}/{len(Fstr)}: {Fstr_i} μJy")

        # create a MS from the residual visibilities
        rfile = 'resid_vis/'+target+'_gap'+str(gap_ix)+ \
                '_F'+Fstr_i+'uJy_'+mstr_i+'_frank_uv_resid'
        resid_suffix = 'gap'+str(gap_ix)+'.F'+Fstr_i+'uJy_'+mstr_i+'.resid'
        
        # Clean up any existing MS files
        os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')
        
        # Import MS with residual visibilities
        ImportMS('/mnt/d/exoALMA_disk_data/data/' +target+'_time_ave_continuum.ms', rfile, suffix=resid_suffix, make_resid=False)

        # prepare for imaging
        im_outfile = target+'_'+resid_suffix
        for ext in ['.image', '.model', '.pb', '.residual']:
            os.system('rm -rf '+im_outfile+ext)
            
        # Copy PSF and sumwt files
        os.system('cp -r '+target+'_gap'+str(gap_ix)+'.'+subsuf+'.psf ' + \
                  im_outfile+'.psf')
        os.system('cp -r '+target+'_gap'+str(gap_ix)+'.'+subsuf+'.sumwt ' + \
                  im_outfile+'.sumwt')

        # Run tclean imaging
        tclean(vis='/mnt/d/exoALMA_disk_data/data/' +target+'_data.'+resid_suffix+'.ms',
               imagename=im_outfile, specmode='mfs', deconvolver='multiscale',
               imsize=1024, cell='.006arcsec', scales=disk.disk[target]['gscales'],
               mask=target+'_gap'+str(gap_ix)+'.'+subsuf+'.custom.mask', 
               gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
               weighting='briggs', robust=disk.disk[target]['crobust'],
               uvtaper=disk.disk[target]['ctaper'], savemodel='none',
               threshold=disk.disk[target]['gthresh'], interactive=False, 
               calcpsf=False)

        # perform the JvM correction
        eps = do_JvM_correction_and_get_epsilon(im_outfile)

        # export the resulting images to FITS files
        exportfits(im_outfile+'.JvMcorr.image', 
                   'resid_images/'+im_outfile+'.JvMcorr.fits', overwrite=True)
        exportfits(im_outfile+'.image', 'resid_images/'+im_outfile+'.fits',
                   overwrite=True)
                   
        # clean up intermediate files
        for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', 
                    '.sumwt', '.JvMcorr.image']:
            os.system('rm -rf '+im_outfile+ext)
        os.system('rm -rf "/mnt/d/exoALMA_disk_data/data/'+target+'_data.'+resid_suffix+'.ms*"')

        elapsed_time = time.time()-t0
        print(f"✓ Mock {i+1} completed: {Fstr_i} μJy in {elapsed_time:.1f}s")
        
        return f"Mock {i+1}: {Fstr_i} μJy - SUCCESS ({elapsed_time:.1f}s)"
        
    except Exception as e:
        print(f"✗ Mock {i+1} FAILED: {str(e)}")
        return f"Mock {i+1}: {Fstr_i} μJy - FAILED: {str(e)}"

# Main parallel execution
print(f"\n=== Starting parallel imaging of {len(Fstr)} mocks ===")
print(f"Using {N_CORES_IMAGING} parallel processes with joblib")
print(f"Target: {target}, Gap: {gap_ix}, Flux: {flux_bin_uJy} μJy")

# Record start time
pipeline_start = time.time()

# Process all mocks in parallel using joblib with threading backend
results = Parallel(n_jobs=N_CORES_IMAGING, backend='threading', verbose=1)(
    delayed(process_single_mock)(i, Fstr[i], mstr[i]) 
    for i in range(len(Fstr))
)

# Print summary
pipeline_time = time.time() - pipeline_start
successful = sum(1 for r in results if 'SUCCESS' in r)
failed = len(results) - successful

print(f"\n=== Parallel Imaging Complete ===")
print(f"Total time: {pipeline_time:.1f} seconds")
print(f"Successful: {successful}/{len(Fstr)} mocks")
print(f"Failed: {failed}/{len(Fstr)} mocks")
print(f"Average time per mock: {pipeline_time/len(Fstr):.1f}s")

if failed > 0:
    print(f"\nFailed mocks:")
    for r in results:
        if 'FAILED' in r:
            print(f"  {r}")

print(f"\nAll FITS files saved to: resid_images/")
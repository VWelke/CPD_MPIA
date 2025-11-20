###############################################################################################
# The script differs from the original DSHARP injection/recovery script by:
# - using the robust 2.0 AA Tau data
# - the final mpars.txt will be split into separate txt for each flux bin so that the recovery rate can be quickly calculated per flux bin once its done
# - reducing the number of injections per flux bin to 100 (from 500)
# - changing the flux range from 0.25-0.01 muJy in 0.01 muJy steps to 1RMS - 90% recovery range, in 0.01 muJy steps
# - MULTICORE: Uses joblib parallel processing to speed up mock processing within each flux bin
###############################################################################################


import os, sys, time
import numpy as np
import sys
import platform
from joblib import Parallel, delayed

# Cross-platform path resolution for DSHARP_source_code
if platform.system() == 'Windows':
    sys.path.append('D:\\CPD_MPIA\\Injection_Recovery_trial_fBf\\DSHARP_source_code')
else:  # Linux/WSL
    sys.path.append('/mnt/d/CPD_MPIA/Injection_Recovery_trial_fBf/DSHARP_source_code')

from inject_CPD import inject_CPD
from frank.geometry import FixedGeometry
from frank.radial_fitters import FrankFitter
from frank.io import save_fit
import diskdictionaryr2_0 as disk


os.makedirs('resid_vis', exist_ok=True)
os.makedirs('mprofiles', exist_ok=True)

# specify target disk and gap
target = "AA_Tau"	# CSD name
gap_ix = 0 # which gap CPD is in (based on dict list)
subsuf = "0" 	# suffix to attach to records (if partial work)

# create injection subfolder for this target and gap
injection_folder = f'./injections'
os.makedirs(injection_folder, exist_ok=True)


# specify mock parameters
F_cpd = np.arange( disk.disk[target]['RMS']/1000, 0.25, 0.01)        # in mJy
n_mocks_per_F = 10  			    # number of mocks per flux bin
 # proposed solution , from figure 6 of Andrews, reduce range form 0.25-0.00 to 0.15-0.05 mJy as most embedds the 0.5 recovery fraction
 # prioritise on the kinks, not gaps
 # reduce n_mocks per F  -> planet kink lower position uncertainty due to smaller width..

# -------


# fixed geometric parameters of CSD
incl, PA = disk.disk[target]['incl'], disk.disk[target]['PA']
offRA, offDEC = disk.disk[target]['dx'], disk.disk[target]['dy']
geom = FixedGeometry(incl, PA, dRA=offRA, dDec=offDEC)

# frank setup
Rmax, Ncoll = 2 * disk.disk[target]['rout'], disk.disk[target]['hyp-Ncoll']
alpha, wsmth = disk.disk[target]['hyp-alpha'], disk.disk[target]['hyp-wsmth']
FF = FrankFitter(Rmax=Rmax, N=Ncoll, geometry=geom, alpha=alpha, 
                 weights_smooth=wsmth)





# load the visibility data
# Cross-platform path resolution for data directory
if platform.system() == 'Windows':
    data_path = 'D:\\exoALMA_disk_data\\data\\' + target + '_time_ave_continuum.vis.npz'
else:  # Linux/WSL
    data_path = '/mnt/d/exoALMA_disk_data/data/' + target + '_time_ave_continuum.vis.npz'

dat = np.load(data_path)
u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

# Multicore configuration
N_CORES = 5

def process_mock_joblib(j, F_cpd_i, r_cpd_j, az_cpd_j, shared_data):
    """Process single mock with joblib parallel processing"""
    # Unpack shared data
    u, v, vis, wgt = shared_data['u'], shared_data['v'], shared_data['vis'], shared_data['wgt']
    target, gap_ix, subsuf = shared_data['target'], shared_data['gap_ix'], shared_data['subsuf']
    
    # Create frank fitter for this worker
    geom = FixedGeometry(shared_data['incl'], shared_data['PA'], 
                        dRA=shared_data['offRA'], dDec=shared_data['offDEC'])
    FF = FrankFitter(Rmax=shared_data['Rmax'], N=shared_data['Ncoll'], 
                     geometry=geom, alpha=shared_data['alpha'], 
                     weights_smooth=shared_data['wsmth'])
    
    # bookkeeping
    file_suffix = '_F'+str(int(np.round(1e3*F_cpd_i))) + 'uJy_'+str(j).zfill(4)

    # inject a mock CPD into the data
    vis_cpd = inject_CPD((u, v, vis, wgt), (F_cpd_i, r_cpd_j, az_cpd_j),
                         incl=shared_data['incl'], PA=shared_data['PA'], 
                         offRA=shared_data['offRA'], offDEC=shared_data['offDEC'])

    # frank modeling of the data + CPD injection
    sol = FF.fit(u, v, vis_cpd, wgt)

    # save the frank results
    save_fit(u, v, vis_cpd, wgt, sol, 
             prefix=target+'_gap'+str(gap_ix)+file_suffix,
             save_vis_fit=False, save_solution=False)

    # clean up file outputs
    os.system('mv '+target+'_gap'+str(gap_ix)+file_suffix + 
              '_frank_uv_resid.npz resid_vis/')
    os.system('mv '+target+'_gap'+str(gap_ix)+file_suffix + 
              '_frank_profile_fit.txt mprofiles/')
    os.system('rm '+target+'_gap'+str(gap_ix)+file_suffix+'_frank*')

    # return parameter values for writing to mpars file
    return (int(np.round(1e3*F_cpd_i)), str(j).zfill(4), r_cpd_j, az_cpd_j)

# Create shared data dictionary for parallel processing
shared_data = {
    'u': u, 'v': v, 'vis': vis, 'wgt': wgt,
    'target': target, 'gap_ix': gap_ix, 'subsuf': subsuf,
    'incl': incl, 'PA': PA, 'offRA': offRA, 'offDEC': offDEC,
    'Rmax': Rmax, 'Ncoll': Ncoll, 'alpha': alpha, 'wsmth': wsmth
}

# loop through mock injection and modeling 
# Note: Individual flux bin mpars files will be created/overwritten as needed

# for each CPD flux bin
t0 = time.time()
for i in range(len(F_cpd)):
    
    # Create separate mpars file for this flux bin
    flux_uJy = int(np.round(1e3*F_cpd[i]))
    mpars_file = f'{injection_folder}/{target}_gap{gap_ix}_F{flux_uJy}uJy_mpars.{subsuf}.txt'
    
    print(f"\n=== Starting flux bin {i+1}/{len(F_cpd)}: {flux_uJy} μJy ===")
    print(f"Mpars file: {mpars_file}")

    # assign random radii and azimuths for mock CPDs (in disk plane)
    rgap_cen = disk.disk[target]['rgap'][gap_ix]
    gap_span = 0.5 * disk.disk[target]['wgap'][gap_ix]
    r_cpd = np.random.uniform(rgap_cen - gap_span, rgap_cen + gap_span, 
                              n_mocks_per_F)
    az_cpd = np.random.randint(-180, 180, n_mocks_per_F)


    # Process all mocks in this flux bin in parallel
    print(f"Using {N_CORES} cores for parallel processing of {n_mocks_per_F} mocks")
    
    results = Parallel(n_jobs=N_CORES, backend='threading')(
        delayed(process_mock_joblib)(j, F_cpd[i], r_cpd[j], az_cpd[j], shared_data)
        for j in range(n_mocks_per_F)
    )
    
    # Write all results to mpars file
    with open(mpars_file, 'w') as f:  # Use 'w' instead of 'a' since we write all at once
        for result in results:
            if result:  # Check if result is not None
                f.write('%i    %s    %.3f    %i\n' % result)
    
    # Flux bin completed - ask user whether to continue
    print(f"\n=== Flux bin {flux_uJy} μJy completed ({n_mocks_per_F} mocks) ===")
    print(f"Mpars file saved: {mpars_file}")
    
    # Check for pause file or user input
    if os.path.exists('PAUSE_AFTER_FLUX_BIN'):
        print("PAUSE_AFTER_FLUX_BIN file detected. Pausing...")
        while os.path.exists('PAUSE_AFTER_FLUX_BIN'):
            print("Remove 'PAUSE_AFTER_FLUX_BIN' file to continue to next flux bin, or create 'STOP_AT_90_PERCENT' to exit.")
            time.sleep(5)
            if os.path.exists('STOP_AT_90_PERCENT'):
                print("STOP_AT_90_PERCENT file detected. Stopping execution.")
                print(f"Completed {i+1}/{len(F_cpd)} flux bins.")
                exit()
    
    # Optional: Ask for user input (comment out if running unattended)
    # response = input("Continue to next flux bin? (y/n/stop): ").lower()
    # if response in ['n', 'no', 'stop']:
    #     print(f"Stopping after flux bin {flux_uJy} μJy. Completed {i+1}/{len(F_cpd)} flux bins.")
    #     break

print(f"\nTotal time: {time.time() - t0:.1f} seconds")
print(f"Completed all {len(F_cpd)} flux bins.")
"""
v0 script written by Myriam Benisty and Stefano Facchini (Feb 2st 2022)
v1 version adapted by Stefano Facchini and Myriam Benisty (Mar 2nd 2022)
v2 version adapted to include ACA data by Jane Huang and compacted code by Gianni Cataldi (May 19th 2022)
v3 (current version) includes EB alignment by Ryan Loomis and Richard Teague (Aug 2022)


This script is written for CASA 6.2.1.7. 
It can easily be adapted to other versions of CASA, e.g. by modifying the fixvis task to phaseshift task.

The numba package is used for the alignment script, but it is not necessary (it helps with the speed-up)

The scripts are written to be compatible mpicasa.
However, after testingthe exoALMA collaboration decided not to use mpicasa in any of the calibration scripts,
since they realized that differences in the results between casa and mpicasa could arise.

Person in charge for this source:
P. Curone

"""

import os
import numpy as np
import shutil

#path to your local copy of the gihub repository
#(make sure to do a 'git pull' to have the most up-to-date version)
github_path = '/users/pcurone/calibration_scripts'
import sys
sys.path.append(github_path)
import alignment
execfile(os.path.join(github_path,'reduction_utils_exoalma.py'))
#increase the number of figures that can be open before issuing a warning
matplotlib.rcParams.update({'figure.max_open_warning':80})

prefix = 'AA_Tau'

data_folderpath = '/lustre/cv/projects/exoALMA/ALMA_PL_calibrated_data/AA_Tau'
TM_names = {'LB':'TM1','SB':'TM2', 'ACA':'ACA'}
PL_calibrated_vis = {key:os.path.join(data_folderpath,f'{TM_name}/calibrated_final.ms')
                     for key,TM_name in TM_names.items()}

for baseline_key,TM_name in TM_names.items():
    listobs(
        vis=PL_calibrated_vis[baseline_key],
        listfile=f'{prefix}_{TM_name}_calibrated_final.ms.txt',
        overwrite=True,
    )

# System properties.

incl =  59.1    # deg, Loomis et al. (2017)
PA   =  93.2    # deg, Loomis et al. (2017)
v_sys = 6.5     # km/s; from listobs

# Whether to run tclean in parallel or not.

use_parallel = True

fields = {'ACA':'AA_Tau','SB':'AA_Tau','LB':'AA_Tau'}

number_of_EBs = {'ACA':5,'SB':3,'LB':4}

for baseline_key,vis in PL_calibrated_vis.items():
    split_all_obs(msfile=vis,nametemplate=f'{prefix}_{baseline_key}_EB')

for baseline_key,n_EB in number_of_EBs.items():
    for i in range(n_EB):
        vis = f'{prefix}_{baseline_key}_EB{i}.ms'
        listobs(
            vis=vis,
            listfile=f'{vis}.txt',
            overwrite=True,
        )

"""
Check that spws and field numbering is what you expect them to be now from the listobs files
In particular, make sure that there are 4 spws and that there is a single field with
field ID = 0
"""



# Line rest frequencies from splatalogue (https://splatalogue.online)
rest_freq_12CO = 345.79598990e9 #J=3-2
rest_freq_13CO = 330.58796530e9 #J=3-2 (ignoring splitting)
rest_freq_CS   = 342.88285030e9 #J=7-6

data_params_LB = {f'LB{i}': {'vis' : f'{prefix}_LB_EB{i}.ms',
                             'name' : f'LB_EB{i}',
                             'field' : fields['LB'],
                             'line_spws': np.array([0,2,3]), # list of spws containing lines
                             'line_freqs': np.array([rest_freq_13CO,rest_freq_CS,rest_freq_12CO]), #frequencies (Hz) corresponding to line_spws
                             'cont_spws': None,
                             'width_array': None,
                             }
                  for i in range(number_of_EBs['LB'])}

data_params_SB = {f'SB{i}': {'vis' : f'{prefix}_SB_EB{i}.ms',
                             'name' : f'SB_EB{i}',
                             'field' : fields['SB'],
                             'line_spws': np.array([0,2,3]), # list of spws containing lines
                             'line_freqs': np.array([rest_freq_13CO,rest_freq_CS,rest_freq_12CO]), #frequencies (Hz) corresponding to line_spws
                             'cont_spws': None,
                             'width_array': None,
                             }
                  for i in range(number_of_EBs['SB'])}

data_params_ACA = {f'ACA{i}': {'vis' : f'{prefix}_ACA_EB{i}.ms',
                               'name' : f'ACA_EB{i}',
                               'field' : fields['ACA'],
                               'line_spws': np.array([0,2,3]), # list of spws containing lines
                               'line_freqs': np.array([rest_freq_13CO,rest_freq_CS,rest_freq_12CO]), #frequencies (Hz) corresponding to line_spws
                               'cont_spws': None,
                               'width_array': None,
                               }
                   for i in range(number_of_EBs['ACA'])}

data_params = data_params_LB.copy()
data_params.update(data_params_SB)
data_params.update(data_params_ACA)


figures_foldername = 'figures'
os.mkdir(figures_foldername)

def get_figures_folderpath(foldername):
    return os.path.join(figures_foldername,foldername)

def make_figures_folder(folderpath):
    if os.path.isdir(folderpath):
        print(f'going to deleted folder {folderpath} and its content')
        valid_answer = 'yes'
        answer = input(f'to confirm, type \'{valid_answer}\': ')
        if answer == valid_answer:
            shutil.rmtree(folderpath)
        else:
            print('aborting')
            return
    os.mkdir(folderpath)
    return folderpath

preselfcal_amp_figures_folder = get_figures_folderpath('1_preselfcal_amp_figures')
make_figures_folder(preselfcal_amp_figures_folder)

#adjust these plot ranges according to your data
uvdist_amp_plotranges = {'ACA':[0,60,0,0.22], #xmin,xmax,ymin,ymax (x=uvdist, y=amp)
                         'SB':[0,500,0,0.25],
                         'LB':[0,3500,0,0.22]}


for params in data_params.values():
    plotms(vis=params['vis'],
            xaxis='channel',
            yaxis='amplitude',
            field=params['field'],
            ydatacolumn='data',
            avgtime='1e8',
            avgscan=True,
            avgbaseline=True,
            #There is a bug in plotms that plots the data wrongly if there is unequal flagging
            #between the polarisations
            #the bug occurs if we plot amp and we average over baselines
            #thus we color by polarization such that
            #we easily identify this issue
            #if you find issues, then try plotting XX and YY polarisation separately
            #to verify that it is only a plotms bug and not a problem with the data
            coloraxis='corr',
            iteraxis='spw',
            showgui = False,
            exprange='all',
            plotfile=os.path.join(preselfcal_amp_figures_folder,
                                  prefix+'_'+params['name']+'_chan-v-amp_preselfcal.png'))
    baseline_key = params['name'].split('_')[0]
    plotms(vis=params['vis'],
            xaxis='UVdist',
            yaxis='amplitude',
            spw='1',
            field=params['field'],
            ydatacolumn='data',
            avgtime='1e8',
            avgscan=True,
            avgchannel='3840',
            showgui = False,
            plotrange=uvdist_amp_plotranges[baseline_key],
            plotfile=os.path.join(preselfcal_amp_figures_folder,
                                  prefix+'_'+params['name']+'_uvdist-v-amp_cont_spw_preselfcal.png'))

'''
In the uvdist-v-amp plot of SB EB2 there are lower amplitudes apparently coming from the same antenna DA58.
However, it seems that it is just due to high decoherence, so I will not proceed with flagging this antenna and fix decoherence with self-calibration. 
'''

for params in data_params.values():
    flagchannels_string = get_flagchannels(ms_dict=params,output_prefix=prefix,
                                           velocity_range = np.array([-15., 15.])+v_sys)
    avg_cont(ms_dict=params, output_prefix=prefix, flagchannels = flagchannels_string,
             contspws = params['cont_spws'], width_array=params['width_array'])

"""
Double-check that the channels idenfified are at the center of the spws,
due to a potential issue with the data_desc_id key in the ms table of some programs
"""

# Flagchannels input string for LB_EB0: '0:837~3005, 2:793~3042, 3:788~3056'
# Flagchannels input string for LB_EB1: '0:837~3005, 2:793~3042, 3:788~3056'
# Flagchannels input string for LB_EB2: '0:837~3005, 2:793~3042, 3:788~3056'
# Flagchannels input string for LB_EB3: '0:837~3005, 2:793~3042, 3:788~3056'
# Flagchannels input string for SB_EB0: '0:838~3006, 2:792~3041, 3:791~3059'
# Flagchannels input string for SB_EB1: '0:838~3006, 2:792~3041, 3:791~3059'
# Flagchannels input string for SB_EB2: '0:838~3006, 2:792~3041, 3:791~3059'
# Flagchannels input string for ACA_EB0: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB1: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB2: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB3: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB4: '0:963~3131, 2:925~3174, 3:919~3187'

preselfcal_initcont_amp_folder = get_figures_folderpath(
                                  '2_preselfcal_initcont_amp_figures')
make_figures_folder(preselfcal_initcont_amp_folder)

uv_ranges = {'LB':'125~150m','SB':'60~80m','ACA':'0~50m'}
# This range for SB resulted in more informative plots (waterfalls were seen in this way)

for params in data_params.values():
    vis = prefix+'_'+params['name']+'_initcont.ms'
    baseline_key = params['name'].split('_')[0]
    plotms(vis=vis,
           xaxis='UVdist',
           overwrite=True,
           yaxis='amp',
           coloraxis='spw',
           avgtime='1e8',
           avgscan=True,
           showgui=False,
           plotrange=uvdist_amp_plotranges[baseline_key],
           plotfile=os.path.join(preselfcal_initcont_amp_folder,
                                 prefix+'_'+params['name']+'_uvdist-v-amp_initcont_preselfcal.png'))

    plotms(vis=vis,
            xaxis='time',
            yaxis='amp',
            avgspw=True,
            uvrange=uv_ranges[baseline_key],
            avgchannel='10000',
            avgbaseline=True,
            coloraxis='corr',
            showgui=False,
            overwrite=True,
            plotfile=os.path.join(preselfcal_initcont_amp_folder,
                                  prefix+'_'+params['name']+'_amp_v_time_initcont_preselfcal.png')
            )


for params in data_params_ACA.values():
    vis = prefix+'_'+params['name']+'_initcont.ms'
    baseline_key = params['name'].split('_')[0]
    plotms(vis=vis,
           xaxis='UVdist',
           overwrite=True,
           yaxis='amp',
           coloraxis='spw',
           avgtime='1e8',
           avgscan=True,
           showgui=False,
           plotrange=[0,50,0,0.30],
           plotfile=os.path.join(preselfcal_initcont_amp_folder,
                                 prefix+'_'+params['name']+'_uvdist-v-amp_initcont_preselfcal.png'))

""" Define simple masks and clean scales for imaging """
mask_pa = PA #position angle of mask in degrees
mask_semimajor = 3 #semimajor axis of mask in arcsec
mask_semiminor = mask_semimajor*np.cos(incl/180.*np.pi) #semiminor axis of mask in arcsec
mask_ra = '04h34m55.428s'     # from listobs
mask_dec = '+24.28.52.580'

mask_TM = f'ellipse[[{mask_ra},{mask_dec}], [{mask_semimajor:.3f}arcsec, {mask_semiminor:.3f}arcsec], {mask_pa:.1f}deg]'
# Cellsize: ~beam/6-7
LB_cellsize =  '0.015arcsec'
# Image size: ~primary beam 1.22*lam/A = 32'' with A=12m (19 arcsec)
LB_imsize = 1200 # primary beam
LB_scales = [0,8,15,30,80]

SB_cellsize =  '0.040arcsec'
SB_imsize = 400 # primary beam
SB_scales = [0,8,15,30]

noise_annulus_TM = f"annulus[[{mask_ra}, {mask_dec}],['4.arcsec', '6.arcsec']]"

mask_semimajor_ACA = 7. #semimajor axis of mask in arcsec
mask_semiminor_ACA = 7. #semiminor axis of mask in arcsec
mask_ACA = f'ellipse[[{mask_ra},{mask_dec}], [{mask_semimajor_ACA:.3f}arcsec, {mask_semiminor_ACA:.3f}arcsec], {mask_pa:.1f}deg]'
# Cellsize: ~beam/6-7, synthesized beam: ~3.3-5'' (ACA)
ACA_cellsize =  '0.5arcsec'
# Image size: ~primary beam 1.22*lam/A = 32'' with A=7m
ACA_imsize = 100 # primary beam

noise_annulus_ACA = f"annulus[[{mask_ra}, {mask_dec}],['10.arcsec', '18.arcsec']]"

#sizes in [arcsec] of the zoomed and overview plots of the output pngs
image_png_plot_sizes = {'LB':[3,10],'ACA':[10,30]}
image_png_plot_sizes['SB'] = image_png_plot_sizes['LB']

preselfcal_images_folder = get_figures_folderpath('3_preselfcal_images')
make_figures_folder(preselfcal_images_folder)

for EB_key,params in data_params_LB.items():
    imagename = prefix+'_'+params['name']+'_initcont_image'
    #clean down to 6 sigma
    tclean_wrapper(
                 vis=prefix+'_'+params['name']+'_initcont.ms',
                 imagename=imagename,
                 deconvolver='multiscale',
                 scales=LB_scales,
                 mask=mask_TM,
                 threshold='0.5mJy',
                 cellsize=LB_cellsize,
                 imsize=LB_imsize,
                 parallel=use_parallel,
                 savemodel = 'modelcolumn',
                )
    estimate_SNR(f'{imagename}.image',disk_mask=mask_TM,noise_mask=noise_annulus_TM)
    rms = imstat(imagename = f'{imagename}.image', region = noise_annulus_TM)['rms'][0]
    data_params_LB[EB_key]['rms'] = rms
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['LB'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=preselfcal_images_folder,
                       mask=f'{imagename}.mask',noise_annulus=noise_annulus_TM)

#AA_Tau_LB_EB0_initcont_image.image
#Beam 0.094 arcsec x 0.069 arcsec (23.02 deg)
#Flux inside disk mask: 181.35 mJy
#Peak intensity of source: 3.37 mJy/beam
#rms: 6.95e-02 mJy/beam
#Peak SNR: 48.39

#AA_Tau_LB_EB1_initcont_image.image
#Beam 0.089 arcsec x 0.069 arcsec (-12.47 deg)
#Flux inside disk mask: 161.35 mJy
#Peak intensity of source: 2.73 mJy/beam
#rms: 8.72e-02 mJy/beam
#Peak SNR: 31.32
 
#AA_Tau_LB_EB2_initcont_image.image
#Beam 0.108 arcsec x 0.066 arcsec (-38.23 deg)
#Flux inside disk mask: 176.10 mJy
#Peak intensity of source: 3.33 mJy/beam
#rms: 7.33e-02 mJy/beam
#Peak SNR: 45.37

#AA_Tau_LB_EB3_initcont_image.image
#Beam 0.097 arcsec x 0.068 arcsec (-23.19 deg)
#Flux inside disk mask: 157.15 mJy
#Peak intensity of source: 2.84 mJy/beam
#rms: 8.37e-02 mJy/beam
#Peak SNR: 33.96


for EB_key,params in data_params_SB.items():
    imagename = prefix+'_'+params['name']+'_initcont_image'
    #clean down to 6 sigma
    tclean_wrapper(
                 vis=prefix+'_'+params['name']+'_initcont.ms',
                 imagename=imagename,
                 deconvolver='multiscale',
                 scales=SB_scales,
                 mask=mask_TM,
                 threshold='6mJy',
                 cellsize=SB_cellsize,
                 imsize=SB_imsize,
                 parallel=use_parallel,
                 savemodel = 'modelcolumn',
                )
    estimate_SNR(f'{imagename}.image',disk_mask=mask_TM,noise_mask=noise_annulus_TM)
    rms = imstat(imagename = f'{imagename}.image', region = noise_annulus_TM)['rms'][0]
    data_params_SB[EB_key]['rms'] = rms
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['SB'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=preselfcal_images_folder,
                       mask=f'{imagename}.mask',noise_annulus=noise_annulus_TM)

#AA_Tau_SB_EB0_initcont_image.image
#Beam 0.687 arcsec x 0.442 arcsec (-40.41 deg)
#Flux inside disk mask: 170.01 mJy
#Peak intensity of source: 50.94 mJy/beam
#rms: 4.83e-01 mJy/beam
#Peak SNR: 105.49

#AA_Tau_SB_EB1_initcont_image.image
#Beam 0.582 arcsec x 0.476 arcsec (18.62 deg)
#Flux inside disk mask: 177.65 mJy
#Peak intensity of source: 45.17 mJy/beam
#rms: 5.48e-01 mJy/beam
#Peak SNR: 82.37

#AA_Tau_SB_EB2_initcont_image.image
#Beam 0.692 arcsec x 0.513 arcsec (-49.24 deg)
#Flux inside disk mask: 142.96 mJy
#Peak intensity of source: 43.15 mJy/beam
#rms: 5.84e-01 mJy/beam
#Peak SNR: 73.93


for EB_key,params in data_params_ACA.items():
    imagename = prefix+'_'+params['name']+'_initcont_image'
    #clean down to 6 sigma
    tclean_wrapper(
                 vis=prefix+'_'+params['name']+'_initcont.ms',
                 imagename=imagename,
                 deconvolver='hogbom',
                 mask=mask_ACA,
                 threshold='50mJy',
                 cellsize=ACA_cellsize,
                 imsize=ACA_imsize,
                 parallel=use_parallel,
                 savemodel = 'modelcolumn',
                )
    estimate_SNR(f'{imagename}.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
    rms = imstat(imagename = f'{imagename}.image', region = noise_annulus_ACA)['rms'][0]
    data_params_ACA[EB_key]['rms'] = rms
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['ACA'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=preselfcal_images_folder,
                       mask=f'{imagename}.mask',noise_annulus=noise_annulus_ACA)

#AA_Tau_ACA_EB0_initcont_image.image
#Beam 4.999 arcsec x 3.980 arcsec (-38.36 deg)
#Flux inside disk mask: 152.73 mJy
#Peak intensity of source: 167.90 mJy/beam
#rms: 7.00e+00 mJy/beam
#Peak SNR: 23.99

#AA_Tau_ACA_EB1_initcont_image.image
#Beam 6.055 arcsec x 3.498 arcsec (-41.15 deg)
#Flux inside disk mask: 152.15 mJy
#Peak intensity of source: 170.94 mJy/beam
#rms: 6.06e+00 mJy/beam
#Peak SNR: 28.20

#AA_Tau_ACA_EB2_initcont_image.image
#Beam 7.933 arcsec x 2.930 arcsec (-48.84 deg)
#Flux inside disk mask: 148.46 mJy
#Peak intensity of source: 165.09 mJy/beam
#rms: 7.22e+00 mJy/beam
#Peak SNR: 22.87

#AA_Tau_ACA_EB3_initcont_image.image
#Beam 5.167 arcsec x 3.826 arcsec (-42.28 deg)
#Flux inside disk mask: 149.61 mJy
#Peak intensity of source: 169.41 mJy/beam
#rms: 6.98e+00 mJy/beam
#Peak SNR: 24.27

#AA_Tau_ACA_EB4_initcont_image.image
#Beam 5.317 arcsec x 3.742 arcsec (-43.79 deg)
#Flux inside disk mask: 153.39 mJy
#Peak intensity of source: 172.20 mJy/beam
#rms: 7.08e+00 mJy/beam
#Peak SNR: 24.33



######
# SELF-CAL INDIVIDUAL EBs
######
""" Self-calibration parameters """
single_EB_contspws = '0~3'

single_EB_spw_mapping = [0,0,0,0]

EB_selfcal_shift_folder = get_figures_folderpath(
                 '4_individual_EB_selfcal_and_shift_figures')
make_figures_folder(EB_selfcal_shift_folder)


""" One round of phase-only self-cal """
for params in data_params.values():
    single_EB_p1 = prefix+'_'+params['name']+'_initcont.p1'
    os.system('rm -rf '+single_EB_p1)
    gaincal(vis=prefix+'_'+params['name']+'_initcont.ms',caltable=single_EB_p1,
        gaintype='T', spw=single_EB_contspws,combine='scan,spw',
        calmode='p', solint='inf', minsnr=4., minblperant=3)
# 1 of 36 solutions flagged due to SNR < 4 in spw=0 at 2022/09/21/10:27:58.5


    """ Print calibration png file """
    plotms(single_EB_p1,
           xaxis='time',
           yaxis='GainPhase',
           overwrite=True,
           showgui=False,
           plotfile=os.path.join(EB_selfcal_shift_folder,
                                 prefix+'_'+params['name']+'_initcont_gain_p1_phase_vs_time.png'))

    """ Apply the solutions """
    applycal(vis=prefix+'_'+params['name']+'_initcont.ms', spw=single_EB_contspws,spwmap=single_EB_spw_mapping,
        gaintable=[single_EB_p1], interp='linearPD', applymode='calonly', calwt=True)
    split(vis=prefix+'_'+params['name']+'_initcont.ms',
          outputvis=prefix+'_'+params['name']+'_initcont_selfcal.ms',
          datacolumn='corrected')


### Image self-cal'd EBs ###

for params in data_params_LB.values():
    imagename = prefix+'_'+params['name']+'_initcont_selfcal_image'
    tclean_wrapper(
                 vis=prefix+'_'+params['name']+'_initcont_selfcal.ms',
                 imagename=imagename,
                 deconvolver='multiscale',
                 scales=LB_scales,
                 mask=mask_TM,
                 threshold='0.5mJy',
                 cellsize=LB_cellsize,
                 imsize=LB_imsize,
                 parallel=use_parallel,
                )
    estimate_SNR(f'{imagename}.image',disk_mask=mask_TM,noise_mask=noise_annulus_TM)
    rms = params['rms']
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['LB'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=EB_selfcal_shift_folder)

#AA_Tau_LB_EB0_initcont_selfcal_image.image
#Beam 0.094 arcsec x 0.069 arcsec (23.02 deg)
#Flux inside disk mask: 182.83 mJy
#Peak intensity of source: 3.42 mJy/beam
#rms: 6.79e-02 mJy/beam
#Peak SNR: 50.38

#AA_Tau_LB_EB1_initcont_selfcal_image.image
#Beam 0.089 arcsec x 0.069 arcsec (-12.47 deg)
#Flux inside disk mask: 181.81 mJy
#Peak intensity of source: 3.10 mJy/beam
#rms: 6.80e-02 mJy/beam
#Peak SNR: 45.63 

#AA_Tau_LB_EB2_initcont_selfcal_image.image
#Beam 0.108 arcsec x 0.066 arcsec (-38.23 deg)
#Flux inside disk mask: 177.29 mJy
#Peak intensity of source: 3.32 mJy/beam
#rms: 7.18e-02 mJy/beam
#Peak SNR: 46.22

#AA_Tau_LB_EB3_initcont_selfcal_image.image
#Beam 0.097 arcsec x 0.068 arcsec (-23.19 deg)
#Flux inside disk mask: 178.25 mJy
#Peak intensity of source: 3.22 mJy/beam
#rms: 6.46e-02 mJy/beam
#Peak SNR: 49.87


# Compute intensity ratio to check position shifts between different LB EBs before
# applying alignment
default(immath)
for params in data_params_LB.values():
    ref_image = prefix+'_LB_EB0_initcont_selfcal_image.image'
    imagename = prefix+'_'+params['name']+'_initcont_selfcal_image'
    os.system('rm -rf '+imagename+'.ratio')
    immath(imagename=[ref_image,imagename+'.image'],mode='evalexpr',
           outfile=imagename+'.ratio',
           expr='iif(IM0 > 3*'+str(data_params_LB['LB0']['rms'])+', IM1/IM0, 0)'
           )
    generate_image_png(f'{imagename}.ratio',plot_sizes=[2*mask_semimajor,2*mask_semimajor],
                       color_scale_limits=[0.5,1.5],image_units='ratio',
                       save_folder=EB_selfcal_shift_folder)


for params in data_params_SB.values():
    imagename = prefix+'_'+params['name']+'_initcont_selfcal_image'
    tclean_wrapper(
                 vis=prefix+'_'+params['name']+'_initcont_selfcal.ms',
                 imagename=imagename,
                 deconvolver='multiscale',
                 scales=SB_scales,
                 mask=mask_TM,
                 threshold='1.2mJy',
                 cellsize=SB_cellsize,
                 imsize=SB_imsize,
                 parallel=use_parallel,
                )
    estimate_SNR(f'{imagename}.image',disk_mask=mask_TM,noise_mask=noise_annulus_TM)
    rms = params['rms']
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['SB'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=EB_selfcal_shift_folder)

#AA_Tau_SB_EB0_initcont_selfcal_image.image
#Beam 0.687 arcsec x 0.442 arcsec (-40.41 deg)
#Flux inside disk mask: 183.75 mJy
#Peak intensity of source: 53.38 mJy/beam
#rms: 1.56e-01 mJy/beam
#Peak SNR: 341.44

#AA_Tau_SB_EB1_initcont_selfcal_image.image
#Beam 0.582 arcsec x 0.476 arcsec (18.62 deg)
#Flux inside disk mask: 191.16 mJy
#Peak intensity of source: 47.35 mJy/beam
#rms: 1.82e-01 mJy/beam
#Peak SNR: 260.73

#AA_Tau_SB_EB2_initcont_selfcal_image.image
#Beam 0.692 arcsec x 0.513 arcsec (-49.24 deg)
#Flux inside disk mask: 160.61 mJy
#Peak intensity of source: 45.63 mJy/beam
#rms: 2.92e-01 mJy/beam
#Peak SNR: 156.22



## Compute intensity ratio to check position shifts between differnet SB EBs before applying alignment
default(immath)
for params in data_params_SB.values():
    ref_image = prefix+'_SB_EB0_initcont_selfcal_image.image'
    imagename = prefix+'_'+params['name']+'_initcont_selfcal_image'
    os.system('rm -rf '+imagename+'.ratio')
    immath(imagename=[ref_image,imagename+'.image'],mode='evalexpr',
           outfile=imagename+'.ratio',
           expr='iif(IM0 > 3*'+str(data_params_SB['SB0']['rms'])+', IM1/IM0, 0)')
    generate_image_png(f'{imagename}.ratio',
                       plot_sizes=[2*mask_semimajor,2*mask_semimajor],
                       color_scale_limits=[0.5,1.5],image_units='ratio',
                       save_folder=EB_selfcal_shift_folder)


for params in data_params_ACA.values():
    imagename = prefix+'_'+params['name']+'_initcont_selfcal_image'
    tclean_wrapper(
                 vis=prefix+'_'+params['name']+'_initcont_selfcal.ms',
                 imagename=imagename,
                 deconvolver='hogbom',
                 mask=mask_ACA,
                 threshold='30mJy',
                 cellsize=ACA_cellsize,
                 imsize=ACA_imsize,
                 parallel=use_parallel,
                )
    estimate_SNR(f'{imagename}.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
    rms = params['rms']
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['ACA'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=EB_selfcal_shift_folder)

#AA_Tau_ACA_EB0_initcont_selfcal_image.image
#Beam 4.998 arcsec x 3.980 arcsec (-38.37 deg)
#Flux inside disk mask: 165.42 mJy
#Peak intensity of source: 172.95 mJy/beam
#rms: 3.49e+00 mJy/beam
#Peak SNR: 49.54

#AA_Tau_ACA_EB1_initcont_selfcal_image.image
#Beam 6.055 arcsec x 3.497 arcsec (-41.15 deg)
#Flux inside disk mask: 164.75 mJy
#Peak intensity of source: 175.14 mJy/beam
#rms: 3.21e+00 mJy/beam
#Peak SNR: 54.55

#AA_Tau_ACA_EB2_initcont_selfcal_image.image
#Beam 7.933 arcsec x 2.930 arcsec (-48.84 deg)
#Flux inside disk mask: 157.78 mJy
#Peak intensity of source: 169.05 mJy/beam
#rms: 4.52e+00 mJy/beam
#Peak SNR: 37.39

#AA_Tau_ACA_EB3_initcont_selfcal_image.image
#Beam 5.167 arcsec x 3.826 arcsec (-42.28 deg)
#Flux inside disk mask: 168.26 mJy
#Peak intensity of source: 174.96 mJy/beam
#rms: 2.61e+00 mJy/beam
#Peak SNR: 67.11

#AA_Tau_ACA_EB4_initcont_selfcal_image.image
#Beam 5.316 arcsec x 3.742 arcsec (-43.79 deg)
#Flux inside disk mask: 171.78 mJy
#Peak intensity of source: 177.68 mJy/beam
#rms: 2.64e+00 mJy/beam
#Peak SNR: 67.24


## Compute intensity ratio to check position shifts between differnet ACA EBs before applying alignment
default(immath)
for params in data_params_ACA.values():
    ref_image = prefix+'_ACA_EB0_initcont_selfcal_image.image'
    imagename = prefix+'_'+params['name']+'_initcont_selfcal_image'
    os.system('rm -rf '+imagename+'.ratio')
    immath(imagename=[ref_image,imagename+'.image'],mode='evalexpr',
           outfile=imagename+'.ratio',
           expr='iif(IM0 > 3*'+str(data_params_ACA['ACA0']['rms'])+', IM1/IM0, 0)'
           )
    generate_image_png(f'{imagename}.ratio',
                       plot_sizes=[2*mask_semimajor_ACA,2*mask_semimajor_ACA],
                       color_scale_limits=[0.5,1.5],image_units='ratio',
                       save_folder=EB_selfcal_shift_folder)



######
# ALIGN DATA (go from *initcont_selfcal.ms to *initcont_shift.ms)
######

# Select the LB EB to act as the reference (usually the best SNR one).

reference_for_LB_alignment = f'{prefix}_LB_EB0_initcont_selfcal.ms'
assert 'LB' in reference_for_LB_alignment, 'you need to choose an LB EB for alignment of LB'

alignment_offsets = {}

# All the other EBs will be aligned to the reference_EB
# We also include the reference_EB itself to make sure the coordinate changes are
# copied over.

offset_LB_EBs = ['{}_{}_initcont_selfcal.ms'.format(prefix, params['name'])
                 for params in data_params_LB.values()]

#select the continuum spw with the large bandwidth
continuum_spw_id = 1
# Find the relative offsets and update the phase centers for all offset_EBs.
# cell_size defines the size of the uv grid
alignment_npix = {'LB':1024,'SB':102,'ACA':30}
alignment_cell_size = {'LB':0.01,'SB':0.1,'ACA':1.}
alignment_plot_file_template = os.path.join(EB_selfcal_shift_folder,
                                            'alignment_uv_grid.png')
alignment.align_measurement_sets(reference_ms=reference_for_LB_alignment,
                                 align_ms=offset_LB_EBs,npix=alignment_npix['LB'],
                                 cell_size=alignment_cell_size['LB'],
                                 spwid=continuum_spw_id,plot_uv_grid=True,
                                 plot_file_template=alignment_plot_file_template)

#New coordinates for AA_Tau_LB_EB0_initcont_selfcal.ms
#no shift, reference MS.

#New coordinates for AA_Tau_LB_EB1_initcont_selfcal.ms
#requires a shift of [-0.016481,0.012814]

#New coordinates for AA_Tau_LB_EB2_initcont_selfcal.ms
#requires a shift of [0.0042971,0.0037993]

#New coordinates for AA_Tau_LB_EB3_initcont_selfcal.ms
#requires a shift of [-0.017586,0.013933]


alignment_offsets['LB_EB0'] = [0.0, 0.0] #ref EB
alignment_offsets['LB_EB1'] = [-0.016481,0.012814]
alignment_offsets['LB_EB2'] = [0.0042971,0.0037993]
alignment_offsets['LB_EB3'] = [-0.017586,0.013933]


shifted_LB_EBs = [EB.replace('.ms','_shift.ms') for EB in offset_LB_EBs]

#to check if alignment worked, calculate shift again and verify that shifts are small
for shifted_ms in shifted_LB_EBs:
    if shifted_ms == reference_for_LB_alignment.replace('.ms','_shift.ms'):
        #skip the reference because the fitter will fail for unknown reason
        continue
    offset = alignment.find_offset(reference_ms=reference_for_LB_alignment,
                                   offset_ms=shifted_ms,npix=alignment_npix['LB'],
                                   cell_size=alignment_cell_size['LB'],
                                   spwid=continuum_spw_id)
    print(f'#offset for {shifted_ms}: ',offset)
#offset for AA_Tau_LB_EB1_initcont_selfcal_shift.ms:  [-0.00013227  0.00013935]
#offset for AA_Tau_LB_EB2_initcont_selfcal_shift.ms:  [2.42861341e-04 5.90266682e-05]
#offset for AA_Tau_LB_EB3_initcont_selfcal_shift.ms:  [1.27765975e-04 4.03988289e-05]


# Merge shifted LB EBs for aligning SB EBs
LB_concat_shifted = f'{prefix}_LB_concat_shifted.ms'
os.system(f'rm -rf {LB_concat_shifted}')
concat(vis=shifted_LB_EBs,concatvis=LB_concat_shifted,dirtol='0.1arcsec',
       copypointing=False)


# Align SB EBs to concat shifted LB EBs
reference_for_SB_alignment = LB_concat_shifted

offset_SB_EBs = ['{}_{}_initcont_selfcal.ms'.format(prefix, params['name'])
                 for params in data_params_SB.values()]

alignment.align_measurement_sets(reference_ms=reference_for_SB_alignment,
                                 align_ms=offset_SB_EBs,npix=alignment_npix['SB'],
                                 cell_size=alignment_cell_size['SB'],
                                 spwid=continuum_spw_id,plot_uv_grid=True,
                                 plot_file_template=alignment_plot_file_template)

#/users/pcurone/calibration_scripts/alignment.py:227: UserWarning: some data of AA_Tau_LB_concat_shifted.ms are outside your uv grid warnings.warn(f'some data of {base_ms} are outside your uv grid')

#New coordinates for AA_Tau_SB_EB0_initcont_selfcal.ms
#requires a shift of [0.008156,0.10176]

#New coordinates for AA_Tau_SB_EB1_initcont_selfcal.ms
#requires a shift of [0.0090143,-0.059566]

#New coordinates for AA_Tau_SB_EB2_initcont_selfcal.ms
#requires a shift of [0.047804,0.064352]


alignment_offsets['SB_EB0'] = [0.008156,0.10176]
alignment_offsets['SB_EB1'] = [0.0090143,-0.059566]
alignment_offsets['SB_EB2'] = [0.047804,0.064352]


shifted_SB_EBs = [EB.replace('.ms','_shift.ms') for EB in offset_SB_EBs]

#check by calculating offset again
for shifted_ms in shifted_SB_EBs:
    offset = alignment.find_offset(reference_ms=reference_for_SB_alignment,
                                   offset_ms=shifted_ms,npix=alignment_npix['SB'],
                                   cell_size=alignment_cell_size['SB'],
                                   spwid=continuum_spw_id)
    print(f'#offset for {shifted_ms}: ',offset)
#offset for AA_Tau_SB_EB0_initcont_selfcal_shift.ms:  [-0.00026976  0.00367753]
#offset for AA_Tau_SB_EB1_initcont_selfcal_shift.ms:  [-0.00200944 -0.00217079]
#offset for AA_Tau_SB_EB2_initcont_selfcal_shift.ms:  [0.00311672 0.00232623]


#additional test: compute the offset of the EB SBs to each other:
for shifted_ms in shifted_SB_EBs[1:]:
    offset = alignment.find_offset(reference_ms=shifted_SB_EBs[0],
                                   offset_ms=shifted_ms,npix=alignment_npix['SB'],
                                   cell_size=alignment_cell_size['SB'],
                                   spwid=continuum_spw_id)
    print(f'#offset for {shifted_ms} to SB EB0: ',offset)
#offset for AA_Tau_SB_EB1_initcont_selfcal_shift.ms to SB EB0:  [-0.00767385 -0.00270499]
#offset for AA_Tau_SB_EB2_initcont_selfcal_shift.ms to SB EB0:  [ 0.0045504  -0.00496163]


# Merge shifted SB EBs for aligning ACA EBs
SB_concat_shifted = prefix+'_SB_concat_shifted.ms'
os.system('rm -rf '+SB_concat_shifted)
concat(vis=shifted_SB_EBs,concatvis=SB_concat_shifted,dirtol='0.1arcsec',
       copypointing=False)

# Align ACA EBs to concat shifted SB EBs
reference_for_ACA_alignment = SB_concat_shifted

offset_ACA_EBs = ['{}_{}_initcont_selfcal.ms'.format(prefix, params['name'])
                  for params in data_params_ACA.values()]

alignment.align_measurement_sets(reference_ms=reference_for_ACA_alignment,
                                 align_ms=offset_ACA_EBs,npix=alignment_npix['ACA'],
                                 cell_size=alignment_cell_size['ACA'],
                                 spwid=continuum_spw_id,plot_uv_grid=True,
                                 plot_file_template=alignment_plot_file_template)
#/users/pcurone/calibration_scripts/alignment.py:227: UserWarning: some data of AA_Tau_SB_concat_shifted.ms are outside your uv grid warnings.warn(f'some data of {base_ms} are outside your uv grid')

#New coordinates for AA_Tau_ACA_EB0_initcont_selfcal.ms
#requires a shift of [-0.077447,0.010629]

#New coordinates for AA_Tau_ACA_EB1_initcont_selfcal.ms
#requires a shift of [0.022049,-0.036367]

#New coordinates for AA_Tau_ACA_EB2_initcont_selfcal.ms
#requires a shift of [-0.21914,0.29901]

#New coordinates for AA_Tau_ACA_EB3_initcont_selfcal.ms
#requires a shift of [-0.16025,0.01728]

#New coordinates for AA_Tau_ACA_EB4_initcont_selfcal.ms
#requires a shift of [-0.098011,0.027225]

alignment_offsets['ACA_EB0'] = [-0.077447,0.010629] 
alignment_offsets['ACA_EB1'] = [0.022049,-0.036367]
alignment_offsets['ACA_EB2'] = [-0.21914,0.29901]
alignment_offsets['ACA_EB3'] = [-0.16025,0.01728]
alignment_offsets['ACA_EB4'] = [-0.098011,0.027225]

shifted_ACA_EBs = [EB.replace('.ms','_shift.ms') for EB in offset_ACA_EBs]

#check by calculating offset again
for shifted_ms in shifted_ACA_EBs:
    offset = alignment.find_offset(reference_ms=reference_for_ACA_alignment,
                                   offset_ms=shifted_ms,npix=alignment_npix['ACA'],
                                   cell_size=alignment_cell_size['ACA'],
                                   spwid=continuum_spw_id)
    print(f'#offset for {shifted_ms}: ',offset)
#offset for AA_Tau_ACA_EB0_initcont_selfcal_shift.ms:  [ 0.00328032 -0.0054514 ]
#offset for AA_Tau_ACA_EB1_initcont_selfcal_shift.ms:  [-0.00038474 -0.0019471 ]
#offset for AA_Tau_ACA_EB2_initcont_selfcal_shift.ms:  [-0.02595112  0.04287655]
#offset for AA_Tau_ACA_EB3_initcont_selfcal_shift.ms:  [-0.00458255  0.00176909]
#offset for AA_Tau_ACA_EB4_initcont_selfcal_shift.ms:  [ 0.00279667 -0.00465822]


#additional test: compute the offset of the EBs to each other:
for shifted_ms in shifted_ACA_EBs[1:]:
    offset = alignment.find_offset(reference_ms=shifted_ACA_EBs[0],
                                   offset_ms=shifted_ms,npix=alignment_npix['ACA'],
                                   cell_size=alignment_cell_size['ACA'],
                                   spwid=continuum_spw_id)
    print(f'#offset for {shifted_ms} to ACA EB0: ',offset)
#offset for AA_Tau_ACA_EB1_initcont_selfcal_shift.ms to ACA EB0:  [-0.01736151  0.01408283]
#offset for AA_Tau_ACA_EB2_initcont_selfcal_shift.ms to ACA EB0:  [-0.05361698  0.09002077]
#offset for AA_Tau_ACA_EB3_initcont_selfcal_shift.ms to ACA EB0:  [ 0.00714887 -0.01608356]
#offset for AA_Tau_ACA_EB4_initcont_selfcal_shift.ms to ACA EB0:  [-0.00519453 -0.00914492]


# Remove the '_selfcal' part of the names to match the naming convention below.
for shifted_EB in shifted_LB_EBs+shifted_SB_EBs+shifted_ACA_EBs:
    os.system('mv {} {}'.format(shifted_EB, shifted_EB.replace('_selfcal', '')))



""" Check that the images are indeed aligned after the shift """
for params in data_params_LB.values():
    imagename = prefix+'_'+params['name']+'_initcont_shift_image'
    tclean_wrapper(
                   vis=prefix+'_'+params['name']+'_initcont_shift.ms',
                   imagename=imagename,
                   deconvolver='multiscale',
                   scales=LB_scales,
                   mask=mask_TM,
                   threshold='0.5mJy',
                   cellsize=LB_cellsize,
                   imsize=LB_imsize,
                   parallel=use_parallel,
                  )
    estimate_SNR(f'{imagename}.image',disk_mask=mask_TM,
                 noise_mask=noise_annulus_TM)
    rms = params['rms']
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['LB'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=EB_selfcal_shift_folder)

#AA_Tau_LB_EB0_initcont_shift_image.image
#Beam 0.094 arcsec x 0.069 arcsec (23.02 deg)
#Flux inside disk mask: 185.70 mJy
#Peak intensity of source: 3.40 mJy/beam
#rms: 6.80e-02 mJy/beam
#Peak SNR: 49.96

#AA_Tau_LB_EB1_initcont_shift_image.image
#Beam 0.089 arcsec x 0.069 arcsec (-12.47 deg)
#Flux inside disk mask: 182.80 mJy
#Peak intensity of source: 3.07 mJy/beam
#rms: 6.80e-02 mJy/beam
#Peak SNR: 45.18

#AA_Tau_LB_EB2_initcont_shift_image.image
#Beam 0.108 arcsec x 0.066 arcsec (-38.23 deg)
#Flux inside disk mask: 176.99 mJy
#Peak intensity of source: 3.31 mJy/beam
#rms: 7.19e-02 mJy/beam
#Peak SNR: 46.11

#AA_Tau_LB_EB3_initcont_shift_image.image
#Beam 0.097 arcsec x 0.068 arcsec (-23.19 deg)
#Flux inside disk mask: 178.60 mJy
#Peak intensity of source: 3.24 mJy/beam
#rms: 6.47e-02 mJy/beam
#Peak SNR: 50.08


## Compute intensity ratio between aligned images from LB EBs
default(immath)
for params in data_params_LB.values():
    ref_image = prefix+'_LB_EB0_initcont_shift_image.image'
    imagename = prefix+'_'+params['name']+'_initcont_shift_image'
    os.system('rm -rf '+imagename+'.ratio')
    immath(imagename=[ref_image,imagename+'.image'],mode='evalexpr',
           outfile=imagename+'.ratio',
           expr='iif(IM0 > 3*'+str(data_params_LB['LB0']['rms'])+', IM1/IM0, 0)'
           )
    generate_image_png(f'{imagename}.ratio',
                       plot_sizes=[2*mask_semimajor,2*mask_semimajor],
                       color_scale_limits=[0.5,1.5],image_units='ratio',
                       save_folder=EB_selfcal_shift_folder)

for params in data_params_SB.values():
    imagename = prefix+'_'+params['name']+'_initcont_shift_image'
    tclean_wrapper(
                   vis=prefix+'_'+params['name']+'_initcont_shift.ms',
                   imagename=imagename,
                   deconvolver='multiscale',
                   scales=SB_scales,
                   mask=mask_TM,
                   threshold='1.2mJy',
                   cellsize=SB_cellsize,
                   imsize=SB_imsize,
                   parallel=use_parallel,
                  )
    estimate_SNR(f'{imagename}.image',disk_mask=mask_TM,
                 noise_mask=noise_annulus_TM)
    rms = params['rms']
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['SB'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=EB_selfcal_shift_folder)

#AA_Tau_SB_EB0_initcont_shift_image.image
#Beam 0.687 arcsec x 0.442 arcsec (-40.41 deg)
#Flux inside disk mask: 183.66 mJy
#Peak intensity of source: 53.38 mJy/beam
#rms: 1.58e-01 mJy/beam
#Peak SNR: 338.31 

#AA_Tau_SB_EB1_initcont_shift_image.image
#Beam 0.582 arcsec x 0.476 arcsec (18.62 deg)
#Flux inside disk mask: 191.07 mJy
#Peak intensity of source: 47.31 mJy/beam
#rms: 1.83e-01 mJy/beam
#Peak SNR: 258.01

#AA_Tau_SB_EB2_initcont_shift_image.image
#Beam 0.692 arcsec x 0.513 arcsec (-49.24 deg)
#Flux inside disk mask: 160.49 mJy
#Peak intensity of source: 45.63 mJy/beam
#rms: 2.92e-01 mJy/beam
#Peak SNR: 156.45

## Compute intensity ratio between aligned images from SB EBs
default(immath)
for params in data_params_SB.values():
    ref_image = prefix+'_SB_EB0_initcont_shift_image.image'
    imagename = prefix+'_'+params['name']+'_initcont_shift_image'
    os.system('rm -rf '+imagename+'.ratio')
    immath(imagename=[ref_image,imagename+'.image'],mode='evalexpr',
           outfile=imagename+'.ratio',
           expr='iif(IM0 > 3*'+str(data_params_SB['SB0']['rms'])+', IM1/IM0, 0)'
           )
    generate_image_png(f'{imagename}.ratio',
                       plot_sizes=[2*mask_semimajor,2*mask_semimajor],
                       color_scale_limits=[0.5,1.5],image_units='ratio',
                       save_folder=EB_selfcal_shift_folder)


for params in data_params_ACA.values():
    imagename = prefix+'_'+params['name']+'_initcont_shift_image'
    tclean_wrapper(
                 vis=prefix+'_'+params['name']+'_initcont_shift.ms',
                 imagename=imagename,
                 deconvolver='hogbom',
                 mask=mask_ACA,
                 threshold='30mJy',
                 cellsize=ACA_cellsize,
                 imsize=ACA_imsize,
                 parallel=use_parallel,
                )
    estimate_SNR(f'{imagename}.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
    rms = params['rms']
    generate_image_png(f'{imagename}.image',plot_sizes=image_png_plot_sizes['ACA'],
                       color_scale_limits=[-3*rms,10*rms],
                       save_folder=EB_selfcal_shift_folder)

#AA_Tau_ACA_EB0_initcont_shift_image.image
#Beam 4.998 arcsec x 3.980 arcsec (-38.37 deg)
#Flux inside disk mask: 165.36 mJy
#Peak intensity of source: 173.09 mJy/beam
#rms: 3.44e+00 mJy/beam
#Peak SNR: 50.28

#AA_Tau_ACA_EB1_initcont_shift_image.image
#Beam 6.055 arcsec x 3.497 arcsec (-41.15 deg)
#Flux inside disk mask: 164.66 mJy
#Peak intensity of source: 175.09 mJy/beam
#rms: 3.24e+00 mJy/beam
#Peak SNR: 54.09

#AA_Tau_ACA_EB2_initcont_shift_image.image
#Beam 7.933 arcsec x 2.930 arcsec (-48.84 deg)
#Flux inside disk mask: 158.06 mJy
#Peak intensity of source: 169.75 mJy/beam
#rms: 4.48e+00 mJy/beam
#Peak SNR: 37.93

#AA_Tau_ACA_EB3_initcont_shift_image.image
#Beam 5.167 arcsec x 3.826 arcsec (-42.28 deg)
#Flux inside disk mask: 163.19 mJy
#Peak intensity of source: 175.50 mJy/beam
#rms: 3.41e+00 mJy/beam
#Peak SNR: 51.50

#AA_Tau_ACA_EB4_initcont_shift_image.image
#Beam 5.316 arcsec x 3.742 arcsec (-43.79 deg)
#Flux inside disk mask: 166.71 mJy
#Peak intensity of source: 177.90 mJy/beam
#rms: 3.47e+00 mJy/beam
#Peak SNR: 51.32


## Compute intensity ratio between aligned images from ACA EBs
default(immath)
for params in data_params_ACA.values():
    ref_image = prefix+'_ACA_EB0_initcont_shift_image.image'
    imagename = prefix+'_'+params['name']+'_initcont_shift_image'
    os.system('rm -rf '+imagename+'.ratio')
    immath(imagename=[ref_image,imagename+'.image'],mode='evalexpr',
           outfile=imagename+'.ratio',
           expr='iif(IM0 > 3*'+str(data_params_ACA['ACA0']['rms'])+', IM1/IM0, 0)'
           )
    generate_image_png(f'{imagename}.ratio',
                       plot_sizes=[2*mask_semimajor_ACA,2*mask_semimajor_ACA],
                       color_scale_limits=[0.5,1.5],image_units='ratio',
                       save_folder=EB_selfcal_shift_folder)


# The alignment worked also by looking directly at the images


for params in data_params.values():
    listobs(
        vis=prefix+'_'+params['name']+'_initcont_shift.ms',
        listfile=prefix+'_'+params['name']+'_initcont_shift.ms.txt',
        overwrite=True,
    )




""" Now that everything is aligned, we inspect the flux calibration. """

for params in data_params.values():
    msfile = prefix+'_'+params['name']+'_initcont_shift.ms'
    export_MS(msfile) #export MS contents into Numpy save files


""" Plot deprojected visibility profiles for all data together """
list_npz_files = []
for baseline_key,n_EB in number_of_EBs.items():
    list_npz_files += [f'{prefix}_{baseline_key}_EB{i}_initcont_shift.vis.npz'
                       for i in range(n_EB)]

deprojected_vis_profiles_folder = get_figures_folderpath('5_deprojected_vis_profiles')
make_figures_folder(deprojected_vis_profiles_folder)

plot_deprojected(filelist=list_npz_files,
                 fluxscale=[1.]*(number_of_EBs['LB']+number_of_EBs['SB']+number_of_EBs['ACA']),PA=PA,incl=incl,show_err=True,
                 plot_label=os.path.join(deprojected_vis_profiles_folder,
                                         prefix+'_flux_scale_EB_preselfcal.png'))


flux_comparison_folder = get_figures_folderpath('6_flux_comparisons')
make_figures_folder(flux_comparison_folder)

"""

SB has overlapping uv ranges with both ACA and LB, so
we select an SB EB as the reference for flux scaling.

In case this source does not have ACA data, simply ignore the ACA steps in the list below.
 
If all SB EBs suffer from decoherence and cannot be used, see below

We correct flux differences >4%, if phase noise looks reasonable

If you need to re-scale fluxes, use command:
rescale_flux(vis=prefix+'_LB_EB0_initcont_shift.ms', gencalparameter=[1.044])

The general procedure is as follows:
1) check flux scaling, and if flux differences are >4%, re-scale the fluxes unless
    there is clear de-coherence (e.g. seen as a systematic decrease of scaling with UV distance)
2) do not scale those EBs with phase decoherence. Proceed with self-cal following the steps below 
   to the point that those EBs are self calibrated
3) check flux scaling again
4) if no flux scaling has been applied in 1), and you now still see a flux offset, then
    apply the flux scaling determined in step 3) to the non-self caled EBs
    and repeat the self-cal

If all SB EBs suffer from phase decoherence and cannot be used as the reference
for flux scaling:
1) concat ACA data and self-cal them in phase
2) concat them to SB data and self-cal them in phase
3) check flux offsets
4) if there are flux offsets, go back to 1), but before re-starting the process apply
    the corrections to the non-self caled ACA and SB data, and re-do steps 1) - 3).
    For this you should add additional code (essentially copy/paste the code from the first
    iteration) that repeats steps 1-3, but saves all output with different filenames. Remember
    to use the right filenames when you continue the script after step 6
5) Check flux offsets of LB EBs to a correct SB EB
6) concat LB data to ACA+SB and continue with script

"""


flux_ref_EB = 'SB_EB0'

#if you had to choose an LB EB, then iterate over data_params_SBLB constructed like this:
#data_params_SBLB = data_params_LB.copy()
#data_params_SBLB.update(data_params_SB)
#in that case, the call to estimate_flux_scale is just for exploring purposes by looking at the
#figures. in particular, the derived flux scalings are not used anywhere. this is because
#we need an SB EB to derive flux scalings, because only SB has overlapping baselines
#with both LB and ACA

# having add 'plot.ylim(0, 2.5)' to estimate_flux_scale.py 
for params in data_params.values():
    estimate_flux_scale(reference=prefix+f'_{flux_ref_EB}_initcont_shift.vis.npz',
                        comparison=prefix+'_'+params['name']+'_initcont_shift.vis.npz',
                        incl=incl, PA=PA,
                        plot_label=os.path.join(flux_comparison_folder,
                                                'flux_comparison_'+params['name']+f'_to_{flux_ref_EB}.png'))


#The ratio of the fluxes of AA_Tau_LB_EB0_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.09244
#The scaling factor for gencal is 1.045 for your comparison measurement
#The error on the weighted mean ratio is 1.670e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB1_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.09521
#The scaling factor for gencal is 1.047 for your comparison measurement
#The error on the weighted mean ratio is 1.570e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB2_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.08174
#The scaling factor for gencal is 1.040 for your comparison measurement
#The error on the weighted mean ratio is 1.597e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB3_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.07260
#The scaling factor for gencal is 1.036 for your comparison measurement
#The error on the weighted mean ratio is 1.422e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_SB_EB0_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.00000
#The scaling factor for gencal is 1.000 for your comparison measurement
#The error on the weighted mean ratio is 8.391e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_SB_EB1_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.97654
#The scaling factor for gencal is 0.988 for your comparison measurement
#The error on the weighted mean ratio is 8.438e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_SB_EB2_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.78182
#The scaling factor for gencal is 0.884 for your comparison measurement
#The error on the weighted mean ratio is 6.524e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_EB0_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.97368
#The scaling factor for gencal is 0.987 for your comparison measurement
#The error on the weighted mean ratio is 3.394e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_EB1_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.99224
#The scaling factor for gencal is 0.996 for your comparison measurement
#The error on the weighted mean ratio is 3.658e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_EB2_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.95648
#The scaling factor for gencal is 0.978 for your comparison measurement
#The error on the weighted mean ratio is 5.131e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_EB3_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.98953
#The scaling factor for gencal is 0.995 for your comparison measurement
#The error on the weighted mean ratio is 3.958e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_EB4_initcont_shift.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.99720
#The scaling factor for gencal is 0.999 for your comparison measurement
#The error on the weighted mean ratio is 3.571e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor



# Trying with an LB EB
flux_ref_EB = 'LB_EB0'
for params in data_params.values():
    estimate_flux_scale(reference=prefix+f'_{flux_ref_EB}_initcont_shift.vis.npz',
                        comparison=prefix+'_'+params['name']+'_initcont_shift.vis.npz',
                        incl=incl, PA=PA,
                        plot_label=os.path.join(flux_comparison_folder,
                                                'flux_comparison_'+params['name']+f'_to_{flux_ref_EB}.png'))

'''
I see 'downward trends' in the flux ratios (sign of decoherence) only when SB EBs are involved. Ratios only between LB EBs do not show any clear trend. 
Therefore, the flux offset will have to be evaluated only after the phase self-cal of ACA and ACA+SB. 
'''


########## BEGIN of ACA and ACA+SB self-cal iteration 1 ##########

"""
Self-cal of ACA data only
"""
#for phase self-cal, we clean down to 6 sigma

ACA_selfcal_folder = get_figures_folderpath('7_selfcal_ACA_figures')
make_figures_folder(ACA_selfcal_folder)

""" Merge the ACA executions back into a single MS """
ACA_cont_p0 = prefix+'_ACA_contp0'
os.system('rm -rf %s.ms*' % ACA_cont_p0)
concat(vis=[f'{prefix}_ACA_EB{i}_initcont_shift.ms' for i in range(number_of_EBs['ACA'])],
       concatvis=ACA_cont_p0+'.ms',dirtol='0.1arcsec',copypointing=False)

listobs(
    vis=ACA_cont_p0+'.ms',
    listfile=ACA_cont_p0+'.ms.txt',
    overwrite=True,
)

"""Define mask for ACA clean, using the new center read from listobs J2000 04h34m55.429800s +24.28.52.56450"""
mask_ra = '04h34m55.429800s'
mask_dec = '+24.28.52.56450'
mask_semimajor_ACA = 9. #semimajor axis of mask in arcsec
mask_semiminor_ACA = 9. #semiminor axis of mask in arcsec
mask_ACA = f'ellipse[[{mask_ra},{mask_dec}], [{mask_semimajor_ACA:.3f}arcsec, {mask_semiminor_ACA:.3f}arcsec], {mask_pa:.1f}deg]'
# Cellsize: ~beam/6-7, synthesized beam: ~3.3-5'' (ACA)
ACA_cellsize =  '0.5arcsec'
# Image size: ~primary beam 1.22*lam/A = 32'' with A=7m
ACA_imsize = 100 # primary beam

noise_annulus_ACA = f"annulus[[{mask_ra}, {mask_dec}], ['10.arcsec', '18.arcsec']]"

""" Initial clean (down to 6 sigma)"""
tclean_wrapper(vis=ACA_cont_p0+'.ms', imagename = ACA_cont_p0, mask=mask_ACA, deconvolver='hogbom',
               threshold = '6mJy', savemodel = 'modelcolumn', imsize=ACA_imsize,
               cellsize=ACA_cellsize, robust=0.5, interactive=False, parallel=use_parallel)
estimate_SNR(ACA_cont_p0+'.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
#AA_Tau_ACA_contp0.image
#Beam 5.453 arcsec x 3.533 arcsec (-45.27 deg)
#Flux inside disk mask: 172.72 mJy
#Peak intensity of source: 173.96 mJy/beam
#rms: 9.16e-01 mJy/beam
#Peak SNR: 189.98
rms_ACA = imstat(imagename = ACA_cont_p0+'.image', region = noise_annulus_ACA)['rms'][0]
generate_image_png(ACA_cont_p0+'.image',plot_sizes=image_png_plot_sizes['ACA'],
                   color_scale_limits=[-3*rms_ACA,10*rms_ACA],
                   save_folder=ACA_selfcal_folder,mask=ACA_cont_p0+'.mask',
                   noise_annulus=noise_annulus_ACA)


""" Self-calibration parameters """

""" Look for references antennas from weblog, and pick the first that are listed, overlapping with all EBs """
# ACA0 Refant (from weblog): CM02, CM10, CM05, CM08, CM03, CM12, CM01, CM09, CM07, CM11
# ACA1 Refant (from weblog): CM02, CM10, CM05, CM08, CM03, CM12, CM01, CM09, CM07, CM11
# ACA2 Refant (from weblog): CM10, CM02, CM05, CM03, CM08, CM12, CM01, CM09, CM07, CM11
# ACA3 Refant (from weblog): CM02, CM10, CM05, CM08, CM03, CM12, CM01, CM09, CM07, CM11
# ACA4 Refant (from weblog): CM02, CM10, CM05, CM08, CM03, CM12, CM01, CM09, CM07, CM11
# I use CM02 for EB0, EB1, EB3, EB4, and CM10 for EB2
get_station_numbers(ACA_cont_p0+'.ms','CM02')
get_station_numbers(ACA_cont_p0+'.ms','CM10')
#Observation ID 0: CM02@J502
#Observation ID 1: CM02@J502
#Observation ID 3: CM02@J502
#Observation ID 4: CM02@J502
#Observation ID 0: CM10@J501
#Observation ID 1: CM10@J501
#Observation ID 2: CM10@J501
#Observation ID 3: CM10@J501
#Observation ID 4: CM10@J501

ACA_refant   = 'CM02@J502, CM10@J501'

ACA_contspws = '0~19'

ACA_spw_mapping = [0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,16,16,16,16]

""" First round of phase-only self-cal """
# Use scan-length interval to align SPWs as much as possible
ACA_p1 = prefix+'_ACA.p1'
os.system('rm -rf '+ACA_p1)
gaincal(vis=ACA_cont_p0+'.ms', caltable=ACA_p1, gaintype='G', spw=ACA_contspws,
        refant=ACA_refant, combine='scan', calmode='p', solint='inf', minsnr=3., minblperant=3)


""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(ACA_p1,xaxis='time', yaxis='GainPhase')
""" Print calibration png file """
plotms(ACA_p1,xaxis='time', yaxis='GainPhase',overwrite=True,showgui=False,
       plotfile=os.path.join(ACA_selfcal_folder,prefix+'_ACA_gain_p1_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=ACA_cont_p0+'.ms', spw=ACA_contspws, gaintable=[ACA_p1],
         interp='linear', calwt=True, applymode='calonly')

""" Split off a corrected MS """
ACA_cont_p1 = prefix+'_ACA_contp1'
os.system('rm -rf %s.ms*' % ACA_cont_p1)
split(vis=ACA_cont_p0+'.ms', outputvis=ACA_cont_p1+'.ms', datacolumn='corrected')


""" Image the results; check the resulting map """
#again clean down to 6 sigma
tclean_wrapper(vis=ACA_cont_p1+'.ms', imagename = ACA_cont_p1, mask=mask_ACA, threshold = '6mJy', savemodel = 'modelcolumn', imsize=ACA_imsize, cellsize=ACA_cellsize, robust=0.5, interactive=False, parallel=use_parallel)
estimate_SNR(ACA_cont_p1+'.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
generate_image_png(ACA_cont_p1+'.image',plot_sizes=image_png_plot_sizes['ACA'],
                   color_scale_limits=[-3*rms_ACA,10*rms_ACA],
                   save_folder=ACA_selfcal_folder)
#AA_Tau_ACA_contp1.image
#Beam 5.453 arcsec x 3.533 arcsec (-45.27 deg)
#Flux inside disk mask: 172.99 mJy
#Peak intensity of source: 174.20 mJy/beam
#rms: 9.15e-01 mJy/beam
#Peak SNR: 190.32

""" Second round of phase-only self-cal (ACA only) """
ACA_p2 = prefix+'_ACA.p2'
os.system('rm -rf '+ACA_p2)
gaincal(vis=ACA_cont_p1+'.ms', caltable=ACA_p2, gaintype='T', spw=ACA_contspws,
        refant=ACA_refant, combine='spw', calmode='p', solint='30s', minsnr=3., minblperant=3)
'''
1 of 8 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/08:33:07.6
1 of 8 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/09:02:49.1
1 of 8 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/09:12:32.4
1 of 7 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/09:23:16.3
'''
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(ACA_p2,xaxis='time', yaxis='GainPhase')
""" Print calibration png file """
plotms(ACA_p2,xaxis='time', yaxis='GainPhase',overwrite=True,showgui=False,
       plotfile=os.path.join(ACA_selfcal_folder,prefix+'_ACA_gain_p2_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=ACA_cont_p1+'.ms', spw=ACA_contspws, spwmap = ACA_spw_mapping, gaintable=[ACA_p2], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
ACA_cont_p2 = prefix+'_ACA_contp2'
os.system('rm -rf %s.ms*' % ACA_cont_p2)
split(vis=ACA_cont_p1+'.ms', outputvis=ACA_cont_p2+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=ACA_cont_p2+'.ms', imagename = ACA_cont_p2, mask=mask_ACA, threshold = '4mJy', savemodel = 'modelcolumn', imsize=ACA_imsize, cellsize=ACA_cellsize, robust=0.5, interactive=False, parallel=use_parallel)
estimate_SNR(ACA_cont_p2+'.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
generate_image_png(ACA_cont_p2+'.image',plot_sizes=image_png_plot_sizes['ACA'],
                   color_scale_limits=[-3*rms_ACA,10*rms_ACA],
                   save_folder=ACA_selfcal_folder)
#AA_Tau_ACA_contp2.image
#Beam 5.453 arcsec x 3.533 arcsec (-45.27 deg)
#Flux inside disk mask: 177.11 mJy
#Peak intensity of source: 176.27 mJy/beam
#rms: 7.33e-01 mJy/beam
#Peak SNR: 240.56

non_self_caled_ACA_vis = ACA_cont_p0
self_caled_ACA_visibilities = {'p1':ACA_cont_p1,
                               'p2':ACA_cont_p2}

#be sure to check if the following is correct
ACA_EBs = ('EB0','EB1','EB2', 'EB3', 'EB4')
ACA_EB_spws = ('0,1,2,3','4,5,6,7','8,9,10,11', '12,13,14,15',  '16,17,18,19')

for self_cal_step,self_caled_vis in self_caled_ACA_visibilities.items():
    for EB_key,spw in zip(ACA_EBs,ACA_EB_spws):
        nametemplate = f'{prefix}_ACA_{EB_key}_{self_cal_step}_compare_amp_vs_time'
        visibilities = [self_caled_vis+'.ms',non_self_caled_ACA_vis+'.ms']
        #we plot the whole uv range because sources are unresolved for ACA
        plot_amp_vs_time_comparison(
                nametemplate=nametemplate,visibilities=visibilities,spw=spw,
                uvrange=uv_ranges['ACA'],output_folder=ACA_selfcal_folder)


all_ACA_visibilities = self_caled_ACA_visibilities.copy()
all_ACA_visibilities['p0'] = ACA_cont_p0


for self_cal_step,vis_name in all_ACA_visibilities.items():
    #split out EBs
    vis_ms = vis_name+'.ms'
    nametemplate = vis_ms.replace('.ms','_EB')
    split_all_obs(msfile=vis_ms,nametemplate=nametemplate)
    exported_ms = []
    for i in range(number_of_EBs['ACA']):
        EB_vis = f'{nametemplate}{i}.ms'
        export_MS(EB_vis)
        exported_ms.append(EB_vis.replace('.ms','.vis.npz'))
    for i,exp_ms in enumerate(exported_ms):
        png_filename = f'flux_comparison_ACA_{self_cal_step}_{flux_ref_EB}_to_ACA_EB{i}.png'
        plot_label = os.path.join(ACA_selfcal_folder,png_filename)
        assert 'SB' in flux_ref_EB, 'comparison to ACA is only possible with SB (or ACA itself)'
        estimate_flux_scale(reference=prefix+f'_{flux_ref_EB}_initcont_shift.vis.npz',
                            comparison=exp_ms,incl=incl,
                            PA=PA,plot_label=plot_label)
    fluxscale = [1.,]*number_of_EBs['ACA']
    plot_label = os.path.join(ACA_selfcal_folder,
                              f'deprojected_vis_profiles_ACA_{self_cal_step}.png')
    plot_deprojected(filelist=exported_ms,fluxscale=fluxscale, PA=PA, incl=incl,
                     show_err=True,plot_label=plot_label)



#The ratio of the fluxes of AA_Tau_ACA_contp2_EB0.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.98547
#The scaling factor for gencal is 0.993 for your comparison measurement
#The error on the weighted mean ratio is 3.399e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_contp2_EB1.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.00499
#The scaling factor for gencal is 1.002 for your comparison measurement
#The error on the weighted mean ratio is 3.663e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_contp2_EB2.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.97485
#The scaling factor for gencal is 0.987 for your comparison measurement
#The error on the weighted mean ratio is 5.136e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_contp2_EB3.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.00689
#The scaling factor for gencal is 1.003 for your comparison measurement
#The error on the weighted mean ratio is 3.966e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACA_contp2_EB4.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.00486
#The scaling factor for gencal is 1.002 for your comparison measurement
#The error on the weighted mean ratio is 3.574e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor




"""SELF-CAL ACA + SB data"""
#reminder: clean down to 6 sigma for phase self-cal

SB_selfcal_folder = get_figures_folderpath('8_selfcal_ACASB_figures')
make_figures_folder(SB_selfcal_folder)

""" Merge the ACA self-cal'ed ms with SB ms"""
SB_cont_p0 = prefix+'_ACASB_contp0'
os.system('rm -rf %s.ms*' % SB_cont_p0)

concat(vis=[ACA_cont_p2+'.ms']+[f'{prefix}_SB_EB{i}_initcont_shift.ms' for i
                                          in range(number_of_EBs['SB'])],
            concatvis=SB_cont_p0+'.ms', dirtol='0.1arcsec', copypointing=False)

listobs(
    vis=SB_cont_p0+'.ms',
    listfile=SB_cont_p0+'.ms.txt',
    overwrite=True,
)

"""Define new SB mask using new center"""

mask_pa = PA #position angle of mask in degrees
mask_semimajor = 3 #semimajor axis of mask in arcsec
mask_semiminor = mask_semimajor*np.cos(incl/180.*np.pi) #semiminor axis of mask in arcsec
mask_ra = '04h34m55.429800s'
mask_dec = '+24.28.52.56450'

SB_mask= f'ellipse[[{mask_ra},{mask_dec}], [{mask_semimajor:.3f}arcsec, {mask_semiminor:.3f}arcsec], {mask_pa:.1f}deg]'
# Cellsize: ~beam/6-7


SB_cellsize =  '0.040arcsec'
SB_imsize = 400 # primary beam
SB_scales = [0,8,15,30]

noise_annulus_SB = f"annulus[[{mask_ra}, {mask_dec}],['4.arcsec', '6.arcsec']]"

tclean_wrapper(vis=SB_cont_p0+'.ms', imagename = SB_cont_p0, mask=SB_mask, threshold = '0.8mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_cont_p0+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
rms_SB = imstat(imagename = SB_cont_p0+'.image', region = noise_annulus_SB)['rms'][0]
generate_image_png(SB_cont_p0+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_selfcal_folder,
                   mask=SB_cont_p0+'.mask',noise_annulus=noise_annulus_SB)
#AA_Tau_ACASB_contp0.image
#Beam 0.599 arcsec x 0.486 arcsec (-37.61 deg)
#Flux inside disk mask: 179.05 mJy
#Peak intensity of source: 46.20 mJy/beam
#rms: 1.44e-01 mJy/beam
#Peak SNR: 320.14


""" Look for references antennas from weblog, and pick the first that are listed, overlapping with all EBs """
""" Since we are combined ACA EBs to SB EBs, ref antennas for both ACA and SB are needed """
# For ACA I chose CM02@J502, CM10@J501

# SB EB0: DA46, DV13, DV24, DA45, DV07, DV14
# SB EB1: DA57, DA46, DV13, DA60, DA45, DV24
# SB EB2: DA46, DA57, DA45, DV14, DV24, DV13

""" Get station numbers """
get_station_numbers(SB_cont_p0+'.ms','DA46')
get_station_numbers(SB_cont_p0+'.ms','DA57')
get_station_numbers(SB_cont_p0+'.ms','DV13')
#Observation ID 5: DA46@A001
#Observation ID 6: DA46@A001
#Observation ID 7: DA46@A001
#Observation ID 5: DA57@A036
#Observation ID 6: DA57@A036
#Observation ID 7: DA57@A036
#Observation ID 5: DV13@A019
#Observation ID 6: DV13@A019
#Observation ID 7: DV13@A019


SB_contspws = '0~31'
SB_refant   = 'DA46@A001, DA57@A036, CM02@J502, CM10@J501'

SB_spw_mapping = [0,0,0,0, 4,4,4,4, 8,8,8,8, 12,12,12,12, 16,16,16,16, 20,20,20,20, 24,24,24,24, 28,28,28,28]

SB_p1 = prefix+'_ACASB.p1'
os.system('rm -rf '+SB_p1)
gaincal(vis=SB_cont_p0+'.ms', caltable=SB_p1, gaintype='G', spw=SB_contspws,
        refant=SB_refant, combine='scan,spw', calmode='p', solint='inf', minsnr=3., minblperant=4)
'''
2 of 72 solutions flagged due to SNR < 3 in spw=28 at 2022/09/21/10:28:03.7
'''

""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_p1,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_p1,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_selfcal_folder,prefix+'_SB_gain_p1_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=SB_cont_p0+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_p1], interp='linearPD', calwt=True, applymode='calonly')

SB_cont_p1 = prefix+'_ACASB_contp1'
os.system('rm -rf %s.ms*' % SB_cont_p1)
split(vis=SB_cont_p0+'.ms', outputvis=SB_cont_p1+'.ms', datacolumn='corrected')

tclean_wrapper(vis=SB_cont_p1+'.ms', imagename = SB_cont_p1, mask=SB_mask, threshold = '0.8mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_cont_p1+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_cont_p1+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_selfcal_folder)
#AA_Tau_ACASB_contp1.image
#Beam 0.599 arcsec x 0.486 arcsec (-37.61 deg)
#Flux inside disk mask: 178.96 mJy
#Peak intensity of source: 46.20 mJy/beam
#rms: 1.44e-01 mJy/beam
#Peak SNR: 320.07


SB_p2 = prefix+'_ACASB.p2'
os.system('rm -rf '+SB_p2)
gaincal(vis=SB_cont_p1+'.ms', caltable=SB_p2, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='scan, spw', calmode='p', solint='360s', minsnr=3., minblperant=4)
# Maximum ~4/36 solutions flagged for SB
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_p2,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_p2,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_selfcal_folder,prefix+'_SB_gain_p2_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=SB_cont_p1+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_p2], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_cont_p2 = prefix+'_ACASB_contp2'
os.system('rm -rf %s.ms*' % SB_cont_p2)
split(vis=SB_cont_p1+'.ms', outputvis=SB_cont_p2+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_cont_p2+'.ms', imagename = SB_cont_p2, mask=SB_mask, threshold = '0.8mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_cont_p2+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_cont_p2+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_selfcal_folder)
#AA_Tau_ACASB_contp2.image
#Beam 0.599 arcsec x 0.486 arcsec (-37.61 deg)
#Flux inside disk mask: 179.81 mJy
#Peak intensity of source: 48.41 mJy/beam
#rms: 1.35e-01 mJy/beam
#Peak SNR: 357.50


SB_p3 = prefix+'_ACASB.p3'
os.system('rm -rf '+SB_p3)
gaincal(vis=SB_cont_p2+'.ms', caltable=SB_p3, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='spw', calmode='p', solint='120s', minsnr=3., minblperant=4)
# Maximum ~3/36 solutions flagged for SB
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_p3,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_p3,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_selfcal_folder,prefix+'_SB_gain_p3_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=SB_cont_p2+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_p3], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_cont_p3 = prefix+'_ACASB_contp3'
os.system('rm -rf %s.ms*' % SB_cont_p3)
split(vis=SB_cont_p2+'.ms', outputvis=SB_cont_p3+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_cont_p3+'.ms', imagename = SB_cont_p3, mask=SB_mask, threshold = '0.8mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_cont_p3+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_cont_p3+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_selfcal_folder)
#AA_Tau_ACASB_contp3.image
#Beam 0.599 arcsec x 0.486 arcsec (-37.61 deg)
#Flux inside disk mask: 180.02 mJy
#Peak intensity of source: 49.35 mJy/beam
#rms: 1.33e-01 mJy/beam
#Peak SNR: 371.17


SB_p4 = prefix+'_ACASB.p4'
os.system('rm -rf '+SB_p4)
gaincal(vis=SB_cont_p3+'.ms', caltable=SB_p4, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='spw', calmode='p', solint='60s', minsnr=3., minblperant=4)
# Maximum 4/36 solutions flagged for SB
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_p4,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_p4,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_selfcal_folder,prefix+'_SB_gain_p4_phase_vs_time.png'))

applycal(vis=SB_cont_p3+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_p4], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_cont_p4 = prefix+'_ACASB_contp4'
os.system('rm -rf %s.ms*' % SB_cont_p4)
split(vis=SB_cont_p3+'.ms', outputvis=SB_cont_p4+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_cont_p4+'.ms', imagename = SB_cont_p4, mask=SB_mask, threshold = '0.6mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_cont_p4+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_cont_p4+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_selfcal_folder)
#AA_Tau_ACASB_contp4.image
#Beam 0.599 arcsec x 0.486 arcsec (-37.61 deg)
#Flux inside disk mask: 183.98 mJy
#Peak intensity of source: 51.57 mJy/beam
#rms: 9.62e-02 mJy/beam
#Peak SNR: 536.19


SB_p5 = prefix+'_ACASB.p5'
os.system('rm -rf '+SB_p5)
gaincal(vis=SB_cont_p4+'.ms', caltable=SB_p5, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='spw', calmode='p', solint='20s', minsnr=3., minblperant=4)
# Maximum 6/36 solutions flagged for SB
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_p5,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_p5,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_selfcal_folder,prefix+'_SB_gain_p5_phase_vs_time.png'))

applycal(vis=SB_cont_p4+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_p5], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_cont_p5 = prefix+'_ACASB_contp5'
os.system('rm -rf %s.ms*' % SB_cont_p5)
split(vis=SB_cont_p4+'.ms', outputvis=SB_cont_p5+'.ms', datacolumn='corrected')


""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_cont_p5+'.ms', imagename = SB_cont_p5, mask=SB_mask, threshold = '0.6mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_cont_p5+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_cont_p5+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_selfcal_folder)
#AA_Tau_ACASB_contp5.image
#Beam 0.599 arcsec x 0.486 arcsec (-37.61 deg)
#Flux inside disk mask: 184.45 mJy
#Peak intensity of source: 54.57 mJy/beam
#rms: 9.81e-02 mJy/beam
#Peak SNR: 556.43

#plot amp vs time for a narrow uv range
non_self_caled_SB_vis = SB_cont_p0
self_caled_SB_visibilities = {'p1':SB_cont_p1,
                              'p2':SB_cont_p2,
                              'p3':SB_cont_p3,
                              'p4':SB_cont_p4,
                              'p5':SB_cont_p5}

SB_EBs = ('EB0','EB1', 'EB2')
SB_EB_spws = ('20,21,22,23', '24,25,26,27', '28,29,30,31') #fill in spws for each EB

for self_cal_step,self_caled_vis in self_caled_SB_visibilities.items():
    for EB_key,spw in zip(SB_EBs,SB_EB_spws):
        nametemplate = f'{prefix}_SB_{EB_key}_{self_cal_step}_compare_amp_vs_time'
        visibilities = [self_caled_vis+'.ms',non_self_caled_SB_vis+'.ms']
        plot_amp_vs_time_comparison(
                nametemplate=nametemplate,visibilities=visibilities,spw=spw,
                uvrange=uv_ranges['SB'],output_folder=SB_selfcal_folder)


all_SB_visibilities = self_caled_SB_visibilities.copy()
all_SB_visibilities['p0'] = SB_cont_p0

"""
if SB suffers from decoherence, then take an LB EB as flux reference; modify the code
below accordingly

If you haven't re-scaled the fluxes of LB EBs because all SB EBs had high decoherence,
check flux offsets of LB EBs here and correct them, if needed

"""

for self_cal_step,vis_name in all_SB_visibilities.items():
    #split out SB EBs
    vis_ms = vis_name+'.ms'
    nametemplate = vis_ms.replace('.ms','_EB')
    split_all_obs(msfile=vis_ms,nametemplate=nametemplate)
    exported_ms = []
    #we only compare SB to LB, not ACA to LB
    for i in range(number_of_EBs['ACA'],number_of_EBs['SB']+number_of_EBs['ACA']):
        EB_vis = f'{nametemplate}{i}.ms'
        export_MS(EB_vis) #export MS contents into Numpy save files
        exported_ms.append(EB_vis.replace('.ms','.vis.npz'))
    for i,exp_ms in zip(range(number_of_EBs['ACA'],number_of_EBs['SB']+number_of_EBs['ACA']),exported_ms):
        png_filename = f'flux_comparison_ACASB_EB{i}_{self_cal_step}_to_{flux_ref_EB}.png'
        plot_label = os.path.join(SB_selfcal_folder,png_filename)
        estimate_flux_scale(reference=f'{prefix}_{flux_ref_EB}_initcont_shift.vis.npz',
                            comparison=exp_ms,incl=incl,PA=PA,plot_label=plot_label)
    fluxscale = [1.,]*number_of_EBs['SB']
    plot_label = os.path.join(SB_selfcal_folder,
                              f'deprojected_vis_profiles_SB_{self_cal_step}.png')
    plot_deprojected(filelist=exported_ms,fluxscale=fluxscale, PA=PA, incl=incl,
                     show_err=True,plot_label=plot_label)

#Measurement set exported to AA_Tau_ACASB_contp5_EB7.vis.npz
#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB5.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.07042
#The scaling factor for gencal is 1.035 for your comparison measurement
#The error on the weighted mean ratio is 8.697e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB6.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 1.06089
#The scaling factor for gencal is 1.030 for your comparison measurement
#The error on the weighted mean ratio is 8.785e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB7.vis.npz to
#AA_Tau_SB_EB0_initcont_shift.vis.npz is 0.92291
#The scaling factor for gencal is 0.961 for your comparison measurement
#The error on the weighted mean ratio is 7.134e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor




#Let's look at ACA EBs as well:
self_caled_SB_visibilities = {}
self_caled_SB_visibilities = {'p5':SB_cont_p5}
all_SB_visibilities = self_caled_SB_visibilities.copy()

for self_cal_step,vis_name in all_SB_visibilities.items():
    #split out SB EBs
    vis_ms = vis_name+'.ms'
    nametemplate = vis_ms.replace('.ms','_EB')
    split_all_obs(msfile=vis_ms,nametemplate=nametemplate)
    #we only compare SB to LB, not ACA to LB
    for i in range(number_of_EBs['SB']+number_of_EBs['ACA']):
        EB_vis = f'{nametemplate}{i}.ms'
        export_MS(EB_vis) #export MS contents into Numpy save files

'''
Remember:
EB0 -> ACA EB0
EB1 -> ACA EB1
EB2 -> ACA EB2
EB3 -> ACA EB3
EB4 -> ACA EB4
EB5 -> SB EB0
EB6 -> SB EB1
EB7 -> SB EB3
'''

list_npz_files = []

list_npz_files = [f'{prefix}_ACASB_contp5_EB0.vis.npz',
                  f'{prefix}_ACASB_contp5_EB1.vis.npz',
                  f'{prefix}_ACASB_contp5_EB2.vis.npz',
                  f'{prefix}_ACASB_contp5_EB3.vis.npz',
                  f'{prefix}_ACASB_contp5_EB4.vis.npz',
                  f'{prefix}_ACASB_contp5_EB5.vis.npz',
                  f'{prefix}_ACASB_contp5_EB6.vis.npz',
                  f'{prefix}_ACASB_contp5_EB7.vis.npz',]

plot_label = os.path.join(SB_selfcal_folder,
                          f'deprojected_vis_profiles_ACASB_p5.png')

plot_deprojected(filelist=list_npz_files,
                 fluxscale=[1.]*(number_of_EBs['SB']+number_of_EBs['ACA']),PA=PA,incl=incl,show_err=True,
                 plot_label=plot_label)



list_npz_files = []

list_npz_files = [f'{prefix}_ACASB_contp5_EB0.vis.npz',
                  f'{prefix}_ACASB_contp5_EB1.vis.npz',
                  f'{prefix}_ACASB_contp5_EB2.vis.npz',
                  f'{prefix}_ACASB_contp5_EB3.vis.npz',
                  f'{prefix}_ACASB_contp5_EB4.vis.npz',
                  f'{prefix}_ACASB_contp5_EB5.vis.npz',
                  f'{prefix}_ACASB_contp5_EB6.vis.npz',
                  f'{prefix}_ACASB_contp5_EB7.vis.npz',
                  f'{prefix}_LB_EB0_initcont_shift.vis.npz',
                  f'{prefix}_LB_EB1_initcont_shift.vis.npz',
                  f'{prefix}_LB_EB2_initcont_shift.vis.npz',
                  f'{prefix}_LB_EB3_initcont_shift.vis.npz']

plot_label = os.path.join(SB_selfcal_folder,
                          f'deprojected_vis_profiles_ACASB_p6_and_LB.png')

plot_deprojected(filelist=list_npz_files,
                 fluxscale=[1.]*(number_of_EBs['LB']+number_of_EBs['SB']+number_of_EBs['ACA']),PA=PA,incl=incl,show_err=True,
                 plot_label=plot_label)


for i in np.arange(number_of_EBs['SB']+number_of_EBs['ACA']):
    estimate_flux_scale(reference=f'{prefix}_ACASB_contp5_EB5.vis.npz',
                            comparison=f'{prefix}_ACASB_contp5_EB'+str(i)+'.vis.npz',incl=incl,PA=PA,
                            plot_label=os.path.join(SB_selfcal_folder,f'{prefix}_ACASB_contp5_EB'+str(i)+'_to_ACASB_contp5_EB5_comparison.png'))

for i in np.arange(number_of_EBs['LB']):
    estimate_flux_scale(reference=f'{prefix}_ACASB_contp5_EB5.vis.npz',
                            comparison=f'{prefix}_LB_EB'+str(i)+'_initcont_shift.vis.npz',incl=incl,PA=PA,
                            plot_label=os.path.join(SB_selfcal_folder,f'{prefix}_LB_init_EB'+str(i)+'_to_ACASB_contp5_EB5_comparison.png'))

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB0.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.96873
#The scaling factor for gencal is 0.984 for your comparison measurement
#The error on the weighted mean ratio is 3.324e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB1.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.98580
#The scaling factor for gencal is 0.993 for your comparison measurement
#The error on the weighted mean ratio is 3.580e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB2.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.95829
#The scaling factor for gencal is 0.979 for your comparison measurement
#The error on the weighted mean ratio is 5.029e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB3.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.98941
#The scaling factor for gencal is 0.995 for your comparison measurement
#The error on the weighted mean ratio is 3.880e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB4.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.98573
#The scaling factor for gencal is 0.993 for your comparison measurement
#The error on the weighted mean ratio is 3.495e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB5.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 1.00000
#The scaling factor for gencal is 1.000 for your comparison measurement
#The error on the weighted mean ratio is 7.813e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB6.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.98752
#The scaling factor for gencal is 0.994 for your comparison measurement
#The error on the weighted mean ratio is 7.901e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_contp5_EB7.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.85740
#The scaling factor for gencal is 0.926 for your comparison measurement
#The error on the weighted mean ratio is 6.384e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB0_initcont_shift.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.98784
#The scaling factor for gencal is 0.994 for your comparison measurement
#The error on the weighted mean ratio is 1.463e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB1_initcont_shift.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.99615
#The scaling factor for gencal is 0.998 for your comparison measurement
#The error on the weighted mean ratio is 1.386e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB2_initcont_shift.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.98455
#The scaling factor for gencal is 0.992 for your comparison measurement
#The error on the weighted mean ratio is 1.412e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB3_initcont_shift.vis.npz to
#AA_Tau_ACASB_contp5_EB5.vis.npz is 0.97975
#The scaling factor for gencal is 0.990 for your comparison measurement
#The error on the weighted mean ratio is 1.254e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor


for i in range(6):
    estimate_flux_scale(reference=f'{prefix}_ACASB_contp5_EB5.vis.npz',
                            comparison=f'{prefix}_ACASB_contp'+str(i)+'_EB7.vis.npz',incl=incl,PA=PA,
                            plot_label=os.path.join(SB_selfcal_folder,f'{prefix}_ACASB_contp'+str(i)+'_EB7_to_ACASB_contp5_EB5_comparison.png'))

for i in range(6):
    estimate_flux_scale(reference=f'{prefix}_ACASB_contp5_EB7.vis.npz',
                            comparison=f'{prefix}_ACASB_contp'+str(i)+'_EB6.vis.npz',incl=incl,PA=PA)


'''
The phase selfcal on ACA and ACA+SB improved the decoherence ('downward trend in flux ratios'), but there is still some trend in SB EB1 and EB2.

Looking at flux offests, I'll rescale only ACA EB2 and SB EB2.
All other EBs have flux ratios differences within 4%.
'''

# Check the amp-vs-uvdist plot for SB EB2 after self-cal

plotms(vis='AA_Tau_ACASB_contp5_EB7.ms',
    xaxis='UVdist',
    yaxis='amplitude',
    spw='1',
    field='AA_Tau',
    ydatacolumn='data',
    avgtime='1e8',
    avgscan=True,
    avgchannel='3840',
    showgui = False,
    plotrange=uvdist_amp_plotranges['SB'],
    plotfile=os.path.join(SB_selfcal_folder,
                          'AA_Tau_ACASB_contp5_EB7_uvdist-v-amp.png'))

# There is no more the problem with the visibilities showing low amplitudes in plotms.



########## END of ACA and ACA+SB self-cal iteration 1 ##########


# Rescaling flux of the non-self-caled EBs

rescale_flux(vis=prefix+'_ACA_EB2_initcont_shift.ms', gencalparameter=[0.979])
rescale_flux(vis=prefix+'_SB_EB2_initcont_shift.ms', gencalparameter=[0.926])
#Splitting out rescaled values into new MS: AA_Tau_ACA_EB2_initcont_shift_rescaled.ms
#Splitting out rescaled values into new MS: AA_Tau_SB_EB2_initcont_shift_rescaled.ms

listobs(vis=prefix+'_ACA_EB2_initcont_shift_rescaled.ms',listfile=prefix+'_ACA_EB2_initcont_shift_rescaled.ms'+'.txt',overwrite=True)
listobs(vis=prefix+'_SB_EB2_initcont_shift_rescaled.ms',listfile=prefix+'_SB_EB2_initcont_shift_rescaled.ms'+'.txt',overwrite=True)


########## BEGIN of ACA and ACA+SB self-cal iteration 2 ##########

"""
Self-cal of ACA data only
"""
#for phase self-cal, we clean down to 6 sigma

ACA_iteration2_selfcal_folder = get_figures_folderpath('7.1_selfcal_ACA_iteration2_figures')
make_figures_folder(ACA_iteration2_selfcal_folder)

""" Merge the ACA executions back into a single MS """
ACA_iteration2_cont_p0 = prefix+'_ACA_iteration2_contp0'
os.system('rm -rf %s.ms*' % ACA_iteration2_cont_p0)

# concat the right rescaled files
concat(vis=[f'{prefix}_ACA_EB0_initcont_shift.ms',
            f'{prefix}_ACA_EB1_initcont_shift.ms',
            f'{prefix}_ACA_EB2_initcont_shift_rescaled.ms',
            f'{prefix}_ACA_EB3_initcont_shift.ms',
            f'{prefix}_ACA_EB4_initcont_shift.ms'],
            concatvis=ACA_iteration2_cont_p0+'.ms',
            dirtol='0.1arcsec',copypointing=False)

listobs(
    vis=ACA_iteration2_cont_p0+'.ms',
    listfile=ACA_iteration2_cont_p0+'.ms.txt',
    overwrite=True,
)

""" Initial clean (down to 6 sigma)"""
tclean_wrapper(vis=ACA_iteration2_cont_p0+'.ms', imagename = ACA_iteration2_cont_p0, mask=mask_ACA, deconvolver='hogbom',
               threshold = '6mJy', savemodel = 'modelcolumn', imsize=ACA_imsize,
               cellsize=ACA_cellsize, robust=0.5, interactive=False, parallel=use_parallel)
estimate_SNR(ACA_iteration2_cont_p0+'.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
rms_ACA = imstat(imagename = ACA_iteration2_cont_p0+'.image', region = noise_annulus_ACA)['rms'][0]
generate_image_png(ACA_iteration2_cont_p0+'.image',plot_sizes=image_png_plot_sizes['ACA'],
                   color_scale_limits=[-3*rms_ACA,10*rms_ACA],
                   save_folder=ACA_iteration2_selfcal_folder)
#AA_Tau_ACA_iteration2_contp0.image
#Beam 5.440 arcsec x 3.544 arcsec (-45.19 deg)
#Flux inside disk mask: 173.90 mJy
#Peak intensity of source: 175.11 mJy/beam
#rms: 9.09e-01 mJy/beam
#Peak SNR: 192.69


""" First round of phase-only self-cal """
# Use scan-length interval to align SPWs as much as possible
ACA_iteration2_p1 = prefix+'_ACA_iteration2.p1'
os.system('rm -rf '+ACA_iteration2_p1)
gaincal(vis=ACA_iteration2_cont_p0+'.ms', caltable=ACA_iteration2_p1, gaintype='G', spw=ACA_contspws,
        refant=ACA_refant, combine='scan', calmode='p', solint='inf', minsnr=3., minblperant=3)

""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(ACA_iteration2_p1,xaxis='time', yaxis='GainPhase')
""" Print calibration png file """
plotms(ACA_iteration2_p1,xaxis='time', yaxis='GainPhase',overwrite=True,showgui=False,
       plotfile=os.path.join(ACA_iteration2_selfcal_folder,prefix+'_ACA_iteration2_gain_p1_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=ACA_iteration2_cont_p0+'.ms', spw=ACA_contspws, gaintable=[ACA_iteration2_p1],
         interp='linear', calwt=True, applymode='calonly')

""" Split off a corrected MS """
ACA_iteration2_cont_p1 = prefix+'_ACA_iteration2_contp1'
os.system('rm -rf %s.ms*' % ACA_iteration2_cont_p1)
split(vis=ACA_iteration2_cont_p0+'.ms', outputvis=ACA_iteration2_cont_p1+'.ms', datacolumn='corrected')


""" Image the results; check the resulting map """
#again clean down to 6 sigma
tclean_wrapper(vis=ACA_iteration2_cont_p1+'.ms', imagename = ACA_iteration2_cont_p1, mask=mask_ACA, threshold = '6mJy', savemodel = 'modelcolumn', imsize=ACA_imsize, cellsize=ACA_cellsize, robust=0.5, interactive=False, parallel=use_parallel)
estimate_SNR(ACA_iteration2_cont_p1+'.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
generate_image_png(ACA_iteration2_cont_p1+'.image',plot_sizes=image_png_plot_sizes['ACA'],
                   color_scale_limits=[-3*rms_ACA,10*rms_ACA],
                   save_folder=ACA_iteration2_selfcal_folder)
#AA_Tau_ACA_iteration2_contp1.image
#Beam 5.440 arcsec x 3.544 arcsec (-45.19 deg)
#Flux inside disk mask: 174.17 mJy
#Peak intensity of source: 175.36 mJy/beam
#rms: 9.07e-01 mJy/beam
#Peak SNR: 193.30


""" Second round of phase-only self-cal (ACA only) """
ACA_iteration2_p2 = prefix+'_ACA_iteration2.p2'
os.system('rm -rf '+ACA_iteration2_p2)
gaincal(vis=ACA_iteration2_cont_p1+'.ms', caltable=ACA_iteration2_p2, gaintype='T', spw=ACA_contspws,
        refant=ACA_refant, combine='spw', calmode='p', solint='30s', minsnr=3., minblperant=3)
'''
1 of 8 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/08:33:07.6
1 of 8 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/09:02:49.1
1 of 8 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/09:12:32.4
1 of 7 solutions flagged due to SNR < 3 in spw=8 at 2021/11/01/09:23:16.3
'''
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(ACA_iteration2_p2,xaxis='time', yaxis='GainPhase')
""" Print calibration png file """
plotms(ACA_iteration2_p2,xaxis='time', yaxis='GainPhase',overwrite=True,showgui=False,
       plotfile=os.path.join(ACA_iteration2_selfcal_folder,prefix+'_ACA_iteration2_gain_p2_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=ACA_iteration2_cont_p1+'.ms', spw=ACA_contspws, spwmap = ACA_spw_mapping, gaintable=[ACA_iteration2_p2], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
ACA_iteration2_cont_p2 = prefix+'_ACA_iteration2_contp2'
os.system('rm -rf %s.ms*' % ACA_iteration2_cont_p2)
split(vis=ACA_iteration2_cont_p1+'.ms', outputvis=ACA_iteration2_cont_p2+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=ACA_iteration2_cont_p2+'.ms', imagename = ACA_iteration2_cont_p2, mask=mask_ACA, threshold = '4mJy', savemodel = 'modelcolumn', imsize=ACA_imsize, cellsize=ACA_cellsize, robust=0.5, interactive=False, parallel=use_parallel)
estimate_SNR(ACA_iteration2_cont_p2+'.image', disk_mask = mask_ACA, noise_mask = noise_annulus_ACA)
generate_image_png(ACA_iteration2_cont_p2+'.image',plot_sizes=image_png_plot_sizes['ACA'],
                   color_scale_limits=[-3*rms_ACA,10*rms_ACA],
                   save_folder=ACA_iteration2_selfcal_folder)
#AA_Tau_ACA_iteration2_contp2.image
#Beam 5.440 arcsec x 3.544 arcsec (-45.19 deg)
#Flux inside disk mask: 178.29 mJy
#Peak intensity of source: 177.43 mJy/beam
#rms: 7.35e-01 mJy/beam
#Peak SNR: 241.33


"""SELF-CAL ACA + SB data"""
#reminder: clean down to 6 sigma for phase self-cal

SB_iteration2_selfcal_folder = get_figures_folderpath('8.1_selfcal_ACASB_iteration2_figures')
make_figures_folder(SB_iteration2_selfcal_folder)

""" Merge the ACA self-cal'ed ms with SB ms"""
SB_iteration2_cont_p0 = prefix+'_ACASB_iteration2_contp0'
os.system('rm -rf %s.ms*' % SB_iteration2_cont_p0)

concat(vis=[ACA_iteration2_cont_p2+'.ms',
            f'{prefix}_SB_EB0_initcont_shift.ms',
            f'{prefix}_SB_EB1_initcont_shift.ms',
            f'{prefix}_SB_EB2_initcont_shift_rescaled.ms'],
            concatvis=SB_iteration2_cont_p0+'.ms',
            dirtol='0.1arcsec',copypointing=False)

listobs(
    vis=SB_iteration2_cont_p0+'.ms',
    listfile=SB_iteration2_cont_p0+'.ms.txt',
    overwrite=True,
)


tclean_wrapper(vis=SB_iteration2_cont_p0+'.ms', imagename = SB_iteration2_cont_p0, mask=SB_mask, threshold = '0.85mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_iteration2_cont_p0+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
rms_SB = imstat(imagename = SB_iteration2_cont_p0+'.image', region = noise_annulus_SB)['rms'][0]
generate_image_png(SB_iteration2_cont_p0+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_iteration2_selfcal_folder)
#AA_Tau_ACASB_iteration2_contp0.image
#Beam 0.593 arcsec x 0.484 arcsec (-36.03 deg)
#Flux inside disk mask: 188.33 mJy
#Peak intensity of source: 48.80 mJy/beam
#rms: 1.36e-01 mJy/beam
#Peak SNR: 359.44


SB_iteration2_p1 = prefix+'_ACASB_iteration2.p1'
os.system('rm -rf '+SB_iteration2_p1)
gaincal(vis=SB_iteration2_cont_p0+'.ms', caltable=SB_iteration2_p1, gaintype='G', spw=SB_contspws,
        refant=SB_refant, combine='scan,spw', calmode='p', solint='inf', minsnr=3., minblperant=4)
#2 of 72 solutions flagged due to SNR < 3 in spw=28 at 2022/09/21/10:28:03.7
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_iteration2_p1,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_iteration2_p1,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_iteration2_selfcal_folder,prefix+'_SB_iteration2_gain_p1_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=SB_iteration2_cont_p0+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_iteration2_p1], interp='linearPD', calwt=True, applymode='calonly')

SB_iteration2_cont_p1 = prefix+'_ACASB_iteration2_contp1'
os.system('rm -rf %s.ms*' % SB_iteration2_cont_p1)
split(vis=SB_iteration2_cont_p0+'.ms', outputvis=SB_iteration2_cont_p1+'.ms', datacolumn='corrected')

tclean_wrapper(vis=SB_iteration2_cont_p1+'.ms', imagename = SB_iteration2_cont_p1, mask=SB_mask, threshold = '0.8mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_iteration2_cont_p1+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_iteration2_cont_p1+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_iteration2_selfcal_folder)
#AA_Tau_ACASB_iteration2_contp1.image
#Beam 0.593 arcsec x 0.484 arcsec (-36.03 deg)
#Flux inside disk mask: 188.34 mJy
#Peak intensity of source: 48.83 mJy/beam
#rms: 1.36e-01 mJy/beam
#Peak SNR: 360.10


SB_iteration2_p2 = prefix+'_ACASB_iteration2.p2'
os.system('rm -rf '+SB_iteration2_p2)
gaincal(vis=SB_iteration2_cont_p1+'.ms', caltable=SB_iteration2_p2, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='scan, spw', calmode='p', solint='360s', minsnr=3., minblperant=4)
# Maximum 3/36 solutions flagged in SB EBs
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_iteration2_p2,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_iteration2_p2,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_iteration2_selfcal_folder,prefix+'_SB_iteration2_gain_p2_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=SB_iteration2_cont_p1+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_iteration2_p2], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_iteration2_cont_p2 = prefix+'_ACASB_iteration2_contp2'
os.system('rm -rf %s.ms*' % SB_iteration2_cont_p2)
split(vis=SB_iteration2_cont_p1+'.ms', outputvis=SB_iteration2_cont_p2+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_iteration2_cont_p2+'.ms', imagename = SB_iteration2_cont_p2, mask=SB_mask, threshold = '0.8mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_iteration2_cont_p2+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_iteration2_cont_p2+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_iteration2_selfcal_folder)
#AA_Tau_ACASB_iteration2_contp2.image
#Beam 0.593 arcsec x 0.484 arcsec (-36.03 deg)
#Flux inside disk mask: 190.22 mJy
#Peak intensity of source: 51.58 mJy/beam
#rms: 1.13e-01 mJy/beam
#Peak SNR: 454.61


SB_iteration2_p3 = prefix+'_ACASB_iteration2.p3'
os.system('rm -rf '+SB_iteration2_p3)
gaincal(vis=SB_iteration2_cont_p2+'.ms', caltable=SB_iteration2_p3, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='spw', calmode='p', solint='120s', minsnr=3., minblperant=4)
# Maximum 3/36 solutions flagged for SB EBs
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_iteration2_p3,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_iteration2_p3,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_iteration2_selfcal_folder,prefix+'_SB_iteration2_gain_p3_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=SB_iteration2_cont_p2+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_iteration2_p3], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_iteration2_cont_p3 = prefix+'_ACASB_iteration2_contp3'
os.system('rm -rf %s.ms*' % SB_iteration2_cont_p3)
split(vis=SB_iteration2_cont_p2+'.ms', outputvis=SB_iteration2_cont_p3+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_iteration2_cont_p3+'.ms', imagename = SB_iteration2_cont_p3, mask=SB_mask, threshold = '0.6mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_iteration2_cont_p3+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_iteration2_cont_p3+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_iteration2_selfcal_folder)
#AA_Tau_ACASB_iteration2_contp3.image
#Beam 0.593 arcsec x 0.484 arcsec (-36.03 deg)
#Flux inside disk mask: 190.97 mJy
#Peak intensity of source: 52.69 mJy/beam
#rms: 1.02e-01 mJy/beam
#Peak SNR: 518.94

SB_iteration2_p4 = prefix+'_ACASB_iteration2.p4'
os.system('rm -rf '+SB_iteration2_p4)
gaincal(vis=SB_iteration2_cont_p3+'.ms', caltable=SB_iteration2_p4, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='spw', calmode='p', solint='60s', minsnr=3., minblperant=4)
# Maximum 4/36 solutions flagged for SB
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_iteration2_p4,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_iteration2_p4,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_iteration2_selfcal_folder,prefix+'_SB_iteration2_gain_p4_phase_vs_time.png'))

applycal(vis=SB_iteration2_cont_p3+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_iteration2_p4], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_iteration2_cont_p4 = prefix+'_ACASB_iteration2_contp4'
os.system('rm -rf %s.ms*' % SB_iteration2_cont_p4)
split(vis=SB_iteration2_cont_p3+'.ms', outputvis=SB_iteration2_cont_p4+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_iteration2_cont_p4+'.ms', imagename = SB_iteration2_cont_p4, mask=SB_mask, threshold = '0.6mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_iteration2_cont_p4+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_iteration2_cont_p4+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_iteration2_selfcal_folder)
#AA_Tau_ACASB_iteration2_contp4.image
#Beam 0.593 arcsec x 0.484 arcsec (-36.03 deg)
#Flux inside disk mask: 191.05 mJy
#Peak intensity of source: 53.96 mJy/beam
#rms: 1.02e-01 mJy/beam
#Peak SNR: 530.94



SB_iteration2_p5 = prefix+'_ACASB_iteration2.p5'
os.system('rm -rf '+SB_iteration2_p5)
gaincal(vis=SB_iteration2_cont_p4+'.ms', caltable=SB_iteration2_p5, gaintype='T', spw=SB_contspws,
        refant=SB_refant, combine='spw', calmode='p', solint='20s', minsnr=3., minblperant=4)
# Maximum 6/36 solutions flagged for SB and 4/7 for ACA
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(SB_iteration2_p5,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(SB_iteration2_p5,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(SB_iteration2_selfcal_folder,prefix+'_SB_iteration2_gain_p5_phase_vs_time.png'))

applycal(vis=SB_iteration2_cont_p4+'.ms', spw=SB_contspws, spwmap = SB_spw_mapping, gaintable=[SB_iteration2_p5], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
SB_iteration2_cont_p5 = prefix+'_ACASB_iteration2_contp5'
os.system('rm -rf %s.ms*' % SB_iteration2_cont_p5)
split(vis=SB_iteration2_cont_p4+'.ms', outputvis=SB_iteration2_cont_p5+'.ms', datacolumn='corrected')


""" Image the results; check the resulting map """
tclean_wrapper(vis=SB_iteration2_cont_p5+'.ms', imagename = SB_iteration2_cont_p5, mask=SB_mask, threshold = '0.55mJy', deconvolver='multiscale', scales=SB_scales, savemodel = 'modelcolumn', imsize=SB_imsize, cellsize=SB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(SB_iteration2_cont_p5+'.image', disk_mask = SB_mask, noise_mask = noise_annulus_SB)
generate_image_png(SB_iteration2_cont_p5+'.image',plot_sizes=image_png_plot_sizes['SB'],
                   color_scale_limits=[-3*rms_SB,10*rms_SB],save_folder=SB_iteration2_selfcal_folder)
#AA_Tau_ACASB_iteration2_contp5.image
#Beam 0.593 arcsec x 0.484 arcsec (-36.03 deg)
#Flux inside disk mask: 193.64 mJy
#Peak intensity of source: 57.63 mJy/beam
#rms: 9.12e-02 mJy/beam
#Peak SNR: 631.58


#plot amp vs time for a narrow uv range
non_self_caled_SB_iteration2_vis = SB_iteration2_cont_p0
self_caled_SB_iteration2_visibilities = {'p1':SB_iteration2_cont_p1,
                                         'p2':SB_iteration2_cont_p2,
                                         'p3':SB_iteration2_cont_p3,
                                         'p4':SB_iteration2_cont_p4,
                                         'p5':SB_iteration2_cont_p5}

for self_cal_step,self_caled_vis in self_caled_SB_iteration2_visibilities.items():
    for EB_key,spw in zip(SB_EBs,SB_EB_spws):
        nametemplate = f'{prefix}_SB_iteration2_{EB_key}_{self_cal_step}_compare_amp_vs_time'
        visibilities = [self_caled_vis+'.ms',non_self_caled_SB_iteration2_vis+'.ms']
        plot_amp_vs_time_comparison(
                nametemplate=nametemplate,visibilities=visibilities,spw=spw,
                uvrange=uv_ranges['SB'],output_folder=SB_iteration2_selfcal_folder)

all_SB_iteration2_visibilities = self_caled_SB_iteration2_visibilities.copy()
all_SB_iteration2_visibilities['p0'] = SB_iteration2_cont_p0

for self_cal_step,vis_name in all_SB_iteration2_visibilities.items():
    #split out SB EBs
    vis_ms = vis_name+'.ms'
    nametemplate = vis_ms.replace('.ms','_EB')
    split_all_obs(msfile=vis_ms,nametemplate=nametemplate)
    #we only compare SB to LB, not ACA to LB
    for i in range(number_of_EBs['SB']+number_of_EBs['ACA']):
        EB_vis = f'{nametemplate}{i}.ms'
        export_MS(EB_vis) #export MS contents into Numpy save files

'''
Remember:
EB0 -> ACA EB0
EB1 -> ACA EB1
EB2 -> ACA EB2
EB3 -> ACA EB3
EB4 -> ACA EB4
EB5 -> SB EB0
EB6 -> SB EB1
EB7 -> SB EB3
'''

list_npz_files = []

list_npz_files = [f'{prefix}_ACASB_iteration2_contp5_EB0.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB1.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB2.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB3.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB4.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB5.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB6.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB7.vis.npz',]

plot_label = os.path.join(SB_iteration2_selfcal_folder,
                          f'deprojected_vis_profiles_ACASB_iteration2_p5.png')
plot_deprojected(filelist=list_npz_files,
                 fluxscale=[1.]*(number_of_EBs['SB']+number_of_EBs['ACA']),PA=PA,incl=incl,show_err=True,
                 plot_label=plot_label)



list_npz_files = []

list_npz_files = [f'{prefix}_ACASB_iteration2_contp5_EB0.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB1.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB2.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB3.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB4.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB5.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB6.vis.npz',
                  f'{prefix}_ACASB_iteration2_contp5_EB7.vis.npz',
                  f'{prefix}_LB_EB0_initcont_shift.vis.npz',
                  f'{prefix}_LB_EB1_initcont_shift.vis.npz',
                  f'{prefix}_LB_EB2_initcont_shift.vis.npz',
                  f'{prefix}_LB_EB3_initcont_shift.vis.npz']

plot_label = os.path.join(SB_iteration2_selfcal_folder,
                          f'deprojected_vis_profiles_ACASB_iteration2_p6_and_LB.png')

plot_deprojected(filelist=list_npz_files,
                 fluxscale=[1.]*(number_of_EBs['LB']+number_of_EBs['SB']+number_of_EBs['ACA']),PA=PA,incl=incl,show_err=True,
                 plot_label=plot_label)


for i in np.arange(number_of_EBs['SB']+number_of_EBs['ACA']):
    estimate_flux_scale(reference=f'{prefix}_ACASB_iteration2_contp5_EB5.vis.npz',
                            comparison=f'{prefix}_ACASB_iteration2_contp5_EB'+str(i)+'.vis.npz',incl=incl,PA=PA,
                            plot_label=os.path.join(SB_iteration2_selfcal_folder,f'{prefix}_ACASB_iteration2_contp5_EB'+str(i)+'_to_ACASB_iteration2_contp5_EB5_comparison.png'))

for i in np.arange(number_of_EBs['LB']):
    estimate_flux_scale(reference=f'{prefix}_ACASB_iteration2_contp5_EB5.vis.npz',
                            comparison=f'{prefix}_LB_EB'+str(i)+'_initcont_shift.vis.npz',incl=incl,PA=PA,
                            plot_label=os.path.join(SB_iteration2_selfcal_folder,f'{prefix}_LB_init_EB'+str(i)+'_to_ACASB_iteration2_contp5_EB5_comparison.png'))

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB0.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.96879
#The scaling factor for gencal is 0.984 for your comparison measurement
#The error on the weighted mean ratio is 3.325e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB1.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.98587
#The scaling factor for gencal is 0.993 for your comparison measurement
#The error on the weighted mean ratio is 3.580e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB2.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.99987
#The scaling factor for gencal is 1.000 for your comparison measurement
#The error on the weighted mean ratio is 5.248e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB3.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.98948
#The scaling factor for gencal is 0.995 for your comparison measurement
#The error on the weighted mean ratio is 3.881e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB4.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.98581
#The scaling factor for gencal is 0.993 for your comparison measurement
#The error on the weighted mean ratio is 3.495e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 1.00000
#The scaling factor for gencal is 1.000 for your comparison measurement
#The error on the weighted mean ratio is 7.815e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB6.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.98841
#The scaling factor for gencal is 0.994 for your comparison measurement
#The error on the weighted mean ratio is 7.906e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASB_iteration2_contp5_EB7.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 1.00381
#The scaling factor for gencal is 1.002 for your comparison measurement
#The error on the weighted mean ratio is 7.461e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB0_initcont_shift.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.98825
#The scaling factor for gencal is 0.994 for your comparison measurement
#The error on the weighted mean ratio is 1.464e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB1_initcont_shift.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.99739
#The scaling factor for gencal is 0.999 for your comparison measurement
#The error on the weighted mean ratio is 1.387e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB2_initcont_shift.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.98496
#The scaling factor for gencal is 0.992 for your comparison measurement
#The error on the weighted mean ratio is 1.412e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_LB_EB3_initcont_shift.vis.npz to
#AA_Tau_ACASB_iteration2_contp5_EB5.vis.npz is 0.98062
#The scaling factor for gencal is 0.990 for your comparison measurement
#The error on the weighted mean ratio is 1.255e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor



for j in range(5):
    for i in np.arange(number_of_EBs['SB']+number_of_EBs['ACA']):
        estimate_flux_scale(reference=f'{prefix}_ACASB_iteration2_contp5_EB5.vis.npz',
                                comparison=f'{prefix}_ACASB_iteration2_contp'+str(j)+'_EB'+str(i)+'.vis.npz',incl=incl,PA=PA,
                                plot_label=os.path.join(SB_iteration2_selfcal_folder,f'{prefix}_ACASB_iteration2_contp'+str(j)+'_EB'+str(i)+'_to_ACASB_iteration2_contp5_EB5_comparison.png'))



'''
All flux offsets are within 4%
Some downward trend in the flux ratios is still present for SB EB2 data (and very marginally also for SB EB1).
'''


########## END of ACA and ACA+SB self-cal iteration 2 ##########



"""SELF-CAL COMBINED DATA"""
LB_selfcal_folder = get_figures_folderpath('9_selfcal_ACASBLB_figures')
make_figures_folder(LB_selfcal_folder)


""" Merge the SB self-cal'ed ms with LB ms"""
LB_cont_p0 = prefix+'_ACASBLB_contp0'
os.system('rm -rf %s.ms*' % LB_cont_p0)

concat(vis=[SB_iteration2_cont_p5+'.ms',
            f'{prefix}_LB_EB0_initcont_shift.ms',
            f'{prefix}_LB_EB1_initcont_shift.ms',
            f'{prefix}_LB_EB2_initcont_shift.ms',
            f'{prefix}_LB_EB3_initcont_shift.ms'],
            concatvis=LB_cont_p0+'.ms',
            dirtol='0.1arcsec',copypointing=False)

listobs(
    vis=LB_cont_p0+'.ms',
    listfile=LB_cont_p0+'.ms.txt',
    overwrite=True,
)

mask_pa = PA #position angle of mask in degrees
mask_semimajor = 3 #semimajor axis of mask in arcsec
mask_semiminor = mask_semimajor*np.cos(incl/180.*np.pi) #semiminor axis of mask in arcsec
mask_ra = '04h34m55.429800s'
mask_dec = '+24.28.52.56450'

LB_mask = f'ellipse[[{mask_ra},{mask_dec}], [{mask_semimajor:.3f}arcsec, {mask_semiminor:.3f}arcsec], {mask_pa:.1f}deg]'
noise_annulus_LB = f"annulus[[{mask_ra}, {mask_dec}],['4.arcsec', '6.arcsec']]"

LB_cellsize =  '0.01arcsec'
LB_imsize = 2000 # primary beam
LB_scales=[0,10,20,50,120]

""" clean down to ~6 sigma"""
tclean_wrapper(vis=LB_cont_p0+'.ms', imagename = LB_cont_p0, mask=LB_mask, threshold = '0.2mJy', deconvolver='multiscale', scales=LB_scales, savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_p0+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
rms_LB = imstat(imagename = LB_cont_p0+'.image', region = noise_annulus_LB)['rms'][0]
generate_image_png(LB_cont_p0+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder,
                   mask=LB_cont_p0+'.mask',noise_annulus=noise_annulus_LB)
#AA_Tau_ACASBLB_contp0.image
#Beam 0.098 arcsec x 0.076 arcsec (-19.27 deg)
#Flux inside disk mask: 201.27 mJy
#Peak intensity of source: 3.51 mJy/beam
#rms: 3.13e-02 mJy/beam
#Peak SNR: 112.08


""" Self-calibration parameters """

""" Look for references antennas from weblog, and pick the first that are listed, overlapping with all EBs """
""" Since we are combined LB EBs to SB+ACA EBs, ref antennas for LB, SB and ACA are needed """
# EB0: DA43, DV03, DV04, DV21, DA62, DA60
# EB1: DA43, DV03, DV04, DV21, DA62, DA60
# EB2: DA43, DV03, DV21, DA62, DA60, DA54
# EB3: DA43, DV03, DA60, DV04, DV21, DA65
get_station_numbers(LB_cont_p0+'.ms','DA43')
get_station_numbers(LB_cont_p0+'.ms','DV03')
#Observation ID 5: DA43@A073
#Observation ID 6: DA43@A073
#Observation ID 7: DA43@A073
#Observation ID 8: DA43@A035
#Observation ID 9: DA43@A035
#Observation ID 10: DA43@A035
#Observation ID 11: DA43@A035
#Observation ID 5: DV03@A027
#Observation ID 7: DV03@A027
#Observation ID 8: DV03@A027
#Observation ID 9: DV03@A027
#Observation ID 10: DV03@A027
#Observation ID 11: DV03@A027

get_station_numbers(LB_cont_p0+'.ms','DA46')
get_station_numbers(LB_cont_p0+'.ms','DA57')
get_station_numbers(LB_cont_p0+'.ms','DV13')
#Observation ID 5: DA46@A001
#Observation ID 6: DA46@A001
#Observation ID 7: DA46@A001
#Observation ID 8: DA46@A132
#Observation ID 9: DA46@A132
#Observation ID 10: DA46@A132
#Observation ID 11: DA46@A132
#Observation ID 5: DA57@A036
#Observation ID 6: DA57@A036
#Observation ID 7: DA57@A036
#Observation ID 8: DA57@A124
#Observation ID 9: DA57@A124
#Observation ID 10: DA57@A124
#Observation ID 11: DA57@A124
#Observation ID 5: DV13@A019
#Observation ID 6: DV13@A019
#Observation ID 7: DV13@A019
#Observation ID 8: DV13@A092
#Observation ID 9: DV13@A092
#Observation ID 10: DV13@A092
#Observation ID 11: DV13@A092

'''
Remember:
EB0 -> ACA EB0
EB1 -> ACA EB1
EB2 -> ACA EB2
EB3 -> ACA EB3
EB4 -> ACA EB4
EB5 -> SB EB0
EB6 -> SB EB1
EB7 -> SB EB3
EB8 -> LB EB0
EB9 -> LB EB1
EB10 -> LB EB2
EB11 -> LB EB3
'''

#SB_refant   = 'DA46@A001, DA57@A036, CM02@J502, CM10@J501'
LB_refant   = 'DA43@A035, DA46@A001, DA57@A036, DV03@A027, CM02@J502, CM10@J501'

LB_contspws = '0~47'

LB_spw_mapping = [0,0,0,0, 4,4,4,4, 8,8,8,8, 12,12,12,12, 16,16,16,16, 20,20,20,20, 24,24,24,24, 28,28,28,28,
                  32,32,32,32, 36,36,36,36, 40,40,40,40, 44,44,44,44]

LB_p1 = prefix+'_ACASBLB.p1'
os.system('rm -rf '+LB_p1)
#try gaintype='G', and if there are many flagged solutions, change it to 'T'
gaincal(vis=LB_cont_p0+'.ms', caltable=LB_p1, gaintype='G', spw=LB_contspws,
        refant=LB_refant, combine='scan,spw', calmode='p', solint='inf', minsnr=2.,
        minblperant=4)

""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(LB_p1,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(LB_p1,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_p1_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=LB_cont_p0+'.ms', spw=LB_contspws, spwmap = LB_spw_mapping, gaintable=[LB_p1], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
LB_cont_p1 = prefix+'_ACASBLB_contp1'
os.system('rm -rf %s.ms*' % LB_cont_p1)
split(vis=LB_cont_p0+'.ms', outputvis=LB_cont_p1+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=LB_cont_p1+'.ms', imagename = LB_cont_p1, mask=LB_mask, threshold = '0.19mJy', deconvolver='multiscale', scales=LB_scales, savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_p1+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_p1+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder)
#AA_Tau_ACASBLB_contp1.image
#Beam 0.098 arcsec x 0.076 arcsec (-19.27 deg)
#Flux inside disk mask: 200.92 mJy
#Peak intensity of source: 3.52 mJy/beam
#rms: 3.07e-02 mJy/beam
#Peak SNR: 114.71


""" Second round of phase-only self-cal (LB only) """
LB_p2 = prefix+'_ACASBLB.p2'
os.system('rm -rf '+LB_p2)
gaincal(vis=LB_cont_p1+'.ms', caltable=LB_p2, gaintype='T', spw=LB_contspws,
        refant=LB_refant, combine='spw,scan', calmode='p', solint='360s', minsnr=2., minblperant=4)
# Maximum 3/36 solutions flagged for SB and 20/49 for LB (average about 10/49)
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(LB_p2,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(LB_p2,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_p2_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=LB_cont_p1+'.ms', spw=LB_contspws, spwmap = LB_spw_mapping, gaintable=[LB_p2], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
LB_cont_p2 = prefix+'_ACASBLB_contp2'
os.system('rm -rf %s.ms*' % LB_cont_p2)
split(vis=LB_cont_p1+'.ms', outputvis=LB_cont_p2+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=LB_cont_p2+'.ms', imagename = LB_cont_p2, mask=LB_mask, threshold = '0.18mJy', deconvolver='multiscale', scales=LB_scales, savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_p2+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_p2+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder)
#AA_Tau_ACASBLB_contp2.image
#Beam 0.098 arcsec x 0.076 arcsec (-19.27 deg)
#Flux inside disk mask: 200.42 mJy
#Peak intensity of source: 3.55 mJy/beam
#rms: 3.04e-02 mJy/beam
#Peak SNR: 116.63

""" Third round of phase-only self-cal (LB only) """
"""
Check scan length to decide whether to combine scans. If scans are 2 min, do not combine them.
If they are shorter, combine scans here

For AA Tau, scan lenght for LB EBs is about 30s-1min
"""
LB_p3 = prefix+'_ACASBLB.p3'
os.system('rm -rf '+LB_p3)
gaincal(vis=LB_cont_p2+'.ms', caltable=LB_p3, gaintype='T', spw=LB_contspws,
        refant=LB_refant, combine='spw,scan', calmode='p', solint='120s', minsnr=2., minblperant=4)
# Maximum 3/36 solutions flagged for SB and 20/49 for LB (average of about 10/49)
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(LB_p3,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(LB_p3,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_p3_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=LB_cont_p2+'.ms', spw=LB_contspws, spwmap = LB_spw_mapping, gaintable=[LB_p3], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
LB_cont_p3 = prefix+'_ACASBLB_contp3'
os.system('rm -rf %s.ms*' % LB_cont_p3)
split(vis=LB_cont_p2+'.ms', outputvis=LB_cont_p3+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=LB_cont_p3+'.ms', imagename = LB_cont_p3, mask=LB_mask, threshold = '0.18mJy', deconvolver='multiscale', scales=LB_scales, savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_p3+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_p3+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder)
#AA_Tau_ACASBLB_contp3.image
#Beam 0.098 arcsec x 0.076 arcsec (-19.27 deg)
#Flux inside disk mask: 200.16 mJy
#Peak intensity of source: 3.61 mJy/beam
#rms: 3.03e-02 mJy/beam
#Peak SNR: 119.16


LB_p4 = prefix+'_ACASBLB.p4'
os.system('rm -rf '+LB_p4)
gaincal(vis=LB_cont_p3+'.ms', caltable=LB_p4, gaintype='T', spw=LB_contspws,
        refant=LB_refant, combine='spw', calmode='p', solint='60s', minsnr=2., minblperant=4)
# Maximum 5/36 solutions flagged for SB and 28/49 for LB (average around 17/49)
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(LB_p4,xaxis='time', yaxis='GainPhase',iteraxis='spw')
""" Print calibration png file """
plotms(LB_p4,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_p4_phase_vs_time.png'))

""" Apply the solutions """
applycal(vis=LB_cont_p3+'.ms', spw=LB_contspws, spwmap = LB_spw_mapping, gaintable=[LB_p4], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
LB_cont_p4 = prefix+'_ACASBLB_contp4'
os.system('rm -rf %s.ms*' % LB_cont_p4)
split(vis=LB_cont_p3+'.ms', outputvis=LB_cont_p4+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=LB_cont_p4+'.ms', imagename = LB_cont_p4, mask=LB_mask, threshold = '0.18mJy', deconvolver='multiscale', scales=LB_scales, savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_p4+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_p4+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder)
#AA_Tau_ACASBLB_contp4.image
#Beam 0.098 arcsec x 0.076 arcsec (-19.27 deg)
#Flux inside disk mask: 200.11 mJy
#Peak intensity of source: 3.59 mJy/beam
#rms: 3.03e-02 mJy/beam
#Peak SNR: 118.52

""" Clean down to 1 sigma for amplitude self-cal"""
tclean_wrapper(vis=LB_cont_p4+'.ms', imagename = LB_cont_p4, mask=LB_mask, threshold = '0.030mJy', deconvolver='multiscale', scales=LB_scales, savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize, robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_p4+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_p4+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder)
#AA_Tau_ACASBLB_contp4.image
#Beam 0.098 arcsec x 0.076 arcsec (-19.27 deg)
#Flux inside disk mask: 194.48 mJy
#Peak intensity of source: 3.62 mJy/beam
#rms: 2.93e-02 mJy/beam
#Peak SNR: 123.67


""" Amplitude self-cal"""
LB_ap0 = prefix+'_ACASBLB.ap0'
os.system('rm -rf '+LB_ap0)
gaincal(vis=LB_cont_p4+'.ms', caltable=LB_ap0, gaintype='T', spw=LB_contspws,
        refant=LB_refant, combine='spw, scan', calmode='ap', solint='inf', minsnr=5.0,
        minblperant=4, solnorm=False)
# No flagging
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(LB_ap0,xaxis='time', yaxis='GainPhase',iteraxis='spw')
plotms(LB_ap0,xaxis='time', yaxis='GainAmp',iteraxis='spw')

""" Print calibration png file """
plotms(LB_ap0,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_ap0_phase_vs_time.png'))
plotms(LB_ap0,xaxis='time', yaxis='GainAmp',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_ap0_amp_vs_time.png'))

""" Apply the solutions """
applycal(vis=LB_cont_p4+'.ms', spw=LB_contspws, spwmap = LB_spw_mapping,
         gaintable=[LB_ap0], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
LB_cont_ap0 = prefix+'_ACASBLB_contap0'
os.system('rm -rf %s.ms*' % LB_cont_ap0)
split(vis=LB_cont_p4+'.ms', outputvis=LB_cont_ap0+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
#again clean down to 1 sigma
tclean_wrapper(vis=LB_cont_ap0+'.ms', imagename = LB_cont_ap0, mask=LB_mask,
               threshold = '0.030mJy', deconvolver='multiscale', scales=LB_scales,
               savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize,
               robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_ap0+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_ap0+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder)
#AA_Tau_ACASBLB_contap0.image
#Beam 0.096 arcsec x 0.075 arcsec (-19.65 deg)
#Flux inside disk mask: 196.48 mJy
#Peak intensity of source: 3.46 mJy/beam
#rms: 2.84e-02 mJy/beam
#Peak SNR: 121.81

# lower rms noise

""" Try ampl self-cal on scan length intervals"""
LB_ap1 = prefix+'_ACASBLB.ap1'
os.system('rm -rf '+LB_ap1)
gaincal(vis=LB_cont_ap0+'.ms', caltable=LB_ap1, gaintype='T', spw=LB_contspws,
        refant=LB_refant, combine='spw, scan', calmode='ap', solint='360s', minsnr=5.0,
        minblperant=4, solnorm=False)
# Maximum 3/37 solutions flagged for SB and 32/49 for LB (average of about 12/49)
""" Inspect gain tables interactively and decide whether to manually flag something"""
plotms(LB_ap1,xaxis='time', yaxis='GainPhase',iteraxis='spw')
# 1 solution flagged
plotms(LB_ap1,xaxis='time', yaxis='GainAmp',iteraxis='spw')

""" Print calibration png file """
plotms(LB_ap1,xaxis='time', yaxis='GainPhase',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_ap1_phase_vs_time.png'))
plotms(LB_ap1,xaxis='time', yaxis='GainAmp',iteraxis='spw',exprange='all',
       overwrite=True,showgui=False,
       plotfile=os.path.join(LB_selfcal_folder,prefix+'_LB_gain_ap1_amp_vs_time.png'))

""" Apply the solutions """
applycal(vis=LB_cont_ap0+'.ms', spw=LB_contspws, spwmap = LB_spw_mapping,
         gaintable=[LB_ap1], interp='linearPD', calwt=True, applymode='calonly')

""" Split off a corrected MS """
LB_cont_ap1 = prefix+'_ACASBLB_contap1'
os.system('rm -rf %s.ms*' % LB_cont_ap1)
split(vis=LB_cont_ap0+'.ms', outputvis=LB_cont_ap1+'.ms', datacolumn='corrected')

""" Image the results; check the resulting map """
tclean_wrapper(vis=LB_cont_ap1+'.ms', imagename = LB_cont_ap1, mask=LB_mask,
               threshold = '0.028mJy', deconvolver='multiscale', scales=LB_scales,
               savemodel = 'modelcolumn', imsize=LB_imsize, cellsize=LB_cellsize,
               robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_ap1+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_ap1+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=LB_selfcal_folder)
#AA_Tau_ACASBLB_contap1.image
#Beam 0.095 arcsec x 0.075 arcsec (-20.20 deg)
#Flux inside disk mask: 196.59 mJy
#Peak intensity of source: 3.42 mJy/beam
#rms: 2.78e-02 mJy/beam
#Peak SNR: 123.15


self_caled_LB_visibilities = {'p1':LB_cont_p1,
                              'p2':LB_cont_p2,
                              'p3':LB_cont_p3,
                              'p4':LB_cont_p4,
                              'ap0':LB_cont_ap0,
                              'ap1':LB_cont_ap1}

for vis in self_caled_LB_visibilities.values(): 
    listobs(vis=vis+'.ms',listfile=vis+'.ms.txt',overwrite=True)



for self_cal_step,self_caled_vis in self_caled_LB_visibilities.items():
    for EB_key,spw in zip(SB_EBs,SB_EB_spws):
        nametemplate = f'{prefix}_SB_{EB_key}_{self_cal_step}_compare_amp_vs_time'
        visibilities = [self_caled_vis+'.ms',LB_cont_p0+'.ms']
        plot_amp_vs_time_comparison(
                nametemplate=nametemplate,visibilities=visibilities,spw=spw,
                uvrange=uv_ranges['SB'],output_folder=LB_selfcal_folder)


LB_EBs = ('EB0','EB1','EB2','EB3')
LB_EB_spws = ('32,33,34,35', '36,37,38,39', '40,41,42,43', '44,45,46,47') #fill out by referring to listobs output

for self_cal_step,self_caled_vis in self_caled_LB_visibilities.items():
    for EB_key,spw in zip(LB_EBs,LB_EB_spws):
        nametemplate = f'{prefix}_LB_{EB_key}_{self_cal_step}_compare_amp_vs_time'
        visibilities = [self_caled_vis+'.ms',LB_cont_p0+'.ms']
        #we plot the whole uv range because sources are unresolved for ACA
        plot_amp_vs_time_comparison(
                nametemplate=nametemplate,visibilities=visibilities,spw=spw,
                uvrange=uv_ranges['LB'],output_folder=LB_selfcal_folder)


#flux_comparison_ref_EB is set to the EB of the combined ACASBLB data that corresponds
#to flux_ref_EB
#However, if flux_ref_EB is not an SB EB, you should set flux_comparison_ref_EB to
#an SB EB, since only SB has overlapping baselines with both ACA and LB
#at this point in the script, SB should be correctly scaled and decoherence removed,
#so it should be ok to use SB as a reference
flux_comparison_ref_EB = 5

for self_cal_step,vis in self_caled_LB_visibilities.items():
    nametemplate = f'{prefix}_ACASBLB_cont{self_cal_step}_EB'
    split_all_obs(msfile=vis+'.ms',nametemplate=nametemplate)
    for i in range(number_of_EBs['ACA']+number_of_EBs['SB']+number_of_EBs['LB']):
        export_MS(f'{nametemplate}{i}.ms')
    #compare flux scale of SB and LB (not ACA)
    for i in range(number_of_EBs['ACA'], number_of_EBs['ACA']+number_of_EBs['SB']+number_of_EBs['LB']):
        reference = f'{nametemplate}{flux_comparison_ref_EB}.vis.npz'
        output = f'flux_comparison_EB{i}_to_EB{flux_comparison_ref_EB}'\
                       +f'_ACASBLB_{self_cal_step}.png'
        plot_label = os.path.join(LB_selfcal_folder,output)
        estimate_flux_scale(reference=reference,
                            comparison=f'{nametemplate}{i}.vis.npz',
                            incl=incl, PA=PA,
                            plot_label=plot_label)
    filelist = [f'{nametemplate}{i}.vis.npz' for i in range(number_of_EBs['ACA']+number_of_EBs['SB']+number_of_EBs['LB'])]
    fluxscale = [1.,]*(number_of_EBs['ACA']+number_of_EBs['SB']+number_of_EBs['LB'])
    plot_deprojected(filelist=filelist,fluxscale=fluxscale, PA=PA, incl=incl,
                     show_err=True,
                     plot_label=os.path.join(LB_selfcal_folder,
                                             f'deprojected_vis_profiles_ACASBLB_{self_cal_step}.png'))

list_npz_files = []

list_npz_files = [f'{prefix}_ACASBLB_contap1_EB0.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB1.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB2.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB3.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB4.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB5.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB6.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB7.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB8.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB9.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB10.vis.npz',
                  f'{prefix}_ACASBLB_contap1_EB11.vis.npz',]

for i in np.arange(number_of_EBs['SB']+number_of_EBs['ACA']+number_of_EBs['LB']):
    estimate_flux_scale(reference=f'{prefix}_ACASBLB_contap1_EB5.vis.npz',
                            comparison=f'{prefix}_ACASBLB_contap1_EB'+str(i)+'.vis.npz',incl=incl,PA=PA,
                            plot_label=os.path.join(LB_selfcal_folder,f'{prefix}_ACASBLB_contap1_EB'+str(i)+'_to_ACASBLB_contap1_EB5_comparison.png'))

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB0.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 1.02720
#The scaling factor for gencal is 1.014 for your comparison measurement
#The error on the weighted mean ratio is 3.517e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB1.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 1.03062
#The scaling factor for gencal is 1.015 for your comparison measurement
#The error on the weighted mean ratio is 3.731e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB2.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 1.01959
#The scaling factor for gencal is 1.010 for your comparison measurement
#The error on the weighted mean ratio is 5.316e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB3.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 1.02492
#The scaling factor for gencal is 1.012 for your comparison measurement
#The error on the weighted mean ratio is 4.008e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB4.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 1.02677
#The scaling factor for gencal is 1.013 for your comparison measurement
#The error on the weighted mean ratio is 3.635e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB5.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 1.00000
#The scaling factor for gencal is 1.000 for your comparison measurement
#The error on the weighted mean ratio is 7.807e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB6.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 0.98263
#The scaling factor for gencal is 0.991 for your comparison measurement
#The error on the weighted mean ratio is 7.848e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB7.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 0.98914
#The scaling factor for gencal is 0.995 for your comparison measurement
#The error on the weighted mean ratio is 7.337e-04, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB8.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 0.98031
#The scaling factor for gencal is 0.990 for your comparison measurement
#The error on the weighted mean ratio is 1.443e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB9.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 0.98290
#The scaling factor for gencal is 0.991 for your comparison measurement
#The error on the weighted mean ratio is 1.357e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB10.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 0.97551
#The scaling factor for gencal is 0.988 for your comparison measurement
#The error on the weighted mean ratio is 1.390e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor

#The ratio of the fluxes of AA_Tau_ACASBLB_contap1_EB11.vis.npz to
#AA_Tau_ACASBLB_contap1_EB5.vis.npz is 0.97609
#The scaling factor for gencal is 0.988 for your comparison measurement
#The error on the weighted mean ratio is 1.233e-03, although it's likely that
#the weights in the measurement sets are too off by some constant factor


# All flux offsets within 4%


""" Split out final continuum ms table, with a 30s timebin
"""
LB_cont_averaged = prefix+'_time_ave_continuum'
os.system(f'rm -rf {LB_cont_averaged}.ms*')
split(vis=LB_cont_ap1+'.ms', outputvis=LB_cont_averaged+'.ms', datacolumn='data',
      keepflags=False, timebin='30s')


"""
Now apply these solutions to the line data
"""
calibrate_linedata_folder = get_figures_folderpath('10_apply_cal_to_lines')
make_figures_folder(calibrate_linedata_folder)

""" Check that lines are not flagged in not averaged data"""
for params in data_params.values():
    plotms(vis=params['vis'],
            xaxis='channel',
            yaxis='amplitude',
            field=params['field'],
            ydatacolumn='data',
            avgtime='1e8',
            avgscan=True,
            avgbaseline=True,
            coloraxis='corr',
            iteraxis='spw',
            showgui = False,
            exprange='all',
            plotfile=os.path.join(calibrate_linedata_folder,
                                  prefix+'_'+params['name']+'_chan-v-amp_preselfcal_after_flagging.png')
            )


""" Apply gaintables of individual EBs"""
for params in data_params.values():
    single_EB_p1 = prefix+'_'+params['name']+'_initcont.p1'
    applycal(vis=prefix+'_'+params['name']+'.ms', spw='0~3',spwmap=[0,0,0,0],
             gaintable=[single_EB_p1], interp='linearPD', applymode='calonly', calwt=True)
    split(vis=prefix+'_'+params['name']+'.ms',
          outputvis=prefix+'_'+params['name']+'_no_ave_selfcal.ms',
          datacolumn='corrected')

###
# ALIGN DATA: we re-align the no_ave data, as we have done for the "initcont" ms tables.
# We go from *_no_ave_selfcal.ms to *_no_ave_shift.ms

reference_ms = {'LB':reference_for_LB_alignment,
                'SB':reference_for_SB_alignment,
                'ACA':reference_for_ACA_alignment}
for params in data_params.values():
    unshifted_ms = f'{prefix}_{params["name"]}_no_ave_selfcal.ms'
    array_key,_ = params['name'].split('_') #LB, SB or ACA
    offset = alignment_offsets[params['name']]
    #npix and cell_size are not needed because we do not fit any offset
    alignment.align_measurement_sets(
            reference_ms=reference_ms[array_key],align_ms=[unshifted_ms],
            align_offsets=[offset],npix=None,cell_size=None)

#applying shift [0.0, 0.0] to AA_Tau_LB_EB0_no_ave_selfcal.ms
#applying shift [-0.016481, 0.012814] to AA_Tau_LB_EB1_no_ave_selfcal.ms.
#applying shift [0.0042971, 0.0037993] to AA_Tau_LB_EB2_no_ave_selfcal.ms
#applying shift [-0.017586, 0.013933] to AA_Tau_LB_EB3_no_ave_selfcal.ms

#applying shift [0.008156, 0.10176] to AA_Tau_SB_EB0_no_ave_selfcal.ms
#applying shift [0.0090143, -0.059566] to AA_Tau_SB_EB1_no_ave_selfcal.ms
#applying shift [0.047804, 0.064352] to AA_Tau_SB_EB2_no_ave_selfcal.ms

#applying shift [-0.077447, 0.010629] to AA_Tau_ACA_EB0_no_ave_selfcal.ms
#applying shift [0.022049, -0.036367] to AA_Tau_ACA_EB1_no_ave_selfcal.ms
#applying shift [-0.21914, 0.29901] to AA_Tau_ACA_EB2_no_ave_selfcal.ms
#applying shift [-0.16025, 0.01728] to AA_Tau_ACA_EB3_no_ave_selfcal.ms
#applying shift [-0.098011, 0.027225] to AA_Tau_ACA_EB4_no_ave_selfcal.ms

###


"""
If you have re-scaled fluxes, you need to re-scale the *no_ave* EBs as well, for example:
rescale_flux(f'{prefix}_LB_EB0_no_ave_selfcal_shift.ms', [1.044])
"""
rescale_flux(vis=prefix+'_ACA_EB2_no_ave_selfcal_shift.ms', gencalparameter=[0.979])
rescale_flux(vis=prefix+'_SB_EB2_no_ave_selfcal_shift.ms', gencalparameter=[0.926])
#Splitting out rescaled values into new MS: AA_Tau_ACA_EB2_no_ave_selfcal_shift_rescaled.ms
#Splitting out rescaled values into new MS: AA_Tau_SB_EB2_no_ave_selfcal_shift_rescaled.ms

listobs(vis=prefix+'_ACA_EB2_no_ave_selfcal_shift_rescaled.ms',listfile=prefix+'_ACA_EB2_no_ave_selfcal_shift_rescaled.ms'+'.txt',overwrite=True)
listobs(vis=prefix+'_SB_EB2_no_ave_selfcal_shift_rescaled.ms',listfile=prefix+'_SB_EB2_no_ave_selfcal_shift_rescaled.ms'+'.txt',overwrite=True)



""" Concat not averaged ACA data """
ACA_combined = prefix+'_ACA_no_ave_concat'
os.system('rm -rf %s.ms*' % ACA_combined)
concat(vis=[f'{prefix}_ACA_EB0_no_ave_selfcal_shift.ms',
            f'{prefix}_ACA_EB1_no_ave_selfcal_shift.ms',
            f'{prefix}_ACA_EB2_no_ave_selfcal_shift_rescaled.ms',
            f'{prefix}_ACA_EB3_no_ave_selfcal_shift.ms',
            f'{prefix}_ACA_EB4_no_ave_selfcal_shift.ms'],
       concatvis=ACA_combined+'.ms', dirtol='0.1arcsec', copypointing=False)

listobs(vis=ACA_combined+'.ms',listfile=ACA_combined+'.ms.txt',overwrite=True)

#For the first cal table, we did not combine spws, so every spectral window is mapped onto itself.
#In this case we had 3EBs, if you had e.g. 4EBs you will have something like:
#spwmap = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],ACA_spw_mapping]
#Double check the number of ACA EBs you have and modify accordingly

# BE CAREFUL HERE
# using gaintables from iteration2
applycal(vis=ACA_combined+'.ms', spw='0~19',
         gaintable=[prefix+'_ACA_iteration2.p1',prefix+'_ACA_iteration2.p2'],
         spwmap = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],ACA_spw_mapping],interp=['linear','linearPD'], calwt=True,
         applymode='calonly',flagbackup=False)

os.system('rm -rf '+prefix+'_ACA_no_ave_selfcal.ms*')
split(vis=ACA_combined+'.ms', outputvis=prefix+'_ACA_no_ave_selfcal.ms',
      datacolumn='corrected')

""" Concat not averaged SB data """
SB_combined = prefix+'_ACASB_no_ave_concat'
os.system('rm -rf %s.ms*' % SB_combined)

concat(vis=[f'{prefix}_ACA_no_ave_selfcal.ms',
            f'{prefix}_SB_EB0_no_ave_selfcal_shift.ms',
            f'{prefix}_SB_EB1_no_ave_selfcal_shift.ms',
            f'{prefix}_SB_EB2_no_ave_selfcal_shift_rescaled.ms'],
            concatvis=SB_combined+'.ms',
            dirtol='0.1arcsec',copypointing=False)

listobs(vis=SB_combined+'.ms',listfile=SB_combined+'.ms.txt',overwrite=True)

#Be careful here to be sure to apply the right gain tables, in particular,
#if you did the SB self-cal twice (e.g. SB selfcal #1 on flux-unscaled MS, SB selfcal
# #2 on scaled MS)

# using gaintables from iteration2
applycal(vis=SB_combined+'.ms', spw='0~31',
         gaintable=[prefix+'_ACASB_iteration2.p1',prefix+'_ACASB_iteration2.p2',prefix+'_ACASB_iteration2.p3', 
                    prefix+'_ACASB_iteration2.p4', prefix+'_ACASB_iteration2.p5'],
         spwmap = [SB_spw_mapping]*5,interp=['linearPD']*5, calwt=True,
         applymode='calonly',flagbackup=False)

os.system('rm -rf '+prefix+'_ACASB_no_ave_selfcal.ms*')
split(vis=SB_combined+'.ms', outputvis=prefix+'_ACASB_no_ave_selfcal.ms',
      datacolumn='corrected')

""" Concat not averaged LB data """

LB_combined = prefix+'_ACASBLB_no_ave_concat'
os.system('rm -rf %s.ms*' % LB_combined)

concat(vis=[f'{prefix}_ACASB_no_ave_selfcal.ms',
            f'{prefix}_LB_EB0_no_ave_selfcal_shift.ms',
            f'{prefix}_LB_EB1_no_ave_selfcal_shift.ms',
            f'{prefix}_LB_EB2_no_ave_selfcal_shift.ms',
            f'{prefix}_LB_EB3_no_ave_selfcal_shift.ms'],
            concatvis=LB_combined+'.ms',
            dirtol='0.1arcsec',copypointing=False)

listobs(vis=LB_combined+'.ms',listfile=LB_combined+'.ms.txt',overwrite=True)

applycal(vis=LB_combined+'.ms', spw='0~47',
         gaintable=[prefix+'_ACASBLB.p1',prefix+'_ACASBLB.p2',prefix+'_ACASBLB.p3',
                    prefix+'_ACASBLB.p4',prefix+'_ACASBLB.ap0',prefix+'_ACASBLB.ap1'],
         spwmap = [LB_spw_mapping]*6,interp=['linearPD']*6,
         calwt=True, applymode='calonly',flagbackup=False)

ACASBLB_no_ave_final = f'{prefix}_ACASBLB_no_ave_selfcal_time_ave.ms'
os.system(f'rm -rf {ACASBLB_no_ave_final}*')
""" time average of 30s, tests show there is no different with data without time average """
split(vis=LB_combined+'.ms', outputvis=ACASBLB_no_ave_final,
      datacolumn='corrected', timebin = '30s', keepflags=False)

# Check that solutions have been applied correctly by flagging the line data, averaging and imaging continuum
# continuum has to be the same imaged in the last step of the self-cal
complete_dataset_dict = {'vis' : ACASBLB_no_ave_final,
                         'name' : 'ACASBLB_concat',
                         'field' : fields['LB'],
                         'line_spws': np.array([0,2,3,
                                                4,6,7,
                                                8,10,11,
                                                12,14,15,
                                                16,18,19,
                                                20,22,23,
                                                24,26,27,
                                                28,30,31,
                                                32,34,35,
                                                36,38,39,
                                                40,42,43,
                                                44,46,47]), # list of spws containing lines
                         'line_freqs': np.array([rest_freq_13CO,rest_freq_CS,rest_freq_12CO]*9), #frequencies (Hz) corresponding to line_spws
                         'cont_spws': None,
                         'width_array': None,
                         }

# Flagchannels input string for ACA_EB0: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB1: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB2: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB3: '0:963~3131, 2:925~3174, 3:919~3187'
# Flagchannels input string for ACA_EB4: '0:963~3131, 2:925~3174, 3:919~3187'

# Flagchannels input string for SB_EB0: '0:838~3006, 2:792~3041, 3:791~3059'
# Flagchannels input string for SB_EB1: '0:838~3006, 2:792~3041, 3:791~3059'
# Flagchannels input string for SB_EB2: '0:838~3006, 2:792~3041, 3:791~3059'
 
# Flagchannels input string for LB_EB0: '0:837~3005, 2:793~3042, 3:788~3056'
# Flagchannels input string for LB_EB1: '0:837~3005, 2:793~3042, 3:788~3056'
# Flagchannels input string for LB_EB2: '0:837~3005, 2:793~3042, 3:788~3056'
# Flagchannels input string for LB_EB3: '0:837~3005, 2:793~3042, 3:788~3056'


#use the output of get_flagchannels at the beginning of the script to define fitspw
fitspw =  '0:963~3131, 1:0, 2:925~3174, 3:919~3187,'\
         +'4:963~3131, 5:0, 6:925~3174, 7:919~3187,'\
         +'8:963~3131, 9:0, 10:925~3174, 11:919~3187,'\
         +'12:963~3131, 13:0, 14:925~3174, 15:919~3187,'\
         +'16:963~3131, 17:0, 18:925~3174, 19:919~3187,'\
         +'20:838~3006, 21:0, 22:792~3041, 23:791~3059,'\
         +'24:838~3006, 25:0, 26:792~3041, 27:791~3059,'\
         +'28:838~3006, 29:0, 30:792~3041, 31:791~3059,'\
         +'32:837~3005, 33:0, 34:793~3042, 35:788~3056,'\
         +'36:837~3005, 37:0, 38:793~3042, 39:788~3056,'\
         +'40:837~3005, 41:0, 42:793~3042, 43:788~3056,'\
         +'44:837~3005, 45:0, 46:793~3042, 47:788~3056'

avg_cont(ms_dict=complete_dataset_dict, output_prefix=prefix, flagchannels = fitspw,
         contspws = complete_dataset_dict['cont_spws'], width_array=complete_dataset_dict['width_array'])

# image the avg then cal with same parameters as in last step of self-cal
tclean_wrapper(vis=LB_cont_averaged+'.ms',
               imagename = LB_cont_averaged+'_image', mask=LB_mask,
               threshold = '0.028mJy', deconvolver='multiscale', scales=LB_scales,
               imsize=LB_imsize, cellsize=LB_cellsize,
               robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(LB_cont_averaged+'_image'+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(LB_cont_averaged+'_image'+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=calibrate_linedata_folder)
#AA_Tau_time_ave_continuum_image.image
#Beam 0.095 arcsec x 0.075 arcsec (-20.20 deg)
#Flux inside disk mask: 196.56 mJy
#Peak intensity of source: 3.42 mJy/beam
#rms: 2.78e-02 mJy/beam
#Peak SNR: 123.18

# image the cal then avg with same parameters as in last step of self-cal
complete_dataset_image = prefix+'_'+complete_dataset_dict['name']+'_initcont_image'
tclean_wrapper(vis=prefix+'_'+complete_dataset_dict['name']+'_initcont.ms',
               imagename = complete_dataset_image, mask=LB_mask,
               threshold = '0.028mJy', deconvolver='multiscale', scales=LB_scales,
               imsize=LB_imsize, cellsize=LB_cellsize,
               robust=0.5, interactive=False, parallel=use_parallel, gridder='mosaic')
estimate_SNR(complete_dataset_image+'.image', disk_mask = LB_mask, noise_mask = noise_annulus_LB)
generate_image_png(complete_dataset_image+'.image',plot_sizes=image_png_plot_sizes['LB'],
                   color_scale_limits=[-3*rms_LB,10*rms_LB],save_folder=calibrate_linedata_folder)
#AA_Tau_ACASBLB_concat_initcont_image.image
#Beam 0.095 arcsec x 0.075 arcsec (-20.20 deg)
#Flux inside disk mask: 196.63 mJy
#Peak intensity of source: 3.42 mJy/beam
#rms: 2.78e-02 mJy/beam
#Peak SNR: 123.15

# Plot ratio of cal then avg, and avg then cal. It should be equal to ~one (only difference being the time average):
ref_image = LB_cont_averaged+'_image'+'.image'
os.system('rm -rf '+complete_dataset_image+'.ratio')
immath(imagename=[ref_image,complete_dataset_image+'.image'],mode='evalexpr',
       outfile=complete_dataset_image+'.ratio',
       expr='iif(IM0 > 3*'+str(rms_LB)+', IM1/IM0, 0)'
       )
generate_image_png(f'{complete_dataset_image}.ratio',plot_sizes=[2*mask_semimajor,2*mask_semimajor],
                   color_scale_limits=[0.5,1.5],image_units='ratio',
                   save_folder=calibrate_linedata_folder)
# Check OK, the ratio is equal to ~one

# Do continuum subtraction
contsub_vis = f'{ACASBLB_no_ave_final}.contsub'
os.system(f'rm -rf {contsub_vis}*')
uvcontsub(vis=ACASBLB_no_ave_final, spw='0~47', fitspw=fitspw,
          excludechans=True, solint='int', fitorder=1, want_cont=False)

# Split final ms table into separate spws for 12CO, 13CO, CS and continuum

#12CO
vis_12CO = ACASBLB_no_ave_final[:-3]+'_12CO.ms'
os.system(f'rm -rf {vis_12CO}*')
spw_12CO = '3,7,11,15,19,23,27,31,35,39,43,47'

split(vis=ACASBLB_no_ave_final,outputvis=vis_12CO,spw=spw_12CO,
      datacolumn='data', keepflags=False)
listobs(vis=vis_12CO, listfile=vis_12CO+'.txt', overwrite=True)

split(vis=contsub_vis,outputvis=f'{vis_12CO}.contsub',spw=spw_12CO,
      datacolumn='data', keepflags=False)
listobs(vis=vis_12CO+'.contsub', listfile=vis_12CO+'.contsub.txt', overwrite=True)

#13CO
vis_13CO = ACASBLB_no_ave_final[:-3]+'_13CO.ms'
os.system(f'rm -rf {vis_13CO}*')
spw_13CO = '0,4,8,12,16,20,24,28,32,36,40,44'

split(vis=ACASBLB_no_ave_final,outputvis=vis_13CO,spw=spw_13CO,
      datacolumn='data', keepflags=False)
listobs(vis=vis_13CO, listfile=vis_13CO+'.txt', overwrite=True)

split(vis=contsub_vis,outputvis=f'{vis_13CO}.contsub',spw=spw_13CO,
      datacolumn='data', keepflags=False)
listobs(vis=vis_13CO+'.contsub', listfile=vis_13CO+'.contsub.txt', overwrite=True)

#CS
vis_CS = ACASBLB_no_ave_final[:-3]+'_CS.ms'
os.system(f'rm -rf {vis_CS}*')
spw_CS = '2,6,10,14,18,22,26,30,34,38,42,46'

split(vis=ACASBLB_no_ave_final,outputvis=vis_CS,spw=spw_CS,
      datacolumn='data', keepflags=False)
listobs(vis=vis_CS, listfile=vis_CS+'.txt', overwrite=True)

split(vis=contsub_vis,outputvis=f'{vis_CS}.contsub',spw=spw_CS,
      datacolumn='data', keepflags=False)
listobs(vis=vis_CS+'.contsub', listfile=vis_CS+'.contsub.txt', overwrite=True)

#continuum spw
vis_continuum = ACASBLB_no_ave_final[:-3]+'_contspw.ms'
os.system(f'rm -rf {vis_continuum}*')
spw_continuum = '1,5,9,13,17,21,25,29,33,37,41,45'

split(vis=ACASBLB_no_ave_final,outputvis=vis_continuum,spw=spw_continuum,
      datacolumn='data', keepflags=False)
listobs(vis=vis_continuum, listfile=vis_continuum+'.txt', overwrite=True)

split(vis=contsub_vis,outputvis=f'{vis_continuum}.contsub',spw=spw_continuum,
      datacolumn='data', keepflags=False)
listobs(vis=vis_continuum+'.contsub', listfile=vis_continuum+'.contsub.txt', overwrite=True)












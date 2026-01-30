#!/bin/bash
#SBATCH --job-name=J1852_gap0
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=1




cd /nexus/posix0/MIA-astro-env/myben/vawelke


#### Inject



cd /nexus/posix0/MIA-astro-env/myben/vawelke/J1852_gap0

python J1852_1core_injectloop.py


# (Move the .txt to injection folder)

mkdir -p injections
mv J1852_gap0_mpars.0.txt injections/



#### Pre-imaging to create psf and sumwt   (Change disk name in preimaign.py)


python prepimaging.py

#### Imaging 


cd /mnt/d/CPD_MPIA/Injection_Recovery_trial_fBf/J1852_robust2_0_gap0
/usr/local/bin/CASA/casa-6.6.1-17-pipeline-2024.1.0.8/bin/casa

 execfile("J1852_robust2_0_gap0_imageloop.py", globals())


#### Recovery

(Change recover_loop and assess_recovery disk name, hdu disk name, imdir disk name )

 cd /mnt/d/CPD_MPIA/Injection_Recovery_trial_fBf/J1852_robust2_0_gap0
python recover_loop.py
python assess_recovery.py

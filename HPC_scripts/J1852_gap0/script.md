
```python
# move stuff in
##### CHANGE  DISK_dictionary first!!!!

rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/DM_Tau_gap0/ \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/DM_Tau_gap0/

  rsync -av --progress \
/mnt/d/CPD_MPIA/HPC_scripts/J1615_gap*/ \
astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/

  rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/Source_codes/diskdictionaryr2_0_pix.py \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionaryr2_0.py



rsync -av --progress \
/mnt/d/CPD_MPIA/HPC_scripts/J1615_gap0 \
/mnt/d/CPD_MPIA/HPC_scripts/J1615_gap1 \
/mnt/d/CPD_MPIA/HPC_scripts/J1615_gap2 \
astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/


rsync -av --progress \
/mnt/d/CPD_MPIA/HPC_scripts/Trial_rval/* \
astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/

  rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/Source_codes/diskdictionary_r90/* \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/

# Mkdir residual and 
cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/

mkdir -p resid_vis mprofiles

source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
nohup python -u J1852_gap0_injectloop.py > inject.log 2>&1 & disown


deactivate

pgrep -af J1852_gap0_injectloop.py
tail -f inject.log

"ValueError: Bad value in power spectrum. The power spectrum must be positive and not contain any NaN values. This is likely due to your UVtable (incorrect units or weights),  or the deprojection being applied (incorrect geometry and/or phase center). Else you may want to adjust `rout` (ensure it is larger than the source) or `n` (up to ~300)." 


# Prepimaging
# RMB change cmask to [512pix,512pix]
cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/

source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"

export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py

CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
nohup python -u prepimaging.py > prepimaging.log 2>&1 & disown

deactivate 

tail -f prepimaging.log



# Imageloop

source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"

export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py


CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
nohup casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \
  -c  J1852_robust2_0_gap0_imageloop.py > imageloop.log 2>&1 & disown

tail -f imageloop.log

# Recovery

cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/
mkdir -p recoveries


source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

nohup python -u recover_loop.py > recover_loop.log 2>&1 & disown


tail -f recover_loop.log
deactivate



# Output and assess recovery

# move the last 2 residual fits in recoveries folder
cd resid_images

cp $(ls -1t *.fits | head -n 2) ../recoveries/

rsync -av --progress \
astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/recoveries/ \
/mnt/d/CPD_MPIA/HPC_scripts/J1852_gap0/recoveries/



# If open casa
export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"

export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py
casa --configfile $CASA_CONFIG

````



rsync -av --progress \
astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/recoveries/ \
/mnt/d/CPD_MPIA/HPC_scripts/Not_in/J1852_gap0/recoveries/




rsync -av --progress \
/mnt/d/CPD_MPIA/HPC_scripts/J1615_gap0 \
/mnt/d/CPD_MPIA/HPC_scripts/J1615_gap1 \
/mnt/d/CPD_MPIA/HPC_scripts/J1615_gap2 \
astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/


HD_34282_gap0 HD_34282_gap1 HD_34282_gap2 HD_34282_gap3  LkCa_15_gap0 LkCa_15_gap1


0.5 1.5  2.0  m1_5


rsync -av --progress \
astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0_rm1_5/recoveries/ \
/mnt/d/CPD_MPIA/HPC_scripts/Trial_rval/J1852_gap0_rm1_5/recoveries/


check my disk ditionaries first
note: lazy just do 0.0 1.0 , m0_5



------------------------------------------------------------------------
With bash

bash run_j1852_gap0.sh

# run bash in bg, and run scipts in foreground
nohup bash J1852_r0_5.sh > pipeline.log 2>&1 & disown 
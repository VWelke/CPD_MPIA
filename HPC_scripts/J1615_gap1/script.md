
```python
# move stuff in
##### CHANGE  DISK_dictionary first!!!!

rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/J1852_gap0/ \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/

  rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/Source_codes/diskdictionaryr2_0.py \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionaryr2_0.py

# Mkdir residual and 
cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/

mkdir -p resid_vis mprofiles

source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

nohup python -u J1852_gap0_injectloop.py > inject.log 2>&1 & disown


deactivate

pgrep -af J1852_gap0_injectloop.py
tail -f inject.log


# Prepimaging
# RMB change cmask to [512pix,512pix]
cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/

source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"

export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py

nohup python -u prepimaging.py > prepimaging.log 2>&1 & disown

deactivate 

tail -f prepimaging.log



# Imageloop

source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"

export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py

nohup casa --pipeline --configfile $CASA_CONFIG --nogui --nologger --nologfile -c "execfile('J1852_robust2_0_gap0_imageloop.py')" > imageloop.log 2>&1 & disown

deactivate

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


````
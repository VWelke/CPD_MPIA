


# Mkdir residual and 

mkdir -p resid_vis mprofiles

source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
nohup python -u J1852_gap0_injectloop.py > inject.log 2>&1 & disown

export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"

export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py

CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
nohup python -u prepimaging.py > prepimaging.log 2>&1 & disown

# Imageloop


export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"

export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py


CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
nohup casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \
  -c  J1852_robust2_0_gap0_imageloop.py > imageloop.log 2>&1 & disown

# Recovery

cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0/

mkdir -p recoveries


source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

nohup python -u recover_loop.py > recover_loop.log 2>&1 & disown



deactivate



# Output and assess recovery

# move the last 2 residual fits in recoveries folder
cd resid_images

cp $(ls -1t *.fits | head -n 2) ../recoveries/
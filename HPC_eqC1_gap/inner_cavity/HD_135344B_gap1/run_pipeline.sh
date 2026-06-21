#!/usr/bin/env bash

WORKDIR="/nexus/posix0/MIA-astro-env/myben/vawelke/HPC_eqC1_gap/inner_cavity/HD_135344B_gap1"
VENV="/nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate"
CASA_BIN="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin"
export CASA_CONFIG="/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py"

cd "$WORKDIR"
mkdir -p resid_vis mprofiles recoveries resid_images injections
source "$VENV"
export PATH="${CASA_BIN}:$PATH"

if [ -f "injections/HD_135344B_gap1_mpars.0.txt" ]; then
    echo "=== [HD_135344B_gap1] Inject already done, skipping ==="
else
    echo "=== [HD_135344B_gap1] Inject start: $(date) ==="
    OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    python -u HD_135344B_gap1_injectloop.py > inject.log 2>&1
    echo "=== Inject finished: $(date) ===" >> inject.log
fi

if compgen -G "*.custom.mask" > /dev/null 2>&1; then
    echo "=== Prepimaging already done, skipping ==="
else
    echo "=== [HD_135344B_gap1] Prepimaging start: $(date) ==="
    CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
    MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    python -u prepimaging.py > prepimaging.log 2>&1
    echo "=== Prepimaging finished: $(date) ===" >> prepimaging.log
fi

echo "=== [HD_135344B_gap1] Imageloop start: $(date) ==="
CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \
  -c HD_135344B_robust0_5_gap1_imageloop.py > imageloop.log 2>&1
echo "=== Imageloop finished: $(date) ===" >> imageloop.log

echo "=== [HD_135344B_gap1] Recovery start: $(date) ==="
python -u recover_loop.py > recover_loop.log 2>&1
echo "=== Recovery finished: $(date) ===" >> recover_loop.log

deactivate

cd "$WORKDIR/resid_images"
cp $(ls -1t *.fits | head -n 2) ../recoveries/

echo "=== [HD_135344B_gap1] Pipeline complete: $(date) ==="

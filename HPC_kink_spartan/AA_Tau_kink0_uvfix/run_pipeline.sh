#!/usr/bin/env bash

WORKDIR="/data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan/AA_Tau_kink0_uvfix"
VENV="/home/vwelke/venvs/venv-3.10.4/bin/activate"
CASA_BIN="/data/gpfs/projects/punim3000/software/casa/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin"
export CASA_CONFIG="/data/gpfs/projects/punim3000/MPIA/spartan/casa_config.py"

cd "$WORKDIR"
mkdir -p resid_vis mprofiles recoveries resid_images injections
module load GCCcore/11.3.0 Python/3.10.4
source "$VENV"
export PATH="${CASA_BIN}:$PATH"

if [ -f "injections/AA_Tau_gap2_mpars.0.txt" ]; then
    echo "=== [AA_Tau_kink0_uvfix] Inject already done, skipping ==="
else
    echo "=== [AA_Tau_kink0_uvfix] Inject start: $(date) ==="
    OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    python -u AA_Tau_kink0_injectloop.py > inject.log 2>&1
    echo "=== Inject finished: $(date) ===" >> inject.log
fi

if compgen -G "*.custom.mask" > /dev/null 2>&1; then
    echo "=== Prepimaging already done, skipping ==="
else
    echo "=== [AA_Tau_kink0_uvfix] Prepimaging start: $(date) ==="
    CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
    MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    python -u prepimaging.py > prepimaging.log 2>&1
    echo "=== Prepimaging finished: $(date) ===" >> prepimaging.log
fi

if ! compgen -G "*.custom.mask" > /dev/null 2>&1; then
    echo "=== ERROR: no *.custom.mask after prepimaging (see prepimaging.log) -- stopping ==="
    exit 1
fi

echo "=== [AA_Tau_kink0_uvfix] Imageloop start: $(date) ==="
CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \
  -c AA_Tau_robust0_5_kink0_imageloop.py > imageloop.log 2>&1
echo "=== Imageloop finished: $(date) ===" >> imageloop.log

echo "=== [AA_Tau_kink0_uvfix] Recovery start: $(date) ==="
python -u recover_loop.py > recover_loop.log 2>&1
echo "=== Recovery finished: $(date) ===" >> recover_loop.log

deactivate

cd "$WORKDIR/resid_images"
cp $(ls -1t *.fits | head -n 2) ../recoveries/

echo "=== [AA_Tau_kink0_uvfix] Pipeline complete: $(date) ==="

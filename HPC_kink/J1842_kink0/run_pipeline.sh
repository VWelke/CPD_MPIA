#!/usr/bin/env bash
set -euo pipefail

TARGET="J1842"
KINK="0"
WORKDIR="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/${TARGET}_kink${KINK}"
VENV="/nexus/posix0/MIA-astro-env/myben/vawelke/frank_env/bin/activate"
CASA_BIN="/nexus/posix0/MIA-astro-env/myben/casa-6.6.6-17-pipeline-2024.1.0.8-3/bin"
export CASA_CONFIG="/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py"

cd "$WORKDIR"
mkdir -p resid_vis mprofiles recoveries resid_images

source "$VENV"
export PATH="${CASA_BIN}:$PATH"

echo "=== [${TARGET} kink${KINK}] Inject start: $(date) ==="
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u ${TARGET}_kink${KINK}_injectloop.py > inject.log 2>&1
echo "=== [${TARGET} kink${KINK}] Inject done: $(date) ==="

echo "=== [${TARGET} kink${KINK}] Prepimaging start: $(date) ==="
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u prepimaging.py > prepimaging.log 2>&1
echo "=== [${TARGET} kink${KINK}] Prepimaging done: $(date) ==="

echo "=== [${TARGET} kink${KINK}] Imageloop start: $(date) ==="
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \
  -c ${TARGET}_robust2_0_kink${KINK}_imageloop.py > imageloop.log 2>&1
echo "=== [${TARGET} kink${KINK}] Imageloop done: $(date) ==="

echo "=== [${TARGET} kink${KINK}] Recovery start: $(date) ==="
python -u recover_loop.py > recover_loop.log 2>&1
echo "=== [${TARGET} kink${KINK}] Recovery done: $(date) ==="

deactivate
echo "=== [${TARGET} kink${KINK}] Pipeline complete: $(date) ==="

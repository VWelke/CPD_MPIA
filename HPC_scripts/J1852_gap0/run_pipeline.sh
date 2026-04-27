#!/usr/bin/env bash
set -euo pipefail

# Runs the full pipeline sequentially:
#   injection -> prepimaging -> CASA imageloop -> recovery
# Start it once (recommended):
#   nohup ./run_pipeline.sh > logs/pipeline.out 2>&1 & disown

PIPELINE_DIR="${PIPELINE_DIR:-/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate}"

CASA_BIN_DIR="${CASA_BIN_DIR:-/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin}"
CASA_CONFIG="${CASA_CONFIG:-/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "ERROR: Missing file: $p" >&2
    exit 1
  fi
}

log "cd ${PIPELINE_DIR}"
cd "${PIPELINE_DIR}"

mkdir -p resid_vis mprofiles resid_images recoveries logs

require_file "J1852_gap0_injectloop.py"
require_file "prepimaging.py"
require_file "J1852_robust2_0_gap0_imageloop.py"
require_file "recover_loop.py"
require_file "${VENV_ACTIVATE}"
require_file "${CASA_CONFIG}"

# Threading settings (keep your existing values)
export OMP_NUM_THREADS=2
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CASA_NUM_THREADS=2

# CASA environment
export PATH="${CASA_BIN_DIR}:$PATH"
export CASA_CONFIG

log "Activate venv: ${VENV_ACTIVATE}"
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

log "Stage 1/4: injection"
python -u J1852_gap0_injectloop.py > logs/inject.log 2>&1
log "Injection done"

log "Stage 2/4: prepimaging"
python -u prepimaging.py > logs/prepimaging.log 2>&1
log "Prepimaging done"

log "Stage 3/4: imageloop (CASA)"
# Run CASA in the foreground; this can take a long time.
# --nologger/--nologfile reduces extra CASA log spam.
casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \
  -c J1852_robust2_0_gap0_imageloop.py > logs/imageloop.log 2>&1
log "Imageloop done"

log "Stage 4/4: recovery"
python -u recover_loop.py > logs/recover_loop.log 2>&1
log "Recovery done"

# Optional convenience: copy the newest residual FITS into recoveries/
if compgen -G "resid_images/*.fits" > /dev/null; then
  log "Copy latest 2 FITS into recoveries/"
  (cd resid_images && cp -f $(ls -1t *.fits | head -n 2) ../recoveries/)
fi

log "Pipeline complete"

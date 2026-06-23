#!/usr/bin/env bash
# rsync_to_hpc.sh  —  push new_gap scripts + diskdictionary_newgap to HPC
# Run from Ubuntu terminal:  bash /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/new_gap/rsync_to_hpc.sh

HPC="astronode1"
HPC_BASE="/nexus/posix0/MIA-astro-env/myben/vawelke"
LOCAL_BASE="/mnt/d/CPD_MPIA"

echo "=== [1/2] diskdictionary_newgap.py ==="
rsync -av --progress "${LOCAL_BASE}/HPC_scripts/Source_codes/diskdictionary_r90/diskdictionary_newgap.py" "${HPC}:${HPC_BASE}/Source_codes/diskdictionary_r90/diskdictionary_newgap.py"

echo ""
echo "=== [2/2] inj_rev/new_gap/ pipeline folders ==="
rsync -av --progress --exclude='*.fits' --exclude='*.npz' --exclude='*.log' --exclude='*.txt' --exclude='resid_vis/' --exclude='mprofiles/' --exclude='recoveries/' --exclude='resid_images/' --exclude='injections/' --exclude='assess_figs/' --exclude='__pycache__/' "${LOCAL_BASE}/HPC_scripts/inj_rev/new_gap/" "${HPC}:${HPC_BASE}/inj_rev/new_gap/"

echo ""
echo "=== Done. SSH in and run the pipelines per README_new_gap.md ==="

# eqC1 gap/ — rsync + launch commands

## 1. rsync (run from Ubuntu/WSL terminal)

```bash
HPC="astronode1"
HPC_BASE="/nexus/posix0/MIA-astro-env/myben/vawelke"
LOCAL_BASE="/mnt/d/CPD_MPIA"

rsync -av --progress "${LOCAL_BASE}/HPC_eqC1_gap/diskdictionary_eqC1.py" "${HPC}:${HPC_BASE}/Source_codes/diskdictionary_r90/diskdictionary_eqC1.py"

rsync -av --progress --exclude='*.fits' --exclude='*.npz' --exclude='*.log' --exclude='*.txt' --exclude='resid_vis/' --exclude='mprofiles/' --exclude='recoveries/' --exclude='resid_images/' --exclude='injections/' --exclude='__pycache__/' "${LOCAL_BASE}/HPC_eqC1_gap/gap/" "${HPC}:${HPC_BASE}/inj_rev/eqC1_gap/gap/"
```

## 2. Launch all 7 gap pipelines (run on HPC)

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap"

for folder in HD_135344B_gap0 J1615_gap0 LkCa_15_gap0 MWC_758_gap1 V4046_Sgr_gap1 AA_Tau_gap0 AA_Tau_gap1; do
    echo "=== Launching $folder ==="
    nohup bash "${BASE}/${folder}/run_pipeline.sh" > "${BASE}/${folder}/pipeline.log" 2>&1 &
done
```

## 3. Check progress (run on HPC)

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap"

for folder in HD_135344B_gap0 J1615_gap0 LkCa_15_gap0 MWC_758_gap1 V4046_Sgr_gap1 AA_Tau_gap0 AA_Tau_gap1; do
    echo "--- $folder ---"
    ls "${BASE}/${folder}/injections/" 2>/dev/null | head -3
    tail -3 "${BASE}/${folder}/pipeline.log" 2>/dev/null
    echo ""
done
```

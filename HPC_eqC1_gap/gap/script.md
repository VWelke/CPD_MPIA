# eqC1 gap/ — rsync + launch commands

## 0. Fix BOM on HPC (one-time, for files already uploaded with BOM)

```bash
# Remove UTF-8 BOM from all .sh and .py on HPC (safe to re-run)
find /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap -name "*.sh" -o -name "*.py" | \
    xargs sed -i 's/^\xEF\xBB\xBF//'
```


## 1. rsync (run from Ubuntu/WSL terminal)

```bash
# push all eqC1_gap scripts to HPC
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/

# push diskdictionary_eqC1.py to Source_codes (where scripts import it from)
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/diskdictionary_eqC1.py astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/diskdictionary_eqC1.py
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


### Out put

for folder in HD_135344B_gap0 LkCa_15_gap0 MWC_758_gap1 V4046_Sgr_gap1 ; do
    echo "--- $folder ---"
    ls "${BASE}/${folder}/injections/" 2>/dev/null | head -3
    tail -3 "${BASE}/${folder}/pipeline.log" 2>/dev/null
    echo ""
done
# HPC_huang_gap — rsync + launch commands

## 1. rsync (run from Ubuntu/WSL terminal)

```bash
# push all huang_gap scripts to HPC
rsync -av --progress /mnt/d/CPD_MPIA/HPC_huang_gap/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/

# push diskdictionary_newgap.py to Source_codes (where scripts import it from)
rsync -av --progress /mnt/d/CPD_MPIA/HPC_huang_gap/diskdictionary_newgap.py astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/diskdictionary_newgap.py
```

---

## 2. Launch — gap/ (7 outer ring gaps) (run on HPC)

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap"

for folder in AA_Tau_gap0 AA_Tau_gap1 HD_135344B_gap0 J1615_gap0 LkCa_15_gap0 MWC_758_gap1 V4046_Sgr_gap1; do
    echo "=== Launching $folder ==="
    nohup bash "${BASE}/${folder}/run_pipeline.sh" > "${BASE}/${folder}/pipeline.log" 2>&1 &
done
```

## 3. Check progress — gap/ (run on HPC)

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap"

for folder in AA_Tau_gap0 AA_Tau_gap1 HD_135344B_gap0 J1615_gap0 LkCa_15_gap0 MWC_758_gap1 V4046_Sgr_gap1; do
    echo "--- $folder ---"
    ls "${BASE}/${folder}/injections/" 2>/dev/null | head -3
    tail -3 "${BASE}/${folder}/pipeline.log" 2>/dev/null
    echo ""
done
```

---

## 4. Launch — inner_cavity/ (10 inner cavities) (run on HPC)

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/inner_cavity"

for folder in CQ_Tau_gap0 DM_Tau_gap0 HD_135344B_gap1 J1604_gap0 J1615_gap1 J1852_gap0 LkCa_15_gap1 MWC_758_gap0 SY_Cha_gap0 V4046_Sgr_gap0; do
    echo "=== Launching $folder ==="
    nohup bash "${BASE}/${folder}/run_pipeline.sh" > "${BASE}/${folder}/pipeline.log" 2>&1 &
done
```

## 5. Check progress — inner_cavity/ (run on HPC)

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/inner_cavity"

for folder in CQ_Tau_gap0 DM_Tau_gap0 HD_135344B_gap1 J1604_gap0 J1615_gap1 J1852_gap0 LkCa_15_gap1 MWC_758_gap0 SY_Cha_gap0 V4046_Sgr_gap0; do
    echo "--- $folder ---"
    ls "${BASE}/${folder}/injections/" 2>/dev/null | head -3
    tail -3 "${BASE}/${folder}/pipeline.log" 2>/dev/null
    echo ""
done
```

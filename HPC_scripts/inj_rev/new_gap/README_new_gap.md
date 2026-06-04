# new_gap Injection-Recovery Pipeline — Run Guide

## 1. Upload all folders to HPC (run from Ubuntu terminal)

```bash
rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/Source_codes/diskdictionary_r90/diskdictionary_newgap.py astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/
```

```bash
rsync -av --progress --exclude='*.fits' --exclude='*.npz' --exclude='*.log' --exclude='*.txt' --exclude='resid_vis/' --exclude='mprofiles/' --exclude='recoveries/' --exclude='resid_images/' --exclude='injections/' --exclude='assess_figs/' --exclude='__pycache__/' /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/new_gap/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/new_gap/
```

---

## 2. Run pipelines (SSH into HPC first)

Each folder has a `run_pipeline.sh` that runs inject → prepimaging → imageloop → recovery.
Run in the background so it survives disconnect:

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/new_gap"

# AA_Tau (gaps 0-1)
for g in 0 1; do cd $BASE/AA_Tau_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# CQ_Tau (gap 0)
cd $BASE/CQ_Tau_gap0 && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown

# DM_Tau (gaps 0-1)
for g in 0 1; do cd $BASE/DM_Tau_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# HD_135344B (gaps 0-1)
for g in 0 1; do cd $BASE/HD_135344B_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# HD_143006 (gap 0)
cd $BASE/HD_143006_gap0 && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown

# HD_34282 (gap 0)
cd $BASE/HD_34282_gap0 && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown

# J1604 (gap 0)
cd $BASE/J1604_gap0 && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown

# J1615 (gaps 0-1)
for g in 0 1; do cd $BASE/J1615_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# J1842 (gap 0)
cd $BASE/J1842_gap0 && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown

# J1852 (gap 0)
cd $BASE/J1852_gap0 && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown

# LkCa_15 (gaps 0-1)
for g in 0 1; do cd $BASE/LkCa_15_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# MWC_758 (gaps 0-1)
for g in 0 1; do cd $BASE/MWC_758_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# SY_Cha (gap 0)
cd $BASE/SY_Cha_gap0 && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown
```

Or run one at a time (replace DISK with folder name):
```bash
cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/new_gap/DISK
nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown
```

---

## 3. Monitor progress

```bash
for d in /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/new_gap/*/; do echo "=== $(basename $d) ==="; ls $d; tail -3 ${d}pipeline.log 2>/dev/null; echo; done
```

```bash
pgrep -af run_pipeline.sh
```

---

## 4. Download recoveries (run from Ubuntu terminal)

```bash
for disk in AA_Tau_gap0 AA_Tau_gap1 CQ_Tau_gap0 DM_Tau_gap0 DM_Tau_gap1 HD_135344B_gap0 HD_135344B_gap1 HD_143006_gap0 HD_34282_gap0 J1604_gap0 J1615_gap0 J1615_gap1 J1842_gap0 J1852_gap0 LkCa_15_gap0 LkCa_15_gap1 MWC_758_gap0 MWC_758_gap1 SY_Cha_gap0; do rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/new_gap/${disk}/recoveries/ /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/new_gap/${disk}/recoveries/; done
```

---

## Pipeline stages per folder

| Script | What it does |
|---|---|
| `{target}_gap{N}_injectloop.py` | Frank fit + inject CPD flux into visibilities (uses `diskdictionary_newgap`) |
| `prepimaging.py` | Create custom clean mask |
| `{target}_robust0_5_gap{N}_imageloop.py` | CASA tclean + JvM correction → resid images |
| `recover_loop.py` | Measure peak flux in residual images → recoveries .txt |
| `assess_recovery.py` | Compute recovery fraction + false alarm → plots |

## Run log

| Disk/Gap | Started | Status | Notes |
|---|---|---|---|
| AA_Tau_gap0 | | | |
| AA_Tau_gap1 | | | |
| CQ_Tau_gap0 | | | |
| DM_Tau_gap0 | | | |
| DM_Tau_gap1 | | | |
| HD_135344B_gap0 | | | |
| HD_135344B_gap1 | | | |
| HD_143006_gap0 | | | |
| HD_34282_gap0 | | | |
| J1604_gap0 | | | |
| J1615_gap0 | | | |
| J1615_gap1 | | | |
| J1842_gap0 | | | |
| J1852_gap0 | | | |
| LkCa_15_gap0 | | | |
| LkCa_15_gap1 | | | |
| MWC_758_gap0 | | | |
| MWC_758_gap1 | | | |
| SY_Cha_gap0 | | | |

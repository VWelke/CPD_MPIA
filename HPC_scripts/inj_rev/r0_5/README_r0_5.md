# r0_5 Injection-Recovery Pipeline — Run Guide

## 1. Upload all folders to HPC (run from WSL terminal)

```bash
rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/r0_5/ \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5/
```

Or use the script:
```bash
bash /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/r0_5/rsync_to_hpc.sh
```

If you updated `diskdictionary_r90/diskdictionaryr0_5.py`, also push:
```bash
rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/Source_codes/diskdictionary_r90/ \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/
```

---

## 2. Run a pipeline (SSH into HPC first)

Each folder has a `run_pipeline.sh` that runs inject → prepimaging → imageloop → recovery sequentially.
Run it in the background so it keeps going after you disconnect:

```bash
BASE="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5"

# AA_Tau (gaps 0-3)
for g in 0 1 2 3; do cd $BASE/AA_Tau_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# DM_Tau (gaps 0-2)
for g in 0 1 2; do cd $BASE/DM_Tau_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# HD_135344B (gaps 0-1)
for g in 0 1; do cd $BASE/HD_135344B_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# HD_143006 (gaps 0-1)
for g in 0 1; do cd $BASE/HD_143006_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# HD_34282 (gaps 0-3)
for g in 0 1 2 3; do cd $BASE/HD_34282_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

# J1615 (gaps 0-2)
for g in 0 1 2; do cd $BASE/J1615_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done

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

# V4046_Sgr (gaps 0-1)
for g in 0 1; do cd $BASE/V4046_Sgr_gap$g && nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown; done
```

Or run one at a time (replace DISK with folder name):
```bash
cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5/DISK
nohup bash run_pipeline.sh > pipeline.log 2>&1 & disown
```

---

## 3. Monitor progress

```bash
# Check overall pipeline stage
tail -f /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5/AA_Tau_gap0/pipeline.log

# Check individual stage logs
tail -f inject.log
tail -f prepimaging.log
tail -f imageloop.log
tail -f recover_loop.log

# Check all running pipelines at once
pgrep -af run_pipeline.sh
```

---

## 4. Download recoveries (run from WSL terminal)

```bash
# Single disk
rsync -av --progress \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5/AA_Tau_gap0/recoveries/ \
  /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/r0_5/AA_Tau_gap0/recoveries/

# All at once (all 27 disk/gap combos)
for disk in \
  AA_Tau_gap0 AA_Tau_gap1 AA_Tau_gap2 AA_Tau_gap3 \
  DM_Tau_gap0 DM_Tau_gap1 DM_Tau_gap2 \
  HD_135344B_gap0 HD_135344B_gap1 \
  HD_143006_gap0 HD_143006_gap1 \
  HD_34282_gap0 HD_34282_gap1 HD_34282_gap2 HD_34282_gap3 \
  J1615_gap0 J1615_gap1 J1615_gap2 \
  J1842_gap0 J1852_gap0 \
  LkCa_15_gap0 LkCa_15_gap1 \
  MWC_758_gap0 MWC_758_gap1 \
  SY_Cha_gap0 \
  V4046_Sgr_gap0 V4046_Sgr_gap1; do
  rsync -av --progress \
    astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5/${disk}/recoveries/ \
    /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/r0_5/${disk}/recoveries/
done
```

---

## 5. Run assess_recovery.py locally (after downloading recoveries)

```bash
cd D:\CPD_MPIA\HPC_scripts\inj_rev\r0_5\AA_Tau_gap0
python assess_recovery.py
```

---

## Pipeline stages per folder

| Script | What it does |
|---|---|
| `{target}_gap{N}_injectloop.py` | Frank fit + inject CPD flux into .ms visibilities |
| `prepimaging.py` | Create custom clean mask |
| `{target}_robust0_5_gap{N}_imageloop.py` | CASA tclean + JvM correction → resid images |
| `recover_loop.py` | Measure peak flux in residual images → recoveries .txt |
| `assess_recovery.py` | Compute recovery fraction + false alarm → rprofs .txt + plots |

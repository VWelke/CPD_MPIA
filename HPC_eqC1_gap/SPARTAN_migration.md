# Spartan HPC migration notes

## Access

- Host: `spartan.hpc.unimelb.edu.au`
- Username: `vwelke`
- Remote project dir: `/data/gpfs/projects/punim3000/MPIA`
- SLURM account: `punim3000`

This sandbox has no `rsync` and no SSH key for Spartan (`ssh vwelke@spartan.hpc.unimelb.edu.au` returns
`Permission denied (publickey,...)`), so the commands below need to be run from a terminal that actually
has access (WSL/Ubuntu, per the existing `astronode1` workflow in `HPC_eqC1_gap/gap/script.md`).

## Which `_1sigma` folders still need the 500-injection run

All `*_1sigma` folders under `HPC_eqC1_gap/gap/` and `HPC_eqC1_gap/inner_cavity/` are configured for
`n_mocks_per_F = 500` (single 1-sigma flux bin) in their `*_injectloop.py`. Only 3 have a `recoveries/`
folder already (i.e. have been run): `gap/SY_Cha_gap0_1sigma`, `inner_cavity/J1615_gap1_1sigma`,
`inner_cavity/LkCa_15_gap1_1sigma`.

Still need to be run (12):

**gap/**
- DM_Tau_gap1_1sigma
- HD_135344B_gap0_1sigma
- J1615_gap0_1sigma
- LkCa_15_gap0_1sigma
- V4046_Sgr_gap1_1sigma  ← pilot folder for Spartan migration (see below)

**inner_cavity/**
- CQ_Tau_gap0_1sigma
- DM_Tau_gap0_1sigma
- HD_135344B_gap1_1sigma
- J1604_gap0_1sigma
- J1852_gap0_1sigma
- MWC_758_gap0_1sigma
- V4046_Sgr_gap0_1sigma

## Step 1 — rsync the pilot folder + its shared dependency

```bash
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/gap/V4046_Sgr_gap1_1sigma/ vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/HPC_eqC1_gap/gap/V4046_Sgr_gap1_1sigma/

rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/diskdictionary_eqC1.py vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/Source_codes/diskdictionary_r90/diskdictionary_eqC1.py
```

## Not solved yet — hardcoded old-cluster paths

`V4046_Sgr_gap1_injectloop.py`, `V4046_Sgr_robust0_5_gap1_imageloop.py`, `recover_loop.py`, and
`run_pipeline.sh` all hardcode absolute paths on the old `astronode1` cluster
(`/nexus/posix0/MIA-astro-env/myben/vawelke/...`). Moving just this folder to Spartan is **not** enough
to run it there — these still need Spartan equivalents and the scripts need editing to point at them:

- `Source_codes/` (`inject_CPD.py`, `reduction_utils.py`, `JvM_correction_brief.py`, `ImportMS.py`,
  `diskdictionary_r90/`)
- the `frank_env` Python venv
- the CASA install (`casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8`) + `casa_config.py`
- the `exoALMA_disk_data/measurement_set_spavg/` vis data (`.ms` and `npz/*.vis.npz`)

TODO before this pilot folder is actually runnable on Spartan: confirm where these live (or should be
copied to) under `/data/gpfs/projects/punim3000/MPIA`, then update the paths in the 4 files above.

## SLURM script (template — Spartan partition/module names TBD)

Spartan uses SLURM, unlike `astronode1` where `run_pipeline.sh` is just launched with `nohup`. Draft
below mirrors the same inject → image → recover sequence as `run_pipeline.sh`; fill in the `<...>`
placeholders once the dependencies above are sorted out.

```bash
#!/bin/bash
#SBATCH --job-name=V4046_Sgr_gap1_1sigma
#SBATCH --account=punim3000
#SBATCH --partition=<TBD>              # e.g. cascade / sapphire -- check `sinfo` on Spartan
#SBATCH --time=<TBD>                   # walltime
#SBATCH --cpus-per-task=2
#SBATCH --mem=<TBD>
#SBATCH --output=%x_%j.log

WORKDIR="/data/gpfs/projects/punim3000/MPIA/HPC_eqC1_gap/gap/V4046_Sgr_gap1_1sigma"
VENV="<TBD>/frank_env/bin/activate"          # Spartan path once venv is set up
module load <TBD-CASA-module>                 # or point at a CASA install path, as on astronode1

cd "$WORKDIR"
mkdir -p resid_vis mprofiles recoveries resid_images injections
source "$VENV"

if [ -f "injections/V4046_Sgr_gap1_mpars.0.txt" ]; then
    echo "=== Inject already done, skipping ==="
else
    OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    python -u V4046_Sgr_gap1_injectloop.py > inject.log 2>&1
fi

CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
casa --pipeline --nogui --nologger --nologfile \
  -c V4046_Sgr_robust0_5_gap1_imageloop.py > imageloop.log 2>&1

python -u recover_loop.py > recover_loop.log 2>&1

deactivate

cd "$WORKDIR/resid_images"
cp $(ls -1t *.fits | head -n 2) ../recoveries/
```

Submit with `sbatch submit_spartan.slurm` once the placeholders are filled in.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**CPD_MPIA** is a research project analyzing Circumplanetary Disk (CPD) signals in ALMA radio observations from the DSHARP survey (15 protoplanetary disks). The core workflow is: load FITS residual images → compute per-pixel SNR maps → compare against theoretical CPD emission models → run injection-recovery tests to establish detection limits → constrain planetary mass.

## Environment Setup

```bash
# Activate the Python 3.12 virtual environment (Windows)
.venv\Scripts\activate

# Dependencies (no requirements.txt — install manually):
# pip install numpy astropy gofish frank photutils joblib matplotlib pandas scipy
# CASA (casatools, casatasks) requires separate NRAO installation — NOT pip-installable
```

There is no build step. Scripts and notebooks are run directly.

## Running Code

Most analysis is done in Jupyter notebooks. Key standalone scripts:

```bash
# Run a CASA-based injection-recovery pipeline (requires CASA Python environment)
python Injection_Recovery_trial_fBf/Injection_Recovery_Pipeline.py

# HPC cluster job submission (SLURM — run on cluster, not locally)
# Scripts in HPC_scripts/ submit jobs per disk/gap combination
```

## Architecture & Data Flow

### Main Analysis Classes

**`Median_SNR/disk_residuals_median_SNR.py`** — `DiskResiduals_Median_SNR` is the primary class. For each disk it:
1. Loads FITS residual images (by CLEAN robust parameter: `"0.5"`, `"2.0"`)
2. Reads disk geometry (inclination, PA, center offsets) and ring/gap locations from `.txt` files
3. Builds 2D noise/SNR maps via radial intensity profiles (`create_sigma_mask`)
4. Detects point sources via `photutils`

**`Standard_dy_SNR/disk_residuals.py`** — older, simplified version of the same class; used for legacy analysis only.

### CPD Emission Models

Both models live in `CPD_Emission_Models/`:
- **`Zhu_model.py`** — viscous accretion disk model (Zhu et al. 2018). Inputs: Mstar, Mp, Mdot, orbital radius, Lstar, viscosity α. Outputs flux density at ALMA bands.
- **`Andrew_model.py`** — Andrews parametric model with irradiation + accretion heating.

The models are called over a 2D grid of (planet mass, accretion rate) and results plotted via `Median_SNR/cpd_plot_grid.py`.

### Injection-Recovery Pipeline

**`Injection_Recovery_trial_fBf/Injection_Recovery_Pipeline.py`** orchestrates:
1. Inject fake CPD flux into ALMA `.ms` visibility data
2. Image & deconvolve with CASA `tclean`
3. Measure recovered flux with `imfit`
4. Output recovery fraction per flux bin → detection limit curves

**`HPC_scripts/`** contains SLURM submission scripts and per-disk CASA reduction utilities (`Source_codes/reduction_utils.py`, `diskdictionaryr2_0.py`) that wrap the same pipeline for cluster execution.

### Shared Metadata

`Dictionaries.py` (exists in both `Median_SNR/` and `CPD_Emission_Models/`) defines:
- `disk_arr`: metadata for all 15 DSHARP disks (Mstar, distance, Lstar, ring/gap radii)
- `five_sigma_sources`, `kink_sources`, `strong_res_gaps`, etc.: detection tier catalogs

Pre-computed planet mass–radius and mass–luminosity lookup tables are in `utils/Lp_from_Mp.pkl` and `utils/Rp_from_Mp.pkl`.

### Serialized State

`disk_obj.pkl` (112 MB, git-ignored) stores pre-initialized `DiskResiduals_Median_SNR` objects for all 15 disks — avoids re-reading all FITS files on every notebook restart.

## Key Notebooks

| Notebook | Purpose |
|---|---|
| `Median_SNR/*.ipynb` | Plotting SNR maps and radial profiles |
| `CPD_Emission_Models/CPD_Emission_Models.ipynb` | Emission model grids |
| `Injection_Recovery_trial_fBf/*assess*.ipynb` | Recovery fraction analysis |
| `HPC_kink/` | Kink/planet signature detection trials |

### Gap Definition Methods & HPC Directory Structure

Three gap definition methods are used for injection-recovery, each in its own top-level directory:

| Method | Directory | Dictionary | Status |
|---|---|---|---|
| Original DSHARP gaps | `HPC_scripts/inj_rev/r0_5/` | `diskdictionaryr0_5.py` | Done |
| Visually redefined gaps | `HPC_scripts/inj_rev/new_gap/` | `diskdictionary_newgap.py` | In progress |
| Eq. C1 fitted gaps | **`HPC_eqC1_gap/`** (planned) | `diskdictionary_eqC1.py` (TBD) | Not started |

**`HPC_eqC1_gap/`** — planned top-level directory (sibling of `HPC_scripts/`, `HPC_kink/`):
- One subdirectory per disk/gap (same 19 targets as `new_gap/` but final list TBD — depends on Eq. C1 fits)
- Each disk folder contains: `injectloop.py`, `prepimaging.py`, `imageloop.py`, `recover_loop.py`, `run_pipeline.sh`
- Scripts are identical to `new_gap/` except the dictionary import (`diskdictionary_eqC1`)
- `diskdictionary_eqC1.py` will be filled in as Eq. C1 fits complete — **do not create this directory until gap parameters are defined**

**Eq. C1 gap model** (from paper Appendix): fits brightness temperature profile  
`T_b = T_0 * (r/0.1")^{-q} / (1 + Γ)` where `Γ = (δ_gap − 1) * exp[−(r − r_gap)² / (2σ²_gap)]`  
Fitted parameters per disk: `r_gap`, `σ_gap`, `δ_gap` → used as `rgap`, `wgap` in injection-recovery.

## CASA Dependency

Any script that imports `casatools` or `casatasks` requires a full CASA installation (NRAO's monolithic package). These scripts **cannot run** in the standard `.venv`. The HPC scripts are the primary users. `reduction_utils.py` is a CASA-only helper.

## Data File Conventions

- `.fits` — residual images, SNR maps (git-ignored)
- `.ms` — ALMA measurement sets (git-ignored, ~GB each)
- `.pkl` — serialized Python objects (disk_obj.pkl is 112 MB, git-ignored)
- `.npz`, `.csv` — intermediate numerical outputs (git-ignored)
- Disk trial outputs follow pattern: `{disk_name}_gap{N}/` (e.g., `AA_Tau_gap0/`)

# HPC_kink_spartan: rsync + submit + check commands

These are Spartan copies of the 60 `HPC_kink/` folders for LkCa 15, SY Cha, AA Tau, J1615 and J1842. They contain scripts only, with no old results.

Candidate positions come from `diskdictionary_eqC1.py`:
- r = r_planet from Pinte+2025 Table 2, converted to arcsec with the dictionary distance.
- az = the Figure 5 blue dot, in the azimuth convention that `recover_loop.py` uses (from `verify_figure5_blue_dot.ipynb`).

## Paths assumed on Spartan (check these exist first)

| what | path |
|---|---|
| these folders | `/data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan/<folder>` |
| Source_codes (+ `diskdictionary_r90/`) | `/data/gpfs/projects/punim3000/MPIA/Source_codes/` |
| vis data | `/data/gpfs/projects/punim3000/MPIA/exoALMA_disk_data/measurement_set_spavg/` |
| CASA | `/data/gpfs/projects/punim3000/MPIA/spartan/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin` |
| CASA config | `/data/gpfs/projects/punim3000/MPIA/spartan/casa_config.py` (copy from astronode1; fix any old paths inside it) |
| venv | `/home/vwelke/frank_env/bin/activate` |
| CASA measures (rep folders symlink these) | `/home/vwelke/.casa/data`, `/home/vwelke/.casa/measures.d` |

SLURM settings in each `submit.slurm`: `--partition=sapphire`, `--cpus-per-task=2`, `--mem=16G`, `--time=7-00:00:00`. If any of these are wrong, fix them in every folder at once. For example, from this folder: `sed -i 's/--partition=sapphire/--partition=cascade/' */submit.slurm`

## 1. rsync (run from Ubuntu/WSL terminal)

```bash
rsync -av --progress /mnt/d/CPD_MPIA/HPC_kink_spartan/ vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan/
```

```bash
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/diskdictionary_eqC1.py vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/Source_codes/diskdictionary_r90/diskdictionary_eqC1.py
```

## 2. Submit (on Spartan)

All folders except the `SY_Cha_kink0_1sigma*` ones:

```bash
cd /data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan && for d in */; do case $d in *1sigma*) continue;; esac; (cd $d && sbatch submit.slurm); done
```

The `SY_Cha_kink0_1sigma*` folders reuse the PSF, sumwt and custom clean mask from `../SY_Cha_kink0`. Submit them only once `SY_Cha_kink0` has finished prepimaging, i.e. once this prints a `.custom.mask` file:

```bash
ls -d /data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan/SY_Cha_kink0/*.custom.mask
```

```bash
cd /data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan && for d in SY_Cha_kink0_1sigma*/; do (cd $d && sbatch submit.slurm); done
```

## 3. Check progress

```bash
squeue -u vwelke
```

```bash
for d in /data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan/*/; do echo "=== $d"; ls "$d"; tail -n 3 "$d"slurm_*.log 2>/dev/null; ls "$d"recoveries 2>/dev/null | head -3; done
```

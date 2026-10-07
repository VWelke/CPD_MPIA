# Migrating Source_codes + exoALMA_disk_data to Spartan

Moves the shared code/data this HPC's eqC1_gap/gap/inner_cavity/kink/candidate
pipelines depend on, so Spartan has an exact snapshot of the script versions
actually used for these runs. Run all commands below from the local WSL
terminal (it already has SSH access to `astronode1`; `scp -3` brokers a
direct host-to-host transfer, data does not pass through the laptop).

Target base path on Spartan: `/data/gpfs/projects/punim3000/MPIA/spartan/`

## 1. Source_codes (647K)

```bash
scp -3 -r astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/Source_codes
```

## 2. exoALMA_disk_data (12G)

```bash
scp -3 -r astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/exoALMA_disk_data vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/
```

## 3. CASA (only if Spartan's OS is EL8-compatible -- check first)

```bash
# check Spartan's OS version before copying a prebuilt binary over
ssh vwelke@spartan.hpc.unimelb.edu.au "cat /etc/os-release"
```

If that's RHEL/AlmaLinux/Rocky 8 (or compatible), copy the exact CASA build
used for these runs (`casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8`) plus its
paired measures/calibration data:

```bash
scp -3 -r astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8 vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/spartan/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8
```

```bash
scp -3 -r astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/casa_data vwelke@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim3000/MPIA/spartan/casa_data
```

If the OS doesn't match, download a fresh CASA build for Spartan's specific
OS from https://casa.nrao.edu instead of fighting binary/glibc incompatibilities.

## 4. Python venv -- do NOT copy, recreate fresh instead

`venvs/frank_env` bakes absolute paths into `bin/activate` and the
interpreter shebang line, so a copied venv will not work at a new path on a
new machine. Recreate it on Spartan instead:

```bash
ssh vwelke@spartan.hpc.unimelb.edu.au
python3 -m venv frank_env
source frank_env/bin/activate
pip install numpy astropy frank photutils joblib matplotlib pandas scipy
```

## After migrating

Every script under `Source_codes/` and the per-disk pipeline folders
hardcodes the current HPC's absolute paths
(`/nexus/posix0/MIA-astro-env/myben/vawelke/...`) for `exoALMA_disk_data`,
`Source_codes`, the CASA install, and the venv. These need updating to the
new Spartan paths (`/data/gpfs/projects/punim3000/MPIA/spartan/...`) in
whichever per-disk scripts get used on Spartan -- this is the "update them
anyway" step, not handled by the copy itself.

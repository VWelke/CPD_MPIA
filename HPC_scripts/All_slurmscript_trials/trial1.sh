#!/bin/bash
#SBATCH --job-name=trial1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


export PATH="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin:$PATH"
export CASA_CONFIG=/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py




#### Inject



cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0_trial1/

python -u J1852_gap0_injectloop.py

echo "Injection completed."

#### Pre-imaging to create psf and sumwt   (Change disk name in preimaign.py)


python -u prepimaging.py
echo "Pre-imaging completed."

#### Imaging 


casa --pipeline --configfile "$CASA_CONFIG" --nogui -c J1852_robust2_0_gap0_imageloop.py

echo "Imaging completed."
#### Recovery

python -u recover_loop.py
python -u assess_recovery.py

echo "Recovery assessment completed."

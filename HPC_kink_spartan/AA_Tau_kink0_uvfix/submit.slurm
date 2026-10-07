#!/bin/bash
#SBATCH --job-name=AA_Tau_kink0_uvfix
#SBATCH --account=punim3000
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_%x_%j.log

bash /data/gpfs/projects/punim3000/MPIA/HPC_kink_spartan/AA_Tau_kink0_uvfix/run_pipeline.sh

#!/bin/bash
#SBATCH --job-name=AA_Tau_injection
#SBATCH --output=injection_%j.out
#SBATCH --error=injection_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --partition=compute
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@institution.edu

# Load required modules (adjust for your cluster)
module purge
module load python/3.9
module load numpy
module load scipy

# Set up environment
export OMP_NUM_THREADS=1
export PYTHONPATH="/path/to/your/frank:/path/to/your/DSHARP_source_code:$PYTHONPATH"

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Create pause file before starting
touch PAUSE_AFTER_FLUX_BIN

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Run the injection script
echo "Starting CPD injection and frank fitting..."
python AA_Tau_robust2_0_gap0_injectloop.py

echo "Job completed at: $(date)"
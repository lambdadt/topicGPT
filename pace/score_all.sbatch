#!/bin/bash
#SBATCH -J score_all    # Job name
#SBATCH --account=coc             # Account
#SBATCH --cpus-per-task=12        # Use 12 cores
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH -t 8:00:00                # Time limit (8 hours, adjust after testing)
#SBATCH -q coe-ice                # Partition
#SBATCH -o Report_score_all-%j.out          # Output file
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --mail-user=<gt-username>@gatech.edu

echo "Loading modules"
module load anaconda3

# Activate environment, run ollama
echo "Activate conda env"
conda activate topicgpt 

# Set parameters
ROOT_FOLDER=..

# Optional: Verify CPU usage
echo "Running on $(nproc) cores"

# Run python script
echo "Start running script"
cd ${ROOT_FOLDER}
mkdir -p logs
DATESTR=$(date '+%Y%m%dT%H%M%S')

time python compute_scores.py
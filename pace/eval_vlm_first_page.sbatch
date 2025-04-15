#!/bin/bash
#SBATCH -J eval_vlm_first_page    # Job name
#SBATCH --account=coc             # Account
#SBATCH --cpus-per-task=12        # Use 12 cores
#SBATCH --nodes=1 --ntasks-per-node=4 --gres=gpu
#SBATCH --mem=100G
#SBATCH -t 8:00:00                # Time limit (8 hours, adjust after testing)
#SBATCH -q coe-ice                # Partition
#SBATCH -o Report_eval_vlm_first_page-%j.out          # Output file
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --mail-user=<gt-username>@gatech.edu

echo "GPU Node"
nvidia-smi

echo "Loading modules"
module load anaconda3
module load ollama

# Activate environment, run ollama
echo "Start ollama"
ollama serve &
echo "Activate conda env"
conda activate topicgpt 

# Set parameters
ROOT_FOLDER=..

OUTPUT=output/topic_assignments/vlm_firstpage
DOCS_METADATA=data_pdfs/docs_metadata.csv
METHOD=vlm
PAGE_SELECTION="first" # random
NUM_PAGES=1

# Optional: Verify CPU usage
echo "Running on $(nproc) cores"

# Run python script
echo "Start running script"
cd ${ROOT_FOLDER}
mkdir -p logs
DATESTR=$(date '+%Y%m%dT%H%M%S')

time python program.py assign_topics \
    -o "$OUTPUT" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "data/misc/arxiv_topics.yaml" \
    --arxiv_topic_file "output/generation_arxiv_1.md" \
    --method "$METHOD" \
    --page_selection_criterion "$PAGE_SELECTION" \
    --num_pages $NUM_PAGES
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_vlm_first_page-$SLURM_JOB_ID.out"
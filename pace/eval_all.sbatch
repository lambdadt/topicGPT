#!/bin/bash
#SBATCH -J eval_all    # Job name
#SBATCH --account=coc             # Account
#SBATCH --cpus-per-task=12        # Use 12 cores
#SBATCH --nodes=1 --ntasks-per-node=2 --gres=gpu  # or --gres=gpu:h100:1 if nvidia-smi fails
#SBATCH --mem=32G
#SBATCH -t 16:00:00                # Time limit (8 hours, adjust after testing)
#SBATCH -q coe-ice                # Partition
#SBATCH -o Report_eval_all-%j.out          # Output file
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

DATESTR=$(date '+%Y%m%dT%H%M%S')
OUTPUT=output/topic_assignments/${DATESTR}
DOCS_METADATA=data_pdfs/docs_metadata.csv
ARXIV_TOPICS_YAML="data/misc/arxiv_topics.yaml"
ARXIV_TOPIC_FILE="output/generation_arxiv_1.md"

echo "Evaluating using all methods"
echo "Output directory: $OUTPUT"

# Optional: Verify CPU usage
echo "Running on $(nproc) cores"

# Run python script
echo "Start running script"
cd ${ROOT_FOLDER}
mkdir -p logs

echo "BERTopic; first page"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/bertopic_first_page" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "bertopic" \
    --page_selection_criterion first \
    --num_pages 1 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_bertopic_first_page-$SLURM_JOB_ID.out"

echo "BERTopic; random page"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/bertopic_random_page" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "bertopic" \
    --page_selection_criterion random \
    --num_pages 1 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_bertopic_random_page-$SLURM_JOB_ID.out"

echo "BERTopic; 2 random pages"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/bertopic_2_random_pages" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "bertopic" \
    --page_selection_criterion random \
    --num_pages 2 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_bertopic_2_random_pages-$SLURM_JOB_ID.out"

exit

echo "VLM; first page"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/vlm_first_page" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "vlm" \
    --page_selection_criterion first \
    --num_pages 1 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_vlm_first_page-$SLURM_JOB_ID.out"


echo "VLM; random page"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/vlm_random_page" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "vlm" \
    --page_selection_criterion random \
    --num_pages 1 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_vlm_random_page-$SLURM_JOB_ID.out"

echo "VLM; 2 random pages"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/vlm_2_random_pages" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "vlm" \
    --page_selection_criterion random \
    --num_pages 2 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_vlm_2_random_pages-$SLURM_JOB_ID.out"


echo "LLM; first page"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/llm_first_page" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "llm" \
    --page_selection_criterion first \
    --num_pages 1 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_llm_first_page-$SLURM_JOB_ID.out"

echo "LLM; random page"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/llm_random_page" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "llm" \
    --page_selection_criterion random \
    --num_pages 1 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_llm_random_page-$SLURM_JOB_ID.out"

echo "LLM; 2 random pages"
DATESTR=$(date '+%Y%m%dT%H%M%S')
time python program.py assign_topics \
    -o "${OUTPUT}/llm_2_random_pages" \
    --docs_metadata_csv "${DOCS_METADATA}" \
    --arxiv_topics_yaml "${ARXIV_TOPICS_YAML}" \
    --arxiv_topic_file "${ARXIV_TOPIC_FILE}" \
    --method "llm" \
    --page_selection_criterion random \
    --num_pages 2 \
    --verbose \
    2>&1 | sed 's/\r/\n/g' | tee "logs/${DATESTR}_pace_eval_llm_2_random_pages-$SLURM_JOB_ID.out"
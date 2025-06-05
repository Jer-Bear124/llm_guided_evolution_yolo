#!/bin/bash
#SBATCH --job-name=evaluateGene_Sanity_Check
#SBATCH --output=/storage/ice1/0/2/yzhang3942/llm-guided-evolution/slurm_job_outputs/evaluate_gene_jobs/slurm-%j.out
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:2
#SBATCH -G 2
#SBATCH -C "A100-40GB|A100-80GB|H100"
#SBATCH --mem 128G
#SBATCH -c 4
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
module load gcc/13.2.0
module load cuda/12.1.1

# Activate Conda environment
source /opt/apps/Module/anaconda3/2021.11/bin/activate huggingface_env
# conda info
conda activate llm_env
conda info
echo "Current Conda environment: $CONDA_DEFAULT_ENV"
# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
python ./sota/ultralytics/test.py -network "network_xXx0I15ZDbj9VxK5xCq0cmirgGH.yaml"

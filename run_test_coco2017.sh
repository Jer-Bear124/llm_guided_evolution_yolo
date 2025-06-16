#!/bin/bash
#SBATCH --job-name=test_coco2017
#SBATCH -t 8:00:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:2
#SBATCH --mem 16G
#SBATCH -c 1                    # number of CPU cores
echo "launching test_coco2017"
hostname
# module load anaconda3/2020.07 2021.11
module load cuda/12.6.1
module load anaconda3
export CUDA_VISIBLE_DEVICES=0

conda deactivate
conda activate llm_env
export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
conda info

python tests/sota/test_coco2017.py
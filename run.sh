#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00              		# Runtime in D-HH:MM
#SBATCH --mem 128G
#SBATCH -c 4                          # number of CPU cores
#SBATCH -G 2
#SBATCH --gres=gpu:2
#SBATCH -C "A100-40GB|A100-80GB|H100"
echo "launching LLM Guided Evolution YOLO"
hostname
# module load anaconda3/2020.07 2021.11
# module load cuda/12
# module load cuda/11
module load cuda
module load anaconda3
#export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage

#source /opt/apps/Module/anaconda3/2021.11/bin/activate
conda activate llm_env
conda info

which python
echo CHECK DONE

#python run_improved.py first_test
conda run -n llm_env python ./run_improved.py first_test

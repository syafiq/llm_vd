#!/bin/bash
#SBATCH -N 1 --gpus-per-node=<your_GPU> -t <time_limit> -A <your_slurm_project> -o <output_file>

module load <conda_module>
conda activate <conda_env>
accelerate launch --config_file default_config.yaml finetune.py

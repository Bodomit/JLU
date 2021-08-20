#!/usr/bin/env bash
#SBATCH --job-name=JLU-Train
#SBATCH --partition=k2-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/scratch2/users/40057686/logs/jlu-train/%A-%a.log
#SBATCH --time=3-0
#SBATCH --signal=SIGUSR1@90

module add nvidia-cuda
conda activate houshou3

srun python -m train /mnt/scratch2/users/40057686/results/jlu-train -b 128


#!/usr/bin/env bash
#SBATCH --job-name=JLU-TrainPrimary
#SBATCH --partition=k2-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/scratch2/users/40057686/logs/jlu-train-primary/%A-%a.log
#SBATCH --time=3-0
#SBATCH --signal=SIGUSR1@90

# Invoke with sbatch --array=0-4 ./scripts/train_primary.sh $RESULTS_ROOT_DIR $PRETRAINED_PATH

module add nvidia-cuda

RESULTS_ROOT_DIR=$1
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"

PRETRAINED_PATH=$2
echo "PRETRAINED_PATH: $PRETRAINED_PATH"

LR_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "ALPHA_ID: $ALPHA_ID"

LRS=(1e-4 1e-3 1e-5 1e-6 1e-2)
LR=${LRS[$LR_ID]}

RESULTSDIR=$RESULTS_ROOT_DIR/jlu-train-primary/$LR

echo "LR: $LR"
echo "RESULTSDIR: $RESULTSDIR"

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

srun python -m train $RESULTSDIR --primary-only -b 128 --pretrained $PRETRAINED_PATH -lr $LR


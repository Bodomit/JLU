#!/usr/bin/env bash
#SBATCH --job-name=JLU-TestF
#SBATCH --partition=k2-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150GB
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/scratch2/users/40057686/logs/jlu-train/%A-%a.log
#SBATCH --time=3-0

# Invoke with sbatch --array=0-6 ./scripts/train_alpha_verification_test_obfuscation.sh $RESULTS_ROOT_DIR

module add nvidia-cuda

RESULTS_ROOT_DIR=$1
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"

PRETRAINED_PATH=$2
echo "PRETRAINED_PATH: $PRETRAINED_PATH"

ALPHA_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "ALPHA_ID: $ALPHA_ID"

ALPHAS=(0 0.01 0.1 1 10 100 1000)
ALPHA=${ALPHAS[$ALPHA_ID]}

RESULTSDIR=$RESULTS_ROOT_DIR/$ALPHA

echo "ALPHA: $ALPHA"
echo "RESULTSDIR: $RESULTSDIR"

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

srun python -m obfuscation_train_test $RESULTSDIR -b 128


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

# Invoke with sbatch --array=0-8 ./scripts/train_alpha_verification.sh $RESULTS_ROOT_DIR $PRETRAINED_PATH

module add nvidia-cuda

RESULTS_ROOT_DIR=$1
echo "RESULTS_ROOT_DIR: $RESULTS_ROOT_DIR"

PRETRAINED_PATH=$2
echo "PRETRAINED_PATH: $PRETRAINED_PATH"

ALPHA_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "ALPHA_ID: $ALPHA_ID"

ALPHAS=(0 0.0001 0.001 0.01 0.1 1 10 100 1000)
ALPHA=${ALPHAS[$ALPHA_ID]}

RESULTSDIR=$RESULTS_ROOT_DIR/jlu-train/verification_alphas/$ALPHA

echo "ALPHA: $ALPHA"
echo "RESULTSDIR: $RESULTSDIR"

echo "CPU Stats"
python -c "import os; print('CPUS: ', len(os.sched_getaffinity(0)))"
echo ""

echo "GPU Stats:"
nvidia-smi
echo ""

srun python -m features_train $RESULTSDIR -b 128 --alpha $ALPHA -d vggface2_maadface -p id


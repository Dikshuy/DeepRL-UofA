#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=4G
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

mkdir -p results/jumping_task

CONFIG=$1
LR=$2
TARGET_UPDATE_INTERVAL=$3
MINIBATCH_SIZE=$4
N_STEP=$5
GRADIENT_UPDATE_FREQUENCY=$6
SEED=$7

echo "Running Jumping Task Experiment..."
echo "Configuration: Config=$CONFIG, LR=$LR, Target Update Interval=$TARGET_UPDATE_INTERVAL, Minibatch Size=$MINIBATCH_SIZE, N-Step=$N_STEP, Gradient Update Frequency=$GRADIENT_UPDATE_FREQUENCY, Seed=$SEED"

python3 jumping_task_script.py \
    --config "$CONFIG" \
    --lr "$LR" \
    --target-update-interval "$TARGET_UPDATE_INTERVAL" \
    --minibatch-size "$MINIBATCH_SIZE" \
    --n-step "$N_STEP" \
    --gradient-update-frequency "$GRADIENT_UPDATE_FREQUENCY" \
    --run-label "$SEED" \
    --num-training-episodes 5000 \
    --n-seeds 3

echo "Jumping Task Experiment Completed"
echo "========================================================================================="
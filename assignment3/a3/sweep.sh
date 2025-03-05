#!/bin/bash

mkdir -p data

CONFIGS=(1 2 3)
LRS=(0.0001 0.0003 0.001)
TARGET_UPDATE_INTERVALS=(50 100 200)
MINIBATCH_SIZES=(64 128 256)
N_STEPS=(3 5 7)
GRADIENT_UPDATE_FREQUENCIES=(1 2 4)

LOG_FILE="hyperparameter_sweep.log"
echo "Starting Hyperparameter Sweep" > "$LOG_FILE"

for config in "${CONFIGS[@]}"; do
    for lr in "${LRS[@]}"; do
        for target_update_interval in "${TARGET_UPDATE_INTERVALS[@]}"; do
            for minibatch_size in "${MINIBATCH_SIZES[@]}"; do
                for n_step in "${N_STEPS[@]}"; do
                    for gradient_update_frequency in "${GRADIENT_UPDATE_FREQUENCIES[@]}"; do
                        exp_id="${config}_lr${lr}_tui${target_update_interval}_batch${minibatch_size}_nstep${n_step}_gradu${gradient_update_frequency}"
                        
                        echo "Running experiment: $exp_id" | tee -a "$LOG_FILE"
                        
                        python jumping_task_script.py \
                            --config "$config" \
                            --lr "$lr" \
                            --target-update-interval "$target_update_interval" \
                            --minibatch-size "$minibatch_size" \
                            --n-step "$n_step" \
                            --gradient-update-frequency "$gradient_update_frequency" \
                            --num-training-episodes 5000 \
                            --n-seeds 3 \
                            2>&1 | tee -a "$LOG_FILE"
                        
                        echo "Experiment $exp_id completed" | tee -a "$LOG_FILE"
                    done
                done
            done
        done
    done
done

echo "Hyperparameter Sweep Completed" | tee -a "$LOG_FILE"
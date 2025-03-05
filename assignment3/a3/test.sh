#!/bin/bash

SCRIPT_DIR="./generated_scripts"
mkdir -p $SCRIPT_DIR

CONFIGS=(1)
LRS=(0.0001)
TARGET_UPDATE_INTERVALS=(1000)
MINIBATCH_SIZES=(256)
N_STEPS=(14)
GRADIENT_UPDATE_FREQUENCIES=(4)
SEEDS=(5)

BASE_SCRIPT="run_sweep.sh"

for config in "${CONFIGS[@]}"; do
    for lr in "${LRS[@]}"; do
        for target_update_interval in "${TARGET_UPDATE_INTERVALS[@]}"; do
            for minibatch_size in "${MINIBATCH_SIZES[@]}"; do
                for n_step in "${N_STEPS[@]}"; do
                    for gradient_update_frequency in "${GRADIENT_UPDATE_FREQUENCIES[@]}"; do
                        for seed in "${SEEDS[@]}"; do
                            script_name="$SCRIPT_DIR/jumping_task_config${config}_lr${lr}_tui${target_update_interval}_batch${minibatch_size}_nstep${n_step}_gradu${gradient_update_frequency}_seed${seed}.sh"
                            
                            sed -e "s/CONFIG=.*/CONFIG=$config/" \
                                -e "s/LR=.*/LR=$lr/" \
                                -e "s/TARGET_UPDATE_INTERVAL=.*/TARGET_UPDATE_INTERVAL=$target_update_interval/" \
                                -e "s/MINIBATCH_SIZE=.*/MINIBATCH_SIZE=$minibatch_size/" \
                                -e "s/N_STEP=.*/N_STEP=$n_step/" \
                                -e "s/GRADIENT_UPDATE_FREQUENCY=.*/GRADIENT_UPDATE_FREQUENCY=$gradient_update_frequency/" \
                                -e "s/SEED=.*/SEED=$seed/" \
                                $BASE_SCRIPT > "$script_name"
                            
                            chmod +x "$script_name"
                            
                            sbatch "$script_name"
                            
                            echo "Generated and submitted: $script_name"
                        done
                    done
                done
            done
        done
    done
done

echo "All scripts created and submitted!"
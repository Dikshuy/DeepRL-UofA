#!/bin/bash

SCRIPT_DIR="./generated_scripts"
mkdir -p $SCRIPT_DIR

CONFIGS=(1 2 3)
LRS=(0.0001 0.0003)
TARGET_UPDATE_INTERVALS=(100 200 500 1000)
MINIBATCH_SIZES=(64 128 256)
N_STEPS=(3 5 10 14)
GRADIENT_UPDATE_FREQUENCIES=(1 2 4)
SEEDS=(1 2 3 4 5)

for config in "${CONFIGS[@]}"; do
    for lr in "${LRS[@]}"; do
        for target_update_interval in "${TARGET_UPDATE_INTERVALS[@]}"; do
            for minibatch_size in "${MINIBATCH_SIZES[@]}"; do
                for n_step in "${N_STEPS[@]}"; do
                    for gradient_update_frequency in "${GRADIENT_UPDATE_FREQUENCIES[@]}"; do
                        for seed in "${SEEDS[@]}"; do\
                            script_name="$SCRIPT_DIR/jumping_task_config${config}_lr${lr}_tui${target_update_interval}_batch${minibatch_size}_nstep${n_step}_gradu${gradient_update_frequency}_seed${seed}.sh"
                            
                            cat > "$script_name" << EOL
#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=4G
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

mkdir -p results/jumping_task

echo "Running Jumping Task Experiment..."
echo "Configuration: Config=$config, LR=$lr, Target Update Interval=$target_update_interval, Minibatch Size=$minibatch_size, N-Step=$n_step, Gradient Update Frequency=$gradient_update_frequency, Seed=$seed"

python3 jumping_task_script.py \
    --config $config \
    --lr $lr \
    --target-update-interval $target_update_interval \
    --minibatch-size $minibatch_size \
    --n-step $n_step \
    --gradient-update-frequency $gradient_update_frequency \
    --run-label $seed \
    --num-training-episodes 5000 \
    --n-seeds 1

echo "Jumping Task Experiment Completed"
echo "========================================================================================="
EOL
                            
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
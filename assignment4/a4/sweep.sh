#!/bin/bash

SCRIPT_DIR="./generated_scripts"
mkdir -p $SCRIPT_DIR
OUTPUT_DIR="results"
mkdir -p $OUTPUT_DIR

ENVS=("Ant-v4" "Walker2d-v4")
NUM_SEEDS=5
TOTAL_STEPS=1000000

# Format: "CONFIG_NAME USE_TWIN_CRITICS POLICY_FREQ POLICY_NOISE TAU EXPLORER_TYPE EXPLORER_NOISE"
CONFIGS=(
    "TD3_(Default) 1 2 0.2 0.005 gaussian 0.1"
    "TD3_(Single_Critic) 0 2 0.2 0.005 gaussian 0.1"
    "TD3_(No_Delayed_Updates) 1 1 0.2 0.005 gaussian 0.1"
    "TD3_(More_Delayed_Updates) 1 4 0.2 0.005 gaussian 0.1"
    "TD3_(No_policy_noise) 1 2 0.0 0.005 gaussian 0.1"
    "TD3_(More_policy_noise) 1 2 0.5 0.005 gaussian 0.1"
    "TD3_(Fast_Target_Update) 1 2 0.2 0.05 gaussian 0.1"
    "TD3_(Slow_Target_Update) 1 2 0.2 0.001 gaussian 0.1"
    "TD3_(Ornstein_Uhlenbeck) 1 2 0.2 0.005 ou 0.1"
)

for ENV in "${ENVS[@]}"; do
    mkdir -p "$OUTPUT_DIR/$ENV"
    
    for CONFIG in "${CONFIGS[@]}"; do
        read -r CONFIG_NAME USE_TWIN_CRITICS POLICY_FREQ POLICY_NOISE TAU EXPLORER_TYPE EXPLORER_NOISE<<< "$CONFIG"
        
        for SEED in $(seq 0 $((NUM_SEEDS-1))); do
            SANITIZED_CONFIG_NAME=$(echo "$CONFIG_NAME" | tr " " "_" | tr -d "()")
            
            script_name="$SCRIPT_DIR/${ENV}_${SANITIZED_CONFIG_NAME}_seed${SEED}.sh"
            
            cat > "$script_name" << EOL
#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=8G
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo "Running TD3 Ablation Experiment..."
echo "Environment: $ENV"
echo "Configuration: $CONFIG_NAME"
echo "Seed: $SEED"

mkdir -p "$OUTPUT_DIR/$ENV"

python td3_ablations.py \\
    --env "$ENV" \\
    --config-name "$CONFIG_NAME" \\
    --seed $SEED \\
    --total-steps $TOTAL_STEPS \\
    --output-dir "$OUTPUT_DIR/$ENV" \\
    --use-twin-critics $USE_TWIN_CRITICS \\
    --policy-freq $POLICY_FREQ \\
    --policy-noise $POLICY_NOISE \\
    --tau $TAU \\
    --explorer-type $EXPLORER_TYPE \\
    --explorer-noise $EXPLORER_NOISE

echo "TD3 Ablation Experiment Completed"
echo "========================================================================================="
EOL
            
            chmod +x "$script_name"
            sbatch "$script_name"
            
            echo "Generated and submitted: $script_name"
        done
    done
done

echo "All scripts created and submitted!"
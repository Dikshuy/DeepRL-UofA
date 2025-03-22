#!/bin/bash

SCRIPT_DIR="./generated_scripts"
mkdir -p $SCRIPT_DIR
OUTPUT_DIR="results"
mkdir -p $OUTPUT_DIR

ENVS=("Ant-v4" "Walker2d-v4")

for ENV in "${ENVS[@]}"; do
    plot_script_name="$SCRIPT_DIR/${ENV}_generate_plots.sh"
    cat > "$plot_script_name" << EOL
#!/bin/bash
#SBATCH --account=def-mtaylor3_cpu
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Generating plots for $ENV..."

python td3_ablation.py \\
    --env "$ENV" \\
    --config-name "dummy" \\
    --seed 0 \\
    --output-dir "$OUTPUT_DIR/$ENV" \\
    --plot-only

echo "Plot generation completed for $ENV"
echo "========================================================================================="
EOL
    
    chmod +x "$plot_script_name"
    
    echo "Generated plot script: $plot_script_name"
done

echo "All scripts created and submitted!"
#!/bin/bash

OUTPUT_DIR="results"
ENVS=("Ant-v4" "Walker2d-v4")

echo "Starting plot generation for all environments..."

for ENV in "${ENVS[@]}"; do
    echo "Generating plots for environment: $ENV"
    
    python td3_ablations.py \
        --env "$ENV" \
        --config-name "dummy" \
        --seed 0 \
        --output-dir "$OUTPUT_DIR/$ENV" \
        --plot-only
    
    if [ $? -eq 0 ]; then
        echo "Plot generation completed successfully for $ENV"
    else
        echo "Error: Plot generation failed for $ENV"
    fi
    
    echo "-------------------------------------"
done

echo "All plots generated!"
echo "Plot images are saved in the respective environment directories under $OUTPUT_DIR"
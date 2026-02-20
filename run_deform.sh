#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinysam

# Base directory containing all scenes
BASE_DIR="/Users/cedric/Documents/example_data"

# Loop over all directories inside BASE_DIR
for SCENE_DIR in "$BASE_DIR"/*/; do
    SCENE=$(basename "$SCENE_DIR")
    echo "========================================"
    echo "Processing scene: $SCENE"
    
    PLY_PATH="$SCENE_DIR/point_cloud.ply"
    MODEL_PATH="$SCENE_DIR/deform.pth"
    OUTPUT_PATH="$SCENE_DIR/mask.bin"
    
    # Check if necessary files exist before running
    if [[ ! -f "$PLY_PATH" ]] || [[ ! -f "$MODEL_PATH" ]]; then
        echo "Skipping $SCENE - point_cloud.ply or deform.pth not found."
        continue
    fi

    # Build and run the command
    CMD="python generate_deformation_mask.py \"$PLY_PATH\" \"$MODEL_PATH\" \"$OUTPUT_PATH\" --smooth --threshold 0.1 --max-t 1.0 --num-samples 10"
    echo "Running: $CMD"
    eval $CMD
done

echo "========================================"
echo "Finished processing all scenes."

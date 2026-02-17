#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinysam

# Configuration - modify these paths as needed
PLY_PATH="/Users/cedric/Documents/example_data/sear_steak/point_cloud.ply"
MODEL_PATH="/Users/cedric/Documents/example_data/sear_steak/deform.pth"
OUTPUT_PATH="/Users/cedric/Documents/example_data/sear_steak/mask.bin"

# Optional: Cluster-based masking
# If CLUSTERS_PATH is set, uses cluster-based mode (all splats in a moving cluster are marked dynamic)
CLUSTERS_PATH="/Users/cedric/Documents/example_data/sear_steak/clusters.bin"

# Build the command
CMD="python generate_deformation_mask.py $PLY_PATH $MODEL_PATH $OUTPUT_PATH --smooth --threshold 0.1 --max-t 1.0 --num-samples 10 "

# Add cluster argument if clusters file exists
if [ -n "$CLUSTERS_PATH" ] && [ -f "$CLUSTERS_PATH" ]; then
    echo "Using cluster-based masking: $CLUSTERS_PATH"
    CMD="$CMD --clusters $CLUSTERS_PATH"
else
    if [ -n "$CLUSTERS_PATH" ]; then
        echo "Warning: clusters file not found: $CLUSTERS_PATH"
    fi
    echo "Using per-splat masking"
fi

echo "Running: $CMD"
eval $CMD

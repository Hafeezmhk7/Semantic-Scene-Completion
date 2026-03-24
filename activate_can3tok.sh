#!/bin/bash
# Activation script for can3tok environment
# Usage: source activate_can3tok.sh

# Set Python path
export PATH="/home/yli11/.conda/envs/can3tok/bin:$PATH"

# Set PyTorch library paths
export LD_LIBRARY_PATH="/home/yli11/.conda/envs/can3tok/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/yli11/.conda/envs/can3tok/lib:$LD_LIBRARY_PATH"

# Add simple-knn to Python path
export PYTHONPATH="/home/yli11/scratch/Hafeez_thesis/Can3Tok/submodules/simple-knn:$PYTHONPATH"

echo "✓ can3tok environment activated"
echo "  Python: $(python --version 2>&1 | cut -d' ' -f2)"
echo "  Working dir: /home/yli11/scratch/Hafeez_thesis/Can3Tok"

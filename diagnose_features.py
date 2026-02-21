#!/usr/bin/env python3
"""
Diagnosis Script - Find where 18 features become 62

This script traces the data flow to understand where feature expansion happens.
"""

import torch
import numpy as np
import sys
import os

# Add Can3Tok to path
sys.path.insert(0, "/gpfs/work3/0/prjs1291/Hafeez_thesis/Can3Tok")
os.chdir("/gpfs/work3/0/prjs1291/Hafeez_thesis/Can3Tok")

print("="*60)
print("Feature Dimension Diagnosis")
print("="*60)
print()

# ============================================
# Test 1: Load Dataset
# ============================================
print("Test 1: Loading dataset...")

# Use the corrected SceneSplat loader
from gs_dataset_scenesplat import gs_dataset

data_path = "/gpfs/work3/0/prjs1291/datasets/gaussian_world/preprocessed/interior_gs/train"
if not os.path.exists(data_path):
    data_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs/train"

dataset = gs_dataset(
    root=data_path,
    resol=200,
    random_permute=False,
    train=True,
    sampling_method='hybrid'
)

print(f"✓ Dataset loaded: {len(dataset)} scenes")

# Load one sample
gs_params, idx = dataset[0]
print(f"  Dataset output shape: {gs_params.shape}")
print(f"  Expected: (40000, 18)")
print()

# ============================================
# Test 2: Check Perceiver Model Input
# ============================================
print("Test 2: Checking Perceiver model...")

try:
    from model.michelangelo.utils import instantiate_from_config
    from model.michelangelo.utils.misc import get_config_from_file
    
    config_path = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
    model_config = get_config_from_file(config_path)
    
    if hasattr(model_config, "model"):
        model_config = model_config.model
    
    perceiver = instantiate_from_config(model_config)
    perceiver.eval()
    
    print(f"✓ Model loaded")
    
    # Check input projection layer
    print()
    print("Checking model architecture...")
    
    # Try to find the input projection layer
    if hasattr(perceiver, 'shape_model'):
        if hasattr(perceiver.shape_model, 'encoder'):
            encoder = perceiver.shape_model.encoder
            if hasattr(encoder, 'input_proj'):
                input_proj = encoder.input_proj
                print(f"  Found input_proj layer:")
                print(f"    Type: {type(input_proj)}")
                print(f"    Weight shape: {input_proj.weight.shape}")
                print(f"    Expected input: {input_proj.in_features}")
                print(f"    Output: {input_proj.out_features}")
                print()
                print(f"  → Model expects {input_proj.in_features} input features!")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================
# Test 3: Simulate Training Data Flow
# ============================================
print("Test 3: Simulating training data flow...")

# Create a batch like the training code does
batch_size = 2
batch = torch.from_numpy(np.stack([dataset[i][0] for i in range(batch_size)]))
print(f"  Batch shape: {batch.shape}")
print(f"  Expected: ({batch_size}, 40000, 18)")
print()

# Simulate what training code does
print("  Simulating training transformations...")

# The training code does:
# 1. Type conversion
batch = batch.type(torch.float32)
print(f"    After type conversion: {batch.shape}")

# 2. Random rotation (modifies xyz at positions 4:7)
# Let's skip this for diagnosis

# 3. Voxel PE computation (overwrites positions 0:3)
# Let's skip this for diagnosis

# So after training preprocessing, still 18 features!
print(f"    After training preprocessing: {batch.shape}")
print()

# ============================================
# Test 4: Try Forward Pass
# ============================================
print("Test 4: Attempting forward pass...")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceiver = perceiver.to(device)
    batch = batch.to(device)
    
    print(f"  Input shape: {batch.shape}")
    print(f"  Input features: {batch.shape[-1]}")
    
    # Try the model call as in training code
    with torch.no_grad():
        try:
            # The training code calls:
            # shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(
            #     UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:,:,:3]
            # )
            
            positions = batch[:, :, 0:3]  # For attention
            print(f"  Positions shape: {positions.shape}")
            
            output = perceiver(
                batch,      # Full features
                batch,      # Duplicate
                batch,      # Duplicate
                positions   # PE for attention
            )
            
            print(f"  ✓ Forward pass successful!")
            print(f"    Output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"    Output elements: {len(output)}")
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        print(f"      [{i}] shape: {o.shape}")
            
        except RuntimeError as e:
            if "mat1 and mat2" in str(e):
                print(f"  ✗ Shape mismatch error:")
                print(f"    {e}")
                print()
                print("  Analysis:")
                
                # Parse the error
                error_str = str(e)
                if "x" in error_str:
                    parts = error_str.split("(")[1].split(")")[0].split(" and ")
                    if len(parts) == 2:
                        input_shape = parts[0]
                        weight_shape = parts[1]
                        print(f"    Input shape: {input_shape}")
                        print(f"    Weight shape: {weight_shape}")
                        
                        # Extract dimensions
                        if "x" in input_shape:
                            dims = input_shape.split("x")
                            if len(dims) == 2:
                                total_points = int(dims[0])
                                input_features = int(dims[1])
                                
                                batch_calc = total_points // 40000
                                
                                print()
                                print(f"    Batch size: {batch_calc}")
                                print(f"    Points per scene: {total_points // batch_calc:,}")
                                print(f"    Input features: {input_features}")
                                print(f"    Expected features: {int(weight_shape.split('x')[0])}")
                                print(f"    Missing features: {int(weight_shape.split('x')[0]) - input_features}")
            else:
                print(f"  ✗ Other error: {e}")
                raise
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)
print("Diagnosis Complete")
print("="*60)
print()
print("Summary:")
print("1. Dataset returns 18 features ✓")
print("2. Model expects ? features")
print("3. Need to find where expansion happens")
print()
print("Next steps:")
print("- Check if Perceiver has built-in feature expansion")
print("- OR add feature expansion in dataset loader")
print("- OR add feature expansion in training code")
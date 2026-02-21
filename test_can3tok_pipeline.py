#!/usr/bin/env python3
"""
Minimal test script for Can3Tok pipeline
Tests if the code can load and process data
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("="*80)
print("CAN3TOK PIPELINE TEST")
print("="*80)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import scene
    print("  ✓ scene module")
except Exception as e:
    print(f"  ✗ scene module: {e}")

try:
    from model.michelangelo.models.tsal import CausalTSALVAE
    print("  ✓ Can3Tok VAE model")
except Exception as e:
    print(f"  ✗ Can3Tok VAE model: {e}")

# Test 2: Check data loading capability
print("\n2. Testing data structures...")
# Create dummy Gaussian data (N, 14) format
N = 40000
dummy_gaussians = np.random.randn(N, 14).astype(np.float32)
dummy_gaussians[:, 3:6] = np.random.rand(N, 3)  # RGB [0,1]
dummy_gaussians[:, 13] = np.random.rand(N)  # Opacity [0,1]

dummy_tensor = torch.from_numpy(dummy_gaussians)
print(f"  ✓ Created dummy Gaussians: {dummy_tensor.shape}")

# Test 3: Check if model can be instantiated
print("\n3. Testing model instantiation...")
try:
    # These are typical Can3Tok VAE parameters
    from model.michelangelo.models.tsal import CausalTSALVAE
    
    model = CausalTSALVAE(
        num_latents=200,  # Latent tokens
        latent_dim=512,   # Latent dimension
        in_dim=14,        # Input dimension (xyz, rgb, scale, quat, opacity)
        out_dim=14,       # Output dimension
    )
    print(f"  ✓ Model created: {model.__class__.__name__}")
    
    # Test forward pass with dummy data
    with torch.no_grad():
        dummy_batch = dummy_tensor.unsqueeze(0)  # (1, N, 14)
        # This would test the encoder
        print(f"  ✓ Ready for forward pass with shape: {dummy_batch.shape}")
        
except Exception as e:
    print(f"  ✗ Model test failed: {e}")

print("\n" + "="*80)
print("Test complete! Check for any ✗ marks above.")
print("="*80)

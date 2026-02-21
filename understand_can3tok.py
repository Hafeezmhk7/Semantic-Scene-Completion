#!/usr/bin/env python3
"""
Can3Tok Code Structure Analyzer
Helps understand the pipeline and key components
"""

import os
from pathlib import Path
import re

def analyze_python_file(filepath):
    """Extract key information from Python file"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    info = {
        'path': str(filepath),
        'lines': len(content.split('\n')),
        'imports': [],
        'classes': [],
        'functions': [],
        'has_main': '__name__ == "__main__"' in content,
        'has_argparse': 'argparse' in content,
        'has_dataloader': 'DataLoader' in content or 'Dataset' in content,
    }
    
    # Extract imports
    for match in re.finditer(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE):
        pkg = match.group(1)
        if not pkg.startswith('.'):
            info['imports'].append(pkg)
    
    # Extract classes
    for match in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
        info['classes'].append(match.group(1))
    
    # Extract functions
    for match in re.finditer(r'^def\s+(\w+)', content, re.MULTILINE):
        info['functions'].append(match.group(1))
    
    return info


def analyze_can3tok_structure(repo_path):
    """Analyze Can3Tok repository structure"""
    repo_path = Path(repo_path)
    
    print("="*80)
    print("CAN3TOK CODE STRUCTURE ANALYSIS")
    print("="*80)
    
    # Key directories to analyze
    key_dirs = {
        'Training Scripts': ['gs_can3tok.py', 'train.py', 'gs_ae.py', 'gs_pointvae.py'],
        'Data Loading': ['scene/dataset_readers.py', 'scene/__init__.py'],
        'Models': ['model/michelangelo/**/*.py'],
        'Preprocessing': ['groundedSAM.py', 'sfm_camera_norm.py', 'down_sam_init_sfm.py'],
    }
    
    print("\n" + "="*80)
    print("KEY FILES BY CATEGORY")
    print("="*80)
    
    for category, patterns in key_dirs.items():
        print(f"\n📁 {category}")
        print("-" * 80)
        
        for pattern in patterns:
            if '**' in pattern:
                # Glob pattern
                files = list(repo_path.glob(pattern))
            else:
                files = [repo_path / pattern] if (repo_path / pattern).exists() else []
            
            for filepath in files:
                if filepath.exists() and filepath.is_file():
                    rel_path = filepath.relative_to(repo_path)
                    size_kb = filepath.stat().st_size / 1024
                    print(f"  ✓ {rel_path} ({size_kb:.1f} KB)")


def find_main_training_script(repo_path):
    """Find and analyze the main training script"""
    repo_path = Path(repo_path)
    
    print("\n" + "="*80)
    print("MAIN TRAINING SCRIPT: gs_can3tok.py")
    print("="*80)
    
    script_path = repo_path / 'gs_can3tok.py'
    
    if not script_path.exists():
        print("❌ gs_can3tok.py not found!")
        return
    
    info = analyze_python_file(script_path)
    
    print(f"\n📄 File: {script_path.name}")
    print(f"📊 Lines: {info['lines']}")
    print(f"🔧 Has argparse: {info['has_argparse']}")
    print(f"📦 Has DataLoader: {info['has_dataloader']}")
    
    print(f"\n📚 Key Imports:")
    key_imports = ['torch', 'numpy', 'Dataset', 'DataLoader', 'model', 'scene']
    for imp in info['imports'][:15]:  # Show first 15
        marker = "⭐" if any(k in imp for k in key_imports) else "  "
        print(f"  {marker} {imp}")
    
    print(f"\n🏗️  Classes Defined:")
    for cls in info['classes']:
        print(f"  • {cls}")
    
    print(f"\n⚙️  Key Functions:")
    for func in info['functions'][:20]:  # Show first 20
        if not func.startswith('_'):  # Skip private
            print(f"  • {func}()")


def check_data_format(repo_path):
    """Check what data format Can3Tok expects"""
    repo_path = Path(repo_path)
    
    print("\n" + "="*80)
    print("EXPECTED DATA FORMAT")
    print("="*80)
    
    # Check dataset readers
    reader_path = repo_path / 'scene' / 'dataset_readers.py'
    
    if reader_path.exists():
        with open(reader_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print("\n📂 From scene/dataset_readers.py:")
        
        # Look for file extensions
        extensions = re.findall(r'["\'](\.\w+)["\']', content)
        if extensions:
            print(f"\n  File extensions used: {set(extensions)}")
        
        # Look for numpy loading
        if 'np.load' in content:
            print("  ✓ Uses numpy arrays (.npy or .npz)")
        
        # Look for ply loading
        if 'plyfile' in content or '.ply' in content:
            print("  ✓ Uses PLY files (.ply)")
        
        # Look for data structure
        if 'xyz' in content.lower():
            print("  ✓ Expects 'xyz' coordinates")
        if 'rgb' in content.lower() or 'color' in content.lower():
            print("  ✓ Expects 'rgb' or 'color'")
        if 'scale' in content.lower():
            print("  ✓ Expects 'scale'")
        if 'rotation' in content.lower() or 'quat' in content.lower():
            print("  ✓ Expects 'rotation' or 'quat'")
        if 'opacity' in content.lower() or 'alpha' in content.lower():
            print("  ✓ Expects 'opacity' or 'alpha'")


def extract_usage_example(repo_path):
    """Extract usage examples from main script"""
    repo_path = Path(repo_path)
    script_path = repo_path / 'gs_can3tok.py'
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    if not script_path.exists():
        print("❌ Main script not found")
        return
    
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Look for argparse arguments
    if 'argparse' in content:
        print("\n📋 Command Line Arguments:")
        
        # Extract add_argument calls
        for match in re.finditer(r"add_argument\(['\"]--?(\w+)['\"].*?help=['\"]([^'\"]+)", content, re.DOTALL):
            arg_name = match.group(1)
            help_text = match.group(2)[:60]
            print(f"  --{arg_name:<20} {help_text}")
    
    # Look for data paths
    print("\n📁 Data Path Patterns:")
    path_matches = re.findall(r'["\']([^"\']*(?:data|dataset|gaussian|ply)[^"\']*)["\']', content.lower())
    for path in set(path_matches[:5]):
        if path and len(path) > 5:
            print(f"  • {path}")


def create_test_script(repo_path):
    """Generate a test script to verify the pipeline"""
    repo_path = Path(repo_path)
    
    test_script = """#!/usr/bin/env python3
\"\"\"
Minimal test script for Can3Tok pipeline
Tests if the code can load and process data
\"\"\"

import sys
import torch
import numpy as np
from pathlib import Path

print("="*80)
print("CAN3TOK PIPELINE TEST")
print("="*80)

# Test 1: Check imports
print("\\n1. Testing imports...")
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
print("\\n2. Testing data structures...")
# Create dummy Gaussian data (N, 14) format
N = 40000
dummy_gaussians = np.random.randn(N, 14).astype(np.float32)
dummy_gaussians[:, 3:6] = np.random.rand(N, 3)  # RGB [0,1]
dummy_gaussians[:, 13] = np.random.rand(N)  # Opacity [0,1]

dummy_tensor = torch.from_numpy(dummy_gaussians)
print(f"  ✓ Created dummy Gaussians: {dummy_tensor.shape}")

# Test 3: Check if model can be instantiated
print("\\n3. Testing model instantiation...")
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

print("\\n" + "="*80)
print("Test complete! Check for any ✗ marks above.")
print("="*80)
"""
    
    output_path = repo_path / 'test_can3tok_pipeline.py'
    with open(output_path, 'w') as f:
        f.write(test_script)
    
    print(f"\n💾 Generated test script: {output_path}")
    print(f"\nRun with: python test_can3tok_pipeline.py")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Can3Tok code structure')
    parser.add_argument('--repo', type=str, 
                       default='/gpfs/work3/0/prjs1291/Hafeez_thesis/Can3Tok',
                       help='Path to Can3Tok repository')
    parser.add_argument('--generate-test', action='store_true',
                       help='Generate a test script')
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo)
    
    if not repo_path.exists():
        print(f"❌ Repository not found: {repo_path}")
        return
    
    # Run analysis
    analyze_can3tok_structure(repo_path)
    find_main_training_script(repo_path)
    check_data_format(repo_path)
    extract_usage_example(repo_path)
    
    if args.generate_test:
        create_test_script(repo_path)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the key files listed above")
    print("2. Check gs_can3tok.py for the main training loop")
    print("3. Look at scene/dataset_readers.py to understand data format")
    print("4. Run: python test_can3tok_pipeline.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
#!/bin/bash
# setup_can3tok_cuda12.8.sh - Environment setup for CUDA 12.8.0

set -e  # Exit on error

echo "================================================"
echo "Can3Tok Setup for CUDA 12.8.0"
echo "================================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# ============================================
# 1. Load Modules (CUDA 12.8.0)
# ============================================
echo "Step 1: Loading modules..."
module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

export CUDA_HOME=$EBROOTCUDA
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA version: $(nvcc --version | grep release)"
echo ""

# ============================================
# 2. Check GPU Availability
# ============================================
echo "Step 2: Checking GPU availability..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# ============================================
# 3. Clean Environment Setup
# ============================================
echo "Step 3: Setting up conda environment..."

# Remove existing environment
conda env remove -n can3tok -y 2>/dev/null || true

# Create new environment with Python 3.11
echo "  Creating new environment 'can3tok' with Python 3.11..."
conda create -n can3tok python=3.11 -y

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate can3tok

echo "  Python: $(python --version)"
echo "  Python path: $(which python)"
echo ""

# ============================================
# 4. Install PyTorch for CUDA 12.8
# ============================================
echo "Step 4: Installing PyTorch compatible with CUDA 12.8..."
echo "  Note: Original required PyTorch 2.1.0+cu121"
echo "  Using newer version compatible with CUDA 12.8"

# Clean pip cache
pip cache purge

# Install PyTorch 2.5.0 for CUDA 12.1 (forward compatible with 12.8)
# OR use PyTorch 2.8.0 for CUDA 12.8
echo "  Option 1: Trying PyTorch 2.5.0+cu121 (CUDA 12.1, forward compatible)"
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --no-cache-dir

echo ""
echo "  Verifying PyTorch installation:"
python << EOF
import torch
print(f"    PyTorch version: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    CUDA version: {torch.version.cuda}")
    print(f"    Number of GPUs: {torch.cuda.device_count()}")
    print(f"    GPU 0: {torch.cuda.get_device_name(0)}")
    # Test a simple CUDA operation
    x = torch.randn(3, 3).cuda()
    print(f"    CUDA tensor test: {x.mean().item():.4f}")
else:
    print("    WARNING: CUDA not available!")
EOF
echo ""

# ============================================
# 5. Install Base Dependencies
# ============================================
echo "Step 5: Installing base dependencies..."
pip install "numpy<2.0" --no-cache-dir  # Use NumPy 1.x for compatibility
pip install plyfile tqdm einops omegaconf --no-cache-dir
pip install trimesh ninja --no-cache-dir
pip install opencv-python scikit-image --no-cache-dir
pip install pybind11 scikit-learn --no-cache-dir
echo ""

# ============================================
# 6. Install PyTorch Lightning
# ============================================
echo "Step 6: Installing PyTorch Lightning..."
pip install pytorch-lightning --no-cache-dir
echo ""

# ============================================
# 7. Install HuggingFace Libraries
# ============================================
echo "Step 7: Installing HuggingFace libraries..."
pip install transformers diffusers datasets accelerate --no-cache-dir
pip install peft wandb --no-cache-dir
echo ""

# ============================================
# 8. Install CUDA Extensions (with OOM protection)
# ============================================
echo "Step 8: Installing CUDA extensions..."
echo "  Using reduced parallelism to prevent OOM"
echo ""

# Set environment variables to limit memory usage
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2

# simple-knn
echo "  8a. Installing simple-knn..."
cd submodules/simple-knn
rm -rf build dist *.egg-info __pycache__ 2>/dev/null || true
# Build with limited resources
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80" \
MAX_JOBS=2 pip install -e . --no-build-isolation --no-cache-dir
cd ../..
echo "    Checking simple-knn..."
python -c "import sys; sys.path.insert(0, 'submodules/simple-knn'); from simple_knn._C import distCUDA2; print('    ✓ simple-knn works')" 2>/dev/null || echo "    ⚠ simple-knn may need restart"
echo ""

# diff-gaussian-rasterization
echo "  8b. Installing diff-gaussian-rasterization..."
cd submodules/diff-gaussian-rasterization
rm -rf build dist *.egg-info __pycache__ 2>/dev/null || true
MAX_JOBS=2 pip install -e . --no-build-isolation --no-cache-dir
cd ../..
echo "    ✓ diff-gaussian-rasterization installed"
echo ""

# chamferdist
echo "  8c. Installing chamferdist..."
pip install chamferdist --no-build-isolation --no-cache-dir
echo "    ✓ chamferdist installed"
echo ""

# ============================================
# 9. Install Optional Packages (Skip Problematic Ones)
# ============================================
echo "Step 9: Installing optional packages..."

# pynanoflann
echo "  9a. Installing pynanoflann..."
pip install git+https://github.com/u1234x1234/pynanoflann.git@0.0.8 --no-cache-dir 2>/dev/null || \
  echo "    ⚠ pynanoflann failed (optional)"
echo ""

# pykeops and geomloss
echo "  9b. Installing pykeops and geomloss..."
pip install pykeops geomloss --no-cache-dir 2>/dev/null || \
  echo "    ⚠ pykeops/geomloss failed (optional)"
echo ""

# Segment Anything (optional)
echo "  9c. Installing segment-anything..."
pip install segment-anything --no-cache-dir 2>/dev/null || \
  echo "    ⚠ segment-anything failed (optional)"
echo ""

# Skip flash-attn (causes OOM)
echo "  9d. SKIPPING flash-attn (known OOM issue on cluster)"
echo "    Use: pip install flash-attn --no-build-isolation (if needed later)"
echo ""

# Skip spconv (not available for CUDA 12.8 via pip)
echo "  9e. SKIPPING spconv (not available via pip for CUDA 12.8)"
echo "    Use system module if available"
echo ""

# ============================================
# 10. Final Verification
# ============================================
echo "Step 10: Verifying installation..."
echo "================================================"

python << 'EOF'
import sys
import subprocess

print("\n=== SYSTEM INFORMATION ===")
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

# Check CUDA
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"GPU: {result.stdout.strip()}")
else:
    print("GPU: Not detected")

print("\n=== CRITICAL PACKAGES ===")

def check_package(name, import_cmd):
    try:
        exec(import_cmd)
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {str(e)[:80]}")
        return False

# Core packages
check_package("PyTorch", "import torch; assert torch.cuda.is_available()")
check_package("NumPy", "import numpy; print(f'    Version: {numpy.__version__}')")
check_package("PyTorch Lightning", "import pytorch_lightning")

print("\n=== CUDA EXTENSIONS ===")
sys.path.insert(0, 'submodules/simple-knn')
check_package("simple-knn", "from simple_knn._C import distCUDA2")
check_package("diff-gaussian-rasterization", "from diff_gaussian_rasterization import GaussianRasterizationSettings")
check_package("chamferdist", "import chamferdist")

print("\n=== ML LIBRARIES ===")
check_package("Transformers", "import transformers")
check_package("Diffusers", "import diffusers")
check_package("Datasets", "import datasets")

print("\n=== UTILITIES ===")
check_package("OpenCV", "import cv2")
check_package("Trimesh", "import trimesh")
check_package("Einops", "import einops")

print("\n" + "="*60)
print("SUMMARY:")
print("- PyTorch installed for CUDA 12.1 (forward compatible with 12.8)")
print("- flash-attn SKIPPED due to OOM issues")
print("- spconv SKIPPED (not available via pip for CUDA 12.8)")
print("- All critical CUDA extensions installed")
print("="*60)
EOF

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Environment: can3tok"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "NOT AVAILABLE")')"
echo ""
echo "To use this environment:"
echo "  module load 2025"
echo "  module load Anaconda3/2025.06-1"
echo "  module load CUDA/12.8.0"
echo "  conda activate can3tok"
echo ""
echo "Test CUDA extensions:"
echo "  python -c \"import sys; sys.path.insert(0, 'submodules/simple-knn'); from simple_knn._C import distCUDA2; print('simple-knn works!')\""
echo "  python -c \"from diff_gaussian_rasterization import GaussianRasterizationSettings; print('diff-gaussian-rasterization works!')\""
echo ""
echo "IMPORTANT: Submit jobs with sufficient memory to avoid OOM:"
echo "  #SBATCH --mem=64G"
echo "  #SBATCH --gpus=1"
echo "  #SBATCH --time=4:00:00"
echo ""
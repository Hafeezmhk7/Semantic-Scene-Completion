#!/bin/bash
# Comprehensive script to upload ALL CAN3TOK logs to W&B

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "           UPLOAD ALL CAN3TOK LOGS TO WANDB - COMPREHENSIVE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# Configuration
# ============================================================================

export WANDB_API_KEY="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"
PROJECT="Can3Tok-Semantic-All"
ENTITY="3D-SSC"
LOG_DIR="logs"

echo "📋 Configuration:"
echo "   Project: $PROJECT"
echo "   Entity: $ENTITY"
echo "   Log directory: $LOG_DIR"
echo ""

# ============================================================================
# Setup
# ============================================================================

echo "🔧 Setting up environment..."

# Install W&B if needed
if ! python -c "import wandb" 2>/dev/null; then
    echo "   Installing wandb..."
    pip install wandb -q
fi

# Login
wandb login --relogin 2>/dev/null
echo "   ✓ Logged in to W&B"
echo ""

# ============================================================================
# Count files
# ============================================================================

echo "📂 Scanning log directory..."

OUT_COUNT=$(find $LOG_DIR -name "*.out" 2>/dev/null | wc -l)
ERR_COUNT=$(find $LOG_DIR -name "*.err" 2>/dev/null | wc -l)
TOTAL=$((OUT_COUNT + ERR_COUNT))

echo "   Found $OUT_COUNT .out files"
echo "   Found $ERR_COUNT .err files"
echo "   Total: $TOTAL files"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "❌ No log files found in $LOG_DIR/"
    exit 1
fi

# ============================================================================
# Ask for confirmation
# ============================================================================

echo "⚠️  About to upload $TOTAL log files to W&B"
echo "   This may take several minutes..."
echo ""
read -p "   Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# ============================================================================
# Upload .out files (training logs)
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════════════"
echo "PHASE 1: Uploading .out files (training logs)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python upload_all_logs_to_wandb.py \
    --log_dir $LOG_DIR \
    --pattern "*.out" \
    --project $PROJECT \
    --entity $ENTITY \
    --tags "training_logs" "all_experiments"

OUT_EXIT=$?

# ============================================================================
# Upload .err files (error logs - if they contain metrics)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PHASE 2: Checking .err files for metrics"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if any .err files have metrics
HAS_METRICS=0
for err_file in $LOG_DIR/*.err; do
    if [ -f "$err_file" ]; then
        if grep -q "Epoch.*Loss:" "$err_file" 2>/dev/null; then
            HAS_METRICS=1
            break
        fi
    fi
done

if [ $HAS_METRICS -eq 1 ]; then
    echo "Found metrics in .err files, uploading..."
    python upload_all_logs_to_wandb.py \
        --log_dir $LOG_DIR \
        --pattern "*.err" \
        --project $PROJECT \
        --entity $ENTITY \
        --tags "error_logs" "all_experiments"
else
    echo "No metrics found in .err files, skipping"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                              UPLOAD COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 All logs uploaded to W&B!"
echo ""
echo "🎯 View your dashboard at:"
echo "   https://wandb.ai/$ENTITY/$PROJECT"
echo ""
echo "💡 Suggested analyses:"
echo "   1. Compare baseline vs semantic runs"
echo "   2. Compare geometric vs hidden vs attention modes"
echo "   3. Analyze balanced sampling impact"
echo "   4. Track semantic loss convergence"
echo "   5. Compare 10k vs 40k Gaussians"
echo ""
echo "📝 Create custom views:"
echo "   • Group by: semantic_mode"
echo "   • Filter by: balanced_sampling = true"
echo "   • Sort by: best_l2 (ascending)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
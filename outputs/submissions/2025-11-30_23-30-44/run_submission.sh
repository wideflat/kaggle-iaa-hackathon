#!/bin/bash
set -e  # Exit on error

# ============================================================
# Kaggle Submission Script
# ============================================================
# Configuration
ITERATIONS=2
WORKERS=2
BATCH_SIZE=2

echo "============================================================"
echo "Starting submission pipeline"
echo "  Iterations: $ITERATIONS"
echo "  Workers: $WORKERS"
echo "  Batch size: $BATCH_SIZE"
echo "============================================================"

# Step 1: Feature Engineering + Hyperparameter Tuning
echo ""
echo "Step 1: Running parallel agent..."
python -m src.agent.parallel_agent \
    -n $ITERATIONS \
    -w $WORKERS \
    -b $BATCH_SIZE \
    --feedback \
    --clear

# Step 2: Generate Submission
echo ""
echo "Step 2: Generating submission..."
python -m src.baseline.make_submission \
    --skip-selection \
    --blend-weight 0.0 # weight for Layer 2a (weighted avg)

echo ""
echo "============================================================"
echo "Done! Check outputs/predictions/ for submission file."
echo "============================================================"

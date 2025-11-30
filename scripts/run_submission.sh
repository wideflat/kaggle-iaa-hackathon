#!/bin/bash
set -e  # Exit on error

# ============================================================
# Kaggle Submission Script
# ============================================================
# Configuration
ITERATIONS=4
WORKERS=2
BATCH_SIZE=5

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
    --clear \
    --tune

# Step 2: Generate Submission
echo ""
echo "Step 2: Generating submission..."
python -m src.baseline.make_submission

echo ""
echo "============================================================"
echo "Done! Check outputs/predictions/ for submission file."
echo "============================================================"

# scripts/ml_pipeline.sh
#!/bin/bash
# Complete ML pipeline orchestrator

# Benefits:
# - End-to-end ML workflow
# - Automatic hyperparameter tuning
# - Model versioning
# - Performance tracking

set -e

# Configuration
DATASET=${1:-"ode_dataset.jsonl"}
MODEL_TYPE=${2:-"pattern"}
EXPERIMENT_NAME=${3:-"exp_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_DIR="experiments/$EXPERIMENT_NAME"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/plots"

echo "Starting ML Pipeline"
echo "Dataset: $DATASET"
echo "Model Type: $MODEL_TYPE"
echo "Experiment: $EXPERIMENT_NAME"

# 1. Data Preparation
echo "Step 1: Preparing data..."
python scripts/dataset_splitter.py "$DATASET" \
    --output-dir "$OUTPUT_DIR/data" \
    --strategy stratified \
    --analyze

# 2. Feature Engineering
echo "Step 2: Feature engineering..."
python -m ml_pipeline.utils prepare_ml_dataset \
    "$DATASET" \
    --output-dir "$OUTPUT_DIR/features"

# 3. Model Training
echo "Step 3: Training model..."
python -m ml_pipeline.train_ode_generator \
    "$OUTPUT_DIR/data/train.jsonl" \
    --model "$MODEL_TYPE" \
    --epochs 100 \
    --output-dir "$OUTPUT_DIR/models" \
    --tensorboard-dir "$OUTPUT_DIR/logs"

# 4. Model Evaluation
echo "Step 4: Evaluating model..."
python -m ml_pipeline.evaluation evaluate \
    --model-path "$OUTPUT_DIR/models/best_model.pth" \
    --test-data "$OUTPUT_DIR/data/test.jsonl" \
    --output-dir "$OUTPUT_DIR/evaluation"

# 5. Generate Novel ODEs
echo "Step 5: Generating novel ODEs..."
python -m ml_pipeline.generation generate \
    --model-path "$OUTPUT_DIR/models/best_model.pth" \
    --n-samples 1000 \
    --output "$OUTPUT_DIR/generated_odes.jsonl"

# 6. Verify Generated ODEs
echo "Step 6: Verifying generated ODEs..."
python scripts/verification_pipeline.py \
    "$OUTPUT_DIR/generated_odes.jsonl" \
    --output "$OUTPUT_DIR/generated_verified.jsonl"

# 7. Analysis and Reporting
echo "Step 7: Generating reports..."
python scripts/comprehensive_analyzer.py \
    "$OUTPUT_DIR/generated_verified.jsonl" \
    --output-dir "$OUTPUT_DIR/analysis"

# 8. Create experiment summary
cat > "$OUTPUT_DIR/experiment_summary.txt" << EOF
Experiment: $EXPERIMENT_NAME
Date: $(date)
Dataset: $DATASET
Model Type: $MODEL_TYPE

Results:
- Training completed: $(ls -la "$OUTPUT_DIR/models/best_model.pth")
- Generated ODEs: $(wc -l < "$OUTPUT_DIR/generated_odes.jsonl")
- Verified ODEs: $(wc -l < "$OUTPUT_DIR/generated_verified.jsonl")

Logs: $OUTPUT_DIR/logs/
Analysis: $OUTPUT_DIR/analysis/
EOF

echo "ML Pipeline complete!"
echo "Results saved to: $OUTPUT_DIR"
# scripts/pipeline_manager.sh
#!/bin/bash
# Master pipeline orchestrator with parallel processing

# Benefits:
# - Automates entire workflow from generation to analysis
# - Parallel processing for faster execution
# - Error handling and logging
# - Resource monitoring

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Configuration
PARALLEL_JOBS=${1:-4}
SAMPLES=${2:-1000}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting ODE Pipeline - $TIMESTAMP"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Samples per combination: $SAMPLES"

# Function to monitor system resources
monitor_resources() {
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') - CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')% - Memory: $(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}')" >> "$LOG_DIR/resources_$TIMESTAMP.log"
        sleep 10
    done
}

# Start resource monitoring in background
monitor_resources &
MONITOR_PID=$!

# Parallel generation
echo "Starting parallel ODE generation..."
parallel -j $PARALLEL_JOBS --bar "python main.py --generator {} --samples $SAMPLES --output ode_dataset_{}.jsonl" ::: L1 L2 L3 L4 N1 N2 N3 N7

# Merge datasets
echo "Merging datasets..."
cat ode_dataset_*.jsonl > "ode_dataset_complete_$TIMESTAMP.jsonl"

# Cleanup temporary files
rm ode_dataset_*.jsonl

# Run analysis
echo "Running analysis..."
python analyze_dataset.py "ode_dataset_complete_$TIMESTAMP.jsonl" --visualize --export

# Stop resource monitoring
kill $MONITOR_PID

echo "Pipeline complete! Dataset: ode_dataset_complete_$TIMESTAMP.jsonl"
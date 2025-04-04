#!/bin/bash
# Script to run audio processing with metrics collection for various sample lengths

# Configuration
CLEANFM_EXECUTABLE="./build/cleanfm"
OUTPUT_DIR="metrics_results"
THREADS=12
VISUALIZATION_SCRIPT="metrics_visualization.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Array of sample lengths to test (in seconds)
SAMPLE_LENGTHS=(2 4 8 16)

# Loop through each sample length
for length in "${SAMPLE_LENGTHS[@]}"; do
    echo "Processing ${length}s audio..."
    
    # Input file
    INPUT_FILE="examples/test_runtime/TEST_${length}s.wav"
    
    # Skip if input file doesn't exist
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: File $INPUT_FILE does not exist, skipping."
        continue
    fi
    
    # Output metrics file
    METRICS_FILE="${OUTPUT_DIR}/metrics_${length}s.csv"
    
    # Run the deconvolution with metrics collection
    echo "Running: $CLEANFM_EXECUTABLE -f $INPUT_FILE -t $THREADS -m $METRICS_FILE"
    $CLEANFM_EXECUTABLE -f "$INPUT_FILE" -t "$THREADS" -m "$METRICS_FILE"
    
    # Check if metrics file was created
    if [ ! -f "$METRICS_FILE" ]; then
        echo "Error: Metrics file $METRICS_FILE was not created."
    else
        echo "Metrics saved to $METRICS_FILE"
        
        # Verify CSV has expected columns
        header=$(head -n 1 "$METRICS_FILE")
        if [[ "$header" != *"cosine_sim"* ]]; then
            echo "Warning: Metrics file does not contain cosine similarity column"
        fi
    fi
    
    echo "Completed ${length}s audio"
    echo "-----------------------------------"
done

# Build metrics files argument string for visualization
METRICS_FILES_ARG=""
for length in "${SAMPLE_LENGTHS[@]}"; do
    METRICS_FILE="${OUTPUT_DIR}/metrics_${length}s.csv"
    if [ -f "$METRICS_FILE" ]; then
        METRICS_FILES_ARG="${METRICS_FILES_ARG} ${length}:${METRICS_FILE}"
    fi
done

# Run visualization if any metrics files were created
if [ -n "$METRICS_FILES_ARG" ]; then
    echo "Running visualization script..."
    python "$VISUALIZATION_SCRIPT" --output_dir "${OUTPUT_DIR}/visualizations" --metrics_files ${METRICS_FILES_ARG}
    echo "Visualizations created in ${OUTPUT_DIR}/visualizations"
    
    # Copy important figures to thesis directory if it exists
    THESIS_FIGURES_DIR="thesis/figures/metrics"
    if [ -d "$THESIS_FIGURES_DIR" ]; then
        echo "Copying key figures to thesis directory..."
        mkdir -p "$THESIS_FIGURES_DIR"
        cp "${OUTPUT_DIR}/visualizations/accuracy_metrics_grid.png" "$THESIS_FIGURES_DIR/"
        cp "${OUTPUT_DIR}/visualizations/cosine_sim_comparison.png" "$THESIS_FIGURES_DIR/"
        cp "${OUTPUT_DIR}/visualizations/convergence_rate.png" "$THESIS_FIGURES_DIR/"
        cp "${OUTPUT_DIR}/visualizations/metrics_comparison_table.tex" "$THESIS_FIGURES_DIR/"
        echo "Figures copied to $THESIS_FIGURES_DIR"
    fi
else
    echo "No metrics files available for visualization."
fi

echo "All benchmarks complete."
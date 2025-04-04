#!/bin/bash
# Script to run audio processing benchmark across different sample rates

# Configuration
CLEANFM_EXECUTABLE="./build/cleanfm"
INPUT_FILE="examples/test_runtime/TEST_4s.wav"
OUTPUT_DIR="sample_rate_results"
THREADS=12
SAMPLE_RATES=(8000 11025 16000 22050 32000 44100 48000)

# Check if python dependencies are installed
if ! python -c "import librosa, soundfile" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip3 install librosa soundfile pandas matplotlib numpy
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist"
    exit 1
fi

# Check if executable exists
if [ ! -f "$CLEANFM_EXECUTABLE" ]; then
    echo "Error: Executable $CLEANFM_EXECUTABLE not found"
    exit 1
fi

echo "=== Starting Sample Rate Benchmark ==="
echo "Executable: $CLEANFM_EXECUTABLE"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Thread count: $THREADS"
echo "Sample rates: ${SAMPLE_RATES[*]}"
echo "====================================="

# Build sample rates argument string
SAMPLE_RATES_ARG=""
for rate in "${SAMPLE_RATES[@]}"; do
    SAMPLE_RATES_ARG="$SAMPLE_RATES_ARG $rate"
done

# Run the benchmark
python run_sample_rate_benchmarks.py \
    --input "$INPUT_FILE" \
    --executable "$CLEANFM_EXECUTABLE" \
    --threads "$THREADS" \
    --output_dir "$OUTPUT_DIR" \
    --sample_rates $SAMPLE_RATES_ARG

# Check if the benchmarks were successful
if [ -d "$OUTPUT_DIR/plots" ]; then
    echo "Sample rate benchmarks completed successfully"
    echo "Results available in $OUTPUT_DIR/plots"
    
    # Copy important plots to thesis directory if it exists
    THESIS_FIGURES_DIR="thesis/figures/sample_rate"
    if [ -d "thesis" ]; then
        mkdir -p "$THESIS_FIGURES_DIR"
        echo "Copying key figures to thesis directory..."
        cp "$OUTPUT_DIR/plots/sample_rate_vs_"*.png "$THESIS_FIGURES_DIR/"
        cp "$OUTPUT_DIR/plots/sr_comparison_"*.png "$THESIS_FIGURES_DIR/"
        echo "Figures copied to $THESIS_FIGURES_DIR"
    fi
else
    echo "Error: Benchmark did not complete successfully"
fi
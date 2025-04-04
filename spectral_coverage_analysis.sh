#!/bin/bash
# Script to run spectral coverage analysis for Dirichlet Kernel Deconvolution

# Configuration
CLEANFM_EXECUTABLE="./build/cleanfm"
INPUT_FILE="examples/test_runtime/TEST_2s.wav"
OUTPUT_DIR="spectral_coverage_results"
THREADS=12
COMPONENT_COUNTS=(1000 5000 10000 25000 50000)

# Check if python dependencies are installed
if ! python -c "import librosa, soundfile, numpy, matplotlib" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip3 install librosa soundfile numpy matplotlib pandas
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

echo "=== Starting Spectral Coverage Analysis ==="
echo "Executable: $CLEANFM_EXECUTABLE"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Thread count: $THREADS"
echo "Component counts: ${COMPONENT_COUNTS[*]}"
echo "==========================================="

# Create output directories
mkdir -p "$OUTPUT_DIR/metrics"
mkdir -p "$OUTPUT_DIR/components"
mkdir -p "$OUTPUT_DIR/audio"
mkdir -p "$OUTPUT_DIR/plots"

# Process each component count
for comp_count in "${COMPONENT_COUNTS[@]}"; do
    echo "Processing with $comp_count components..."
    
    # Run deconvolution
    METRICS_FILE="$OUTPUT_DIR/metrics/metrics_${comp_count}.csv"
    COMPONENTS_FILE="$OUTPUT_DIR/components/components_${comp_count}.csv"
    AUDIO_FILE="$OUTPUT_DIR/audio/reconstructed_${comp_count}.wav"
    
    echo "Running deconvolution with $comp_count components limit..."
    # Run deconvolution with component limit using new --max-components flag
    $CLEANFM_EXECUTABLE -f "$INPUT_FILE" -t "$THREADS" -o "$AUDIO_FILE" -m "$METRICS_FILE" -c "$COMPONENTS_FILE" --max-components "$comp_count"
    
    # Check if output files were created
    if [ ! -f "$METRICS_FILE" ] || [ ! -f "$COMPONENTS_FILE" ]; then
        echo "⚠ ERROR: Deconvolution failed for $comp_count components"
        continue
    fi
    
    
    if [ ! -f "$AUDIO_FILE" ]; then
        echo "⚠ ERROR: Decompression failed for $comp_count components"
        continue
    fi
    
    echo "✓ Processed $comp_count components"
done

# Run spectral coverage analysis
echo "Running spectral coverage analysis..."

# Build component counts argument string
COMPONENTS_ARG=""
for count in "${COMPONENT_COUNTS[@]}"; do
    COMPONENTS_ARG="$COMPONENTS_ARG $count"
done

# Create analysis script
ANALYSIS_SCRIPT="$OUTPUT_DIR/spectral_coverage_analysis.py"
cp spectral_coverage_analysis.py "$ANALYSIS_SCRIPT"
chmod +x "$ANALYSIS_SCRIPT"

# Run analysis
python "$ANALYSIS_SCRIPT" \
    --input "$INPUT_FILE" \
    --executable "$CLEANFM_EXECUTABLE" \
    --output_dir "$OUTPUT_DIR" \
    --threads "$THREADS" \
    --components $COMPONENTS_ARG

# Check if the analysis was successful
if [ -f "$OUTPUT_DIR/frequency_band_coverage.png" ]; then
    echo "✓ Spectral coverage analysis completed successfully"
    
    # Copy to thesis directory if it exists
    THESIS_DIR="thesis/figures/spectral_coverage"
    if [ -d "thesis" ]; then
        mkdir -p "$THESIS_DIR"
        echo "Copying figures to thesis directory..."
        cp "$OUTPUT_DIR"/*.png "$THESIS_DIR/"
        cp "$OUTPUT_DIR"/*.tex "$THESIS_DIR/" 2>/dev/null || true
        echo "✓ Thesis materials prepared in $THESIS_DIR"
    fi
else
    echo "⚠ ERROR: Spectral coverage analysis failed"
fi

echo "Done!"
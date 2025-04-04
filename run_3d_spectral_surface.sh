

ORIGINAL_AUDIO="examples/test_runtime/TEST_2s.wav"
SPECTRAL_COVERAGE_DIR="spectral_coverage_results"
OUTPUT_DIR="spectral_surface_plots"
COMPONENT_COUNTS=(512 1024 5120 10240 12800 15360 17920 20480 23040 25600 28160 30720 33280 35840 38400 40960)

# Check if the script exists
if [ ! -f "3d_spectral_surface.py" ]; then
    echo "Copying 3D surface script..."
    cp "$0" "3d_spectral_surface.py"
    echo "Please run this script again after copying the content."
    exit 1
fi

# Check if dependencies are installed
if ! python -c "import numpy, matplotlib, librosa" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip3 install numpy matplotlib librosa pandas scipy
fi

# Check if original audio exists
if [ ! -f "$ORIGINAL_AUDIO" ]; then
    echo "Error: Original audio file not found at $ORIGINAL_AUDIO"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Gather the reconstructed audio files
RECONSTRUCTED_FILES=""
COMPONENT_ARGS=""

for comp_count in "${COMPONENT_COUNTS[@]}"; do
    RECON_FILE="$SPECTRAL_COVERAGE_DIR/audio/reconstructed_${comp_count}.wav"
    
    if [ -f "$RECON_FILE" ]; then
        RECONSTRUCTED_FILES="$RECONSTRUCTED_FILES $RECON_FILE"
        COMPONENT_ARGS="$COMPONENT_ARGS $comp_count"
    else
        echo "Warning: Reconstructed file not found: $RECON_FILE"
    fi
done

if [ -z "$RECONSTRUCTED_FILES" ]; then
    echo "Error: No reconstructed audio files found"
    exit 1
fi

# Run the 3D surface plot script
echo "Generating 3D spectral surface plots..."
echo "Original audio: $ORIGINAL_AUDIO"
echo "Reconstructed files: $RECONSTRUCTED_FILES"
echo "Component counts: $COMPONENT_ARGS"

python 3d_spectral_surface.py \
    --original "$ORIGINAL_AUDIO" \
    --reconstructed $RECONSTRUCTED_FILES \
    --components $COMPONENT_ARGS \
    --output_dir "$OUTPUT_DIR" \
    --time_slice 1.0

# Also generate mid-section view
python 3d_spectral_surface.py \
    --original "$ORIGINAL_AUDIO" \
    --reconstructed $RECONSTRUCTED_FILES \
    --components $COMPONENT_ARGS \
    --output_dir "${OUTPUT_DIR}_midsection" \
    --time_slice 0.5 1.5

# Copy plots to thesis directory if it exists
THESIS_DIR="thesis/figures/spectral_surface"
if [ -d "thesis" ]; then
    mkdir -p "$THESIS_DIR"
    echo "Copying plots to thesis directory..."
    cp "$OUTPUT_DIR"/*.png "$THESIS_DIR/"
    echo "Plots copied to $THESIS_DIR"
fi

echo "Done!"
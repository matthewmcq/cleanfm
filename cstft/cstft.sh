#!/bin/bash
# Script to run Cleaned STFT analysis on audio files
# Modified to accept audio file paths as command line arguments

# --- Configuration ---
DKD_EXECUTABLE="./../build/cleanfm" # Path to the DKD executable
PYTHON_SCRIPT="cleaned_stft.py" # Path to the Python script for frame preparation and visualization
OUTPUT_DIR="cleaned_stft_results" # Directory to store all output
THREADS=12 # Number of threads to use for processing
VISUALIZATION_DIR="stft_visualizations" # Subdirectory within OUTPUT_DIR for final visualizations

# --- STFT Parameters ---
N_FFT=8192 # FFT window size
HOP_LENGTH=128 # Hop length for STFT
MAX_COMPONENTS=50000 # Maximum components to extract per frame

# --- Audio Processing Parameters ---
DURATION=10  # Maximum duration to process in seconds for larger files
SAMPLE_RATE=48000  # Force consistent sample rate

# --- Audio Files (Now from Command Line Arguments) ---
# Check if any arguments were provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <audio_file1> [<audio_file2> ...]"
    echo "Please provide at least one audio file path as an argument."
    exit 1
fi

# Read command line arguments into the AUDIO_FILES array
AUDIO_FILES=("$@")

echo "Found ${#AUDIO_FILES[@]} audio files to process from command line arguments."

# --- Create Output Directories ---
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/$VISUALIZATION_DIR"

# --- Process Each Audio File ---
for audio_file in "${AUDIO_FILES[@]}"; do
    # Check if the audio file exists
    if [ ! -f "$audio_file" ]; then
        echo "Error: Audio file not found: $audio_file. Skipping."
        continue
    fi

    # Get base filename without extension
    filename=$(basename -- "$audio_file")
    base_name="${filename%.*}"

    echo "Processing $filename..."

    # Create file-specific output directories
    # These directories will store temporary frame WAVs, component CSVs, and intermediate results
    file_output_dir="$OUTPUT_DIR/$base_name"
    mkdir -p "$file_output_dir"

    # Step 1: Run Python script to prepare frames and job list
    # The Python script splits the audio, saves frames, and generates a job_list.txt
    echo "Step 1: Preparing frames for $filename..."
    python "$PYTHON_SCRIPT" "$audio_file" \
        --output-dir "$file_output_dir" \
        --n-fft "$N_FFT" \
        --hop-length "$HOP_LENGTH" \
        --max-components "$MAX_COMPONENTS" \
        --sr "$SAMPLE_RATE" \
        --duration "$DURATION" \
        --threads "$THREADS" # Note: This thread count is for the Python script's preparation phase

    # Check if job list was created by the Python script
    job_list="$file_output_dir/job_list.txt"
    if [ ! -f "$job_list" ]; then
        echo "Error: Job list not created by Python script for $filename. Skipping."
        continue
    fi

    # Step 2: Process each frame with DKD
    echo "Step 2: Processing frames with DKD for $filename..."
    # Count total frames from the job list file
    total_frames=$(wc -l < "$job_list")
    counter=0 # Initialize frame counter

    # Read the job list line by line. Each line contains: wav_path,csv_path,max_comp,thread_count
    while IFS=, read -r wav_path csv_path max_comp thread_count; do
        # Run the DKD executable for the current frame's WAV file and output CSV path
        # -ni flag indicates non-iterative mode (if supported/desired by cleanfm)
        "$DKD_EXECUTABLE" -f "$wav_path" -c "$csv_path" -t "$THREADS" -mc "$MAX_COMPONENTS" -ni

        # Update and print progress
        counter=$((counter + 1))
        # Calculate percentage, handling division by zero if total_frames is 0
        percentage=0
        if [ "$total_frames" -gt 0 ]; then
            percentage=$((counter * 100 / total_frames))
        fi
        printf "\rProcessed %d/%d frames (%d%%)" "$counter" "$total_frames" "$percentage"
    done < "$job_list" # Read from the generated job_list.txt
    echo ""  # Print a newline character to finish the progress line

    # Step 3: Generate visualization and clean up temporary files
    # The Python script is run again, this time using the --clean flag
    # It reads the generated component CSVs and creates visualizations.
    echo "Step 3: Generating visualization and cleaning up for $filename..."
    python "$PYTHON_SCRIPT" "$audio_file" \
        --output-dir "$file_output_dir" \
        --n-fft "$N_FFT" \
        --hop-length "$HOP_LENGTH" \
        --max-components "$MAX_COMPONENTS" \
        --sr "$SAMPLE_RATE" \
        --duration "$DURATION" \
        --threads "$THREADS" \
        --clean  # This flag tells the Python script to clean up temporary WAV files

    # Copy visualization files to the central visualization directory
    # Check if visualization files exist before copying
    if [ -f "$file_output_dir/stft_comparison.png" ]; then
        cp "$file_output_dir/stft_comparison.png" "$OUTPUT_DIR/$VISUALIZATION_DIR/${base_name}_comparison.png"
    else
        echo "Warning: Comparison visualization not found for $filename."
    fi
     if [ -f "$file_output_dir/stft_difference.png" ]; then
        cp "$file_output_dir/stft_difference.png" "$OUTPUT_DIR/$VISUALIZATION_DIR/${base_name}_difference.png"
    else
        echo "Warning: Difference visualization not found for $filename."
    fi


    echo "Completed processing $filename"
    echo "-----------------------------------"
done

# --- Optional: Copy visualizations to thesis figures directory ---
THESIS_FIGURES_DIR="thesis/figures/cleaned_stft"
if [ -d "thesis" ]; then
    echo "Copying visualizations to thesis directory..."
    mkdir -p "$THESIS_FIGURES_DIR"
    # Copy all generated visualization PNGs
    cp "$OUTPUT_DIR/$VISUALIZATION_DIR"/*.png "$THESIS_FIGURES_DIR/"
    echo "Visualizations copied to $THESIS_FIGURES_DIR"
fi

echo "All processing complete."
echo "Final visualizations saved to: $OUTPUT_DIR/$VISUALIZATION_DIR"

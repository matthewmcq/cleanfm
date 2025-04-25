#!/usr/bin/env python3
"""
Cleaned STFT Implementation

This script implements a "Cleaned STFT" by:
1. Chopping audio into STFT-sized segments
2. Preparing each segment for DKD processing
3. Processing the DKD results
4. Visualizing and comparing with traditional STFT

Usage:
    python cleaned_stft.py input.wav [options]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import json
import argparse
from scipy import signal
import pandas as pd
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cleaned STFT Implementation using DKD')
    parser.add_argument('input_file', type=str, help='Input audio file')
    parser.add_argument('--output-dir', type=str, default='./cleaned_stft_output',
                      help='Output directory for temporary files and results')
    parser.add_argument('--n-fft', type=int, default=2048,
                      help='FFT size for STFT')
    parser.add_argument('--hop-length', type=int, default=512,
                      help='Hop length for STFT')
    parser.add_argument('--win-length', type=int, default=None,
                      help='Window length for STFT (default: n_fft)')
    parser.add_argument('--window', type=str, default='hann',
                      help='Window function for STFT')
    parser.add_argument('--max-components', type=int, default=200,
                      help='Maximum number of components per frame for DKD')
    parser.add_argument('--sr', type=int, default=None,
                      help='Sample rate for audio loading (default: native)')
    parser.add_argument('--duration', type=float, default=None,
                      help='Duration of audio to process in seconds (default: entire file)')
    parser.add_argument('--threads', type=int, default=4,
                      help='Number of threads for DKD processing')
    parser.add_argument('--clean', action='store_true',
                      help='Clean temporary files after processing')
    parser.add_argument('--dkd-path', type=str, default='./dkd',
                      help='Path to DKD executable')
    parser.add_argument('--frequency-max', type=float, default=None,
                      help='Maximum frequency to display in Hz')
    parser.add_argument('--db-range', type=float, default=80,
                      help='dB range for display')
    
    return parser.parse_args()


def create_frames(y, sr, n_fft, hop_length, win_length=None, window='hann'):
    """
    Chop audio into STFT-sized frames
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        window: Window function
    
    Returns:
        frames: List of windowed audio frames
        indexes: List of start indices for each frame
    """
    if win_length is None:
        win_length = n_fft
    
    # Get the window function
    if window == 'hann':
        win_func = np.hanning(win_length)
    elif window == 'hamming':
        win_func = np.hamming(win_length)
    elif window == 'blackman':
        win_func = np.blackman(win_length)
    elif window == 'rectangular' or window == 'rect':
        win_func = np.ones(win_length)
    else:
        raise ValueError(f"Unsupported window type: {window}")
    
    # Pad win_func if needed
    if win_length < n_fft:
        win_func = np.pad(win_func, (0, n_fft - win_length))
    
    # Calculate the number of frames
    n_frames = 1 + int((len(y) - n_fft) / hop_length)
    
    frames = []
    indexes = []
    
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + n_fft
        
        if end_idx <= len(y):
            frame = y[start_idx:end_idx]
            # Apply window function
            windowed_frame = frame * win_func
            frames.append(windowed_frame)
            indexes.append(start_idx)
    
    return frames, indexes


def prepare_frame_for_dkd(frame, output_dir, frame_idx):
    """
    Prepare a frame for DKD processing by saving it as a WAV file
    
    Args:
        frame: Audio frame
        output_dir: Output directory
        frame_idx: Frame index
    
    Returns:
        wav_path: Path to the saved WAV file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the frame as a WAV file
    wav_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.wav")
    sf.write(wav_path, frame, args.sr)
    
    return wav_path


def read_dkd_components(csv_path):
    """
    Read DKD components from a CSV file
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        components: DataFrame of components
    """
    try:
        components = pd.read_csv(csv_path)
        return components
    except Exception as e:
        print(f"Error reading components from {csv_path}: {e}")
        return pd.DataFrame()


def components_to_magnitude_spectrum(components, n_fft, sr):
    """
    Convert DKD components to a magnitude spectrum
    
    Args:
        components: DataFrame of components
        n_fft: FFT size
        sr: Sample rate
    
    Returns:
        magnitude: Magnitude spectrum
    """
    # Initialize empty spectrum (we'll use n_fft//2 + 1 bins to match librosa STFT output)
    freq_bins = n_fft // 2 + 1
    magnitude = np.zeros(freq_bins)
    
    # Convert each component's frequency to the nearest bin
    for _, component in components.iterrows():
        freq_hz = component['frequency_hz']
        amplitude = component['amplitude']
        
        # Convert Hz to bin index
        bin_idx = int(np.round(freq_hz * n_fft / sr))
        
        # Only add components within the valid frequency range
        if 0 <= bin_idx < freq_bins:
            magnitude[bin_idx] += amplitude
    
    return magnitude


def plot_comparison(y, sr, frames, cleaned_stft, args):
    """
    Plot comparison between traditional STFT and Cleaned STFT
    
    Args:
        y: Audio signal
        sr: Sample rate
        frames: List of frame indices
        cleaned_stft: Cleaned STFT magnitude matrix
        args: Command line arguments
    """
    # Compute traditional STFT
    traditional_stft = np.abs(librosa.stft(y, 
                                        n_fft=args.n_fft, 
                                        hop_length=args.hop_length,
                                        win_length=args.win_length,
                                        window=args.window))
    
    # Trim to match the number of frames in cleaned_stft if needed
    min_frames = min(traditional_stft.shape[1], cleaned_stft.shape[1])
    traditional_stft = traditional_stft[:, :min_frames]
    cleaned_stft = cleaned_stft[:, :min_frames]
    
    # Convert to dB scale with the same reference
    traditional_stft_db = librosa.amplitude_to_db(traditional_stft, ref=np.max)
    cleaned_stft_db = librosa.amplitude_to_db(cleaned_stft, ref=np.max)
    
    # Create the figure
    plt.figure(figsize=(15, 10))
    
    # Plot traditional STFT
    plt.subplot(2, 1, 1)
    img1 = librosa.display.specshow(traditional_stft_db, 
                                   x_axis='time', 
                                   y_axis='log',
                                   sr=sr,
                                   hop_length=args.hop_length,
                                   vmin=traditional_stft_db.max() - args.db_range)
    plt.colorbar(img1, format='%+2.0f dB')
    plt.title('Traditional STFT')
    if args.frequency_max:
        plt.ylim(0, args.frequency_max)
    
    # Plot Cleaned STFT
    plt.subplot(2, 1, 2)
    img2 = librosa.display.specshow(cleaned_stft_db, 
                                   x_axis='time', 
                                   y_axis='log',
                                   sr=sr,
                                   hop_length=args.hop_length,
                                   vmin=cleaned_stft_db.max() - args.db_range)
    plt.colorbar(img2, format='%+2.0f dB')
    plt.title('Cleaned STFT (DKD)')
    if args.frequency_max:
        plt.ylim(0, args.frequency_max)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stft_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'stft_comparison.pdf'))
    
    # Also create a difference plot
    plt.figure(figsize=(15, 5))
    diff = cleaned_stft_db - traditional_stft_db
    img3 = librosa.display.specshow(diff,
                                   x_axis='time',
                                   y_axis='hz',
                                   sr=sr,
                                   hop_length=args.hop_length,
                                   cmap='coolwarm')
    plt.colorbar(img3, format='%+2.0f dB')
    plt.title('Difference: Cleaned STFT - Traditional STFT (dB)')
    if args.frequency_max:
        plt.ylim(0, args.frequency_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stft_difference.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'stft_difference.pdf'))
    
    plt.close('all')


def main(args):
    # Load audio
    y, sr = librosa.load(args.input_file, sr=args.sr, duration=args.duration, mono=False)
    y= y[0]
    
    # Set sample rate if not provided
    if args.sr is None:
        args.sr = sr
    
    # Set window length if not provided
    if args.win_length is None:
        args.win_length = args.n_fft
    
    print(f"Processing audio: {args.input_file}")
    print(f"Sample rate: {sr} Hz, Duration: {len(y)/sr:.2f} seconds")
    print(f"STFT parameters: n_fft={args.n_fft}, hop_length={args.hop_length}, win_length={args.win_length}, window={args.window}")
    
    # Create output directory
    output_dir = args.output_dir
    frames_dir = os.path.join(output_dir, "frames")
    components_dir = os.path.join(output_dir, "components")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(components_dir, exist_ok=True)
    
    # Chop audio into frames
    frames, frame_indexes = create_frames(y, sr, args.n_fft, args.hop_length, args.win_length, args.window)
    print(f"Created {len(frames)} frames")
    
    # Save frames for DKD processing and create job list
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = prepare_frame_for_dkd(frame, frames_dir, i)
        frame_paths.append((i, frame_path))
    
    # Create a job list file for the bash script
    job_list_path = os.path.join(output_dir, "job_list.txt")
    with open(job_list_path, 'w') as f:
        for i, frame_path in frame_paths:
            csv_path = os.path.join(components_dir, f"frame_{i:04d}_components.csv")
            f.write(f"{frame_path},{csv_path},{args.max_components},{args.threads}\n")
    
    print(f"Created job list with {len(frame_paths)} jobs")
    print(f"Run DKD on each frame using the job list: {job_list_path}")
    print("After DKD processing, run this script again with --no-prepare option to generate visualizations")
    
    # Get all CSV files with components
    component_files = sorted(Path(components_dir).glob("frame_*_components.csv"))
    
    if not component_files:
        print("No component files found. Run DKD on the frames first.")
        return
    
    print(f"Found {len(component_files)} component files")
    
    # Read components and create magnitude spectra
    cleaned_stft = []
    for i, csv_path in enumerate(component_files):
        components = read_dkd_components(csv_path)
        if not components.empty:
            magnitude = components_to_magnitude_spectrum(components, args.n_fft, sr)
            cleaned_stft.append(magnitude)
        else:
            # If no components found, use zeros
            cleaned_stft.append(np.zeros(args.n_fft // 2 + 1))
    
    cleaned_stft = np.array(cleaned_stft).T  # Transpose to match librosa STFT shape
    
    # Plot comparison
    plot_comparison(y, sr, frame_indexes, cleaned_stft, args)
    
    print("Generated visualizations:")
    print(f"  {os.path.join(output_dir, 'stft_comparison.png')}")
    print(f"  {os.path.join(output_dir, 'stft_difference.png')}")
    
    # Clean up temporary files if requested
    if args.clean:
        print("Cleaning up temporary files...")
        for i, frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.remove(frame_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
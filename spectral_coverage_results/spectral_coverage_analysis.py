#!/usr/bin/env python3
"""
Spectral coverage analysis for Dirichlet Kernel Deconvolution algorithm.
Visualizes how well different numbers of components cover the frequency spectrum.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import subprocess
from matplotlib.colors import LogNorm
import pandas as pd

def run_deconvolution(executable_path, input_file, threads, components_limit, metrics_file, components_file):
    """Run the Dirichlet kernel deconvolution with specified component limit"""
    cmd = [executable_path, "-f", input_file, "-t", str(threads), 
           "-m", metrics_file, "-c", components_file, "--max-components", str(components_limit)]
    
    print(f"Running deconvolution with {components_limit} components limit...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running deconvolution: {e}")
        return False

def run_decompression(executable_path, input_file, components_file, output_file, threads):
    """Run the decompression to create reconstructed audio"""
    cmd = [executable_path, "-f", input_file, "-c", components_file, "-o", output_file, "-t", str(threads)]
    
    print(f"Running decompression to {output_file}...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running decompression: {e}")
        return False

def compute_stft(audio_file, n_fft=2048, hop_length=512):
    """Compute and return STFT of audio file"""
    y, sr = librosa.load(audio_file, sr=48000, mono=False)
    # print(y.shape)
    
    if y.ndim > 1:
        # take left channel if stereo
        y = y[0]
    # print(y.shape)
    # Compute STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr, y

def plot_stft_comparison(original_file, reconstructed_files, component_counts, output_dir, n_fft=2048, hop_length=512):
    """Plot STFT comparison between original and reconstructed files"""
    # Compute STFT for original file
    orig_S, sr, orig_y = compute_stft(original_file, n_fft, hop_length)
    
    # Create figure
    n_files = len(reconstructed_files)
    fig, axs = plt.subplots(n_files + 1, 1, figsize=(12, 4 * (n_files + 1)), sharex=True)
    
    
    
    # Plot original STFT
    img = librosa.display.specshow(orig_S, x_axis='time', y_axis='log', sr=sr, 
                                  hop_length=hop_length, ax=axs[0])
    axs[0].set_title("Original Audio Spectrogram", fontsize=14)
    axs[0].set_ylabel("Frequency (Hz)", fontsize=12)
    
    max_diff = 0
    min_diff = np.infty
    for i, (recon_file, comp_count) in enumerate(zip(reconstructed_files, component_counts)):
        # Compute absolute difference
        recon_S, _, _ = compute_stft(recon_file, n_fft, hop_length)
        diff = np.abs(orig_S - recon_S)
        
        # Find min and max for color scaling
        if np.max(diff) > max_diff:
            max_diff = np.max(diff)
        if np.min(diff) < min_diff:
            min_diff = np.min(diff)
            
        
    
    # Plot reconstructed STFTs
    for i, (recon_file, comp_count) in enumerate(zip(reconstructed_files, component_counts)):
        recon_S, _, _ = compute_stft(recon_file, n_fft, hop_length)
        
        # normalize to original
        
        librosa.display.specshow(recon_S, x_axis='time', y_axis='log', sr=sr, 
                                hop_length=hop_length, ax=axs[i+1])
        axs[i+1].set_title(f"Reconstructed with {comp_count} Components", fontsize=14)
        axs[i+1].set_ylabel("Frequency (Hz)", fontsize=12)
    
    # Add a colorbar
    # fig.colorbar(img, ax=axs, format="%+2.0f dB")
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, "stft_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved STFT comparison to {output_path}")
    
    return orig_S, [compute_stft(f, n_fft, hop_length)[0] for f in reconstructed_files]

def plot_spectral_difference(orig_S, recon_S_list, component_counts, output_dir):
    """Plot spectral difference (error) between original and reconstructed STFTs"""
    n_recon = len(recon_S_list)
    fig, axs = plt.subplots(n_recon, 1, figsize=(12, 4 * n_recon), sharey=True)
    
    if n_recon == 1:
        axs = [axs]  # Make it iterable if there's only one reconstruction
        
    #want same max/min for all plots
    max_diff = 0
    min_diff = np.infty
    for i, (recon_S, comp_count) in enumerate(zip(recon_S_list, component_counts)):
        # Compute absolute difference
        diff = np.abs(orig_S - recon_S)
        
        # Find min and max for color scaling
        if np.max(diff) > max_diff:
            max_diff = np.max(diff)
        if np.min(diff) < min_diff:
            min_diff = np.min(diff)
            
        
        
    
    # Plot differences
    
    for i, (recon_S, comp_count) in enumerate(zip(recon_S_list, component_counts)):
        # Compute absolute difference
        diff = np.abs(orig_S - recon_S)
        
        new_max = max_diff
        new_min = min_diff
        # Normalize the difference to the original
        # diff = (diff - new_min) / (new_max - new_min)
        
        
        # Plot difference
        img = librosa.display.specshow(diff, x_axis='time', y_axis='log', 
                                      ax=axs[i], cmap='magma')
        axs[i].set_title(f"Spectral Difference with {comp_count} Components", fontsize=14)
        axs[i].set_ylabel("Frequency (Hz)", fontsize=12)
    
    
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, "spectral_difference.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved spectral difference plot to {output_path}")

def plot_frequency_band_coverage(orig_S, recon_S_list, component_counts, output_dir, sr, n_bands=8):
    """Plot coverage percentage across frequency bands"""
    # Define frequency bands (in Hz) - logarithmically spaced
    max_freq = sr // 2
    bands = np.logspace(np.log10(20), np.log10(max_freq), n_bands+1)
    band_names = [f"{int(bands[i])}-{int(bands[i+1])}Hz" for i in range(n_bands)]
    
    # Convert frequency bands to bin indices
    n_bins = orig_S.shape[0]
    bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_bins-1)*2)
    band_bins = []
    for i in range(n_bands):
        start_idx = np.searchsorted(bin_freqs, bands[i])
        end_idx = np.searchsorted(bin_freqs, bands[i+1])
        band_bins.append((start_idx, end_idx))
    
    # Calculate coverage for each band and reconstruction
    coverage_data = []
    
    for comp_count, recon_S in zip(component_counts, recon_S_list):
        band_coverage = []
        
        for i, (start_idx, end_idx) in enumerate(band_bins):
            if start_idx == end_idx:
                # Skip empty bands
                band_coverage.append(0)
                continue
                
            # Calculate error in this band
            band_orig = orig_S[start_idx:end_idx, :]
            band_recon = recon_S[start_idx:end_idx, :]
            
            # Normalize by original power in the band
            orig_power = np.mean(np.abs(band_orig))
            if orig_power < 1e-10:
                # Skip bands with no energy
                band_coverage.append(0)
                continue
            
            # Calculate coverage as 1 - normalized error
            error = np.mean(np.abs(band_orig - band_recon)) / orig_power
            coverage = max(0, min(100, 100 * (1 - error)))
            band_coverage.append(coverage)
        
        coverage_data.append((comp_count, band_coverage))
    
    # Plot frequency band coverage
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(band_names))
    width = 0.8 / len(component_counts)
    
    for i, (comp_count, band_coverage) in enumerate(coverage_data):
        offset = (i - len(component_counts)/2 + 0.5) * width
        plt.bar(x + offset, band_coverage, width, label=f"{comp_count} Components")
    
    plt.xlabel("Frequency Band", fontsize=14)
    plt.ylabel("Coverage (%)", fontsize=14)
    plt.title("Spectral Coverage by Frequency Band", fontsize=16)
    plt.xticks(x, band_names, rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    output_path = os.path.join(output_dir, "frequency_band_coverage.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved frequency band coverage plot to {output_path}")

    # Create a heatmap of coverage by band and component count
    plt.figure(figsize=(12, 6))
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(component_counts), len(band_names)))
    for i, (comp_count, band_coverage) in enumerate(coverage_data):
        heatmap_data[i] = band_coverage
    
    # Plot heatmap
    im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Coverage (%)')
    
    # Add labels
    plt.yticks(np.arange(len(component_counts)), [str(c) for c in component_counts])
    plt.xticks(np.arange(len(band_names)), band_names, rotation=45, ha='right')
    plt.xlabel('Frequency Band', fontsize=14)
    plt.ylabel('Component Count', fontsize=14)
    plt.title('Spectral Coverage Heatmap', fontsize=16)
    
    # Add coverage percentage text on each cell
    for i in range(len(component_counts)):
        for j in range(len(band_names)):
            plt.text(j, i, f"{heatmap_data[i, j]:.1f}%", 
                    ha="center", va="center", color="black" if heatmap_data[i, j] > 50 else "white")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "coverage_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved coverage heatmap to {output_path}")
    
    # Create LaTeX table with coverage data
    with open(os.path.join(output_dir, "coverage_table.tex"), 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Spectral Coverage (\\%) by Frequency Band and Component Count}\n")
        f.write("\\begin{tabular}{|l|")
        for _ in band_names:
            f.write("c|")
        f.write("}\n")
        f.write("\\hline\n")
        
        # Header row
        f.write("Components & ")
        f.write(" & ".join(band_names))
        f.write(" \\\\ \\hline\n")
        
        # Data rows
        for i, (comp_count, band_coverage) in enumerate(coverage_data):
            f.write(f"{comp_count} & ")
            f.write(" & ".join([f"{coverage:.1f}\\%" for coverage in band_coverage]))
            f.write(" \\\\ \\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table to {os.path.join(output_dir, 'coverage_table.tex')}")
    
    return coverage_data, band_names

def main():
    parser = argparse.ArgumentParser(description='Analyze spectral coverage of Dirichlet kernel deconvolution')
    parser.add_argument('--input', type=str, required=True, help='Input audio file')
    parser.add_argument('--executable', type=str, required=True, help='Path to cleanfm executable')
    parser.add_argument('--output_dir', type=str, default='spectral_coverage', help='Output directory')
    parser.add_argument('--threads', type=int, default=12, help='Number of threads to use')
    parser.add_argument('--components', type=int, nargs='+', default=[1000, 5000, 10000, 30000], 
                      help='Component counts to analyze')
    parser.add_argument('--run', action='store_true', help='Run deconvolution and decompression')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'components'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    
    # Define output files
    metrics_files = []
    components_files = []
    reconstructed_files = []
    
    for comp_count in args.components:
        metrics_file = os.path.join(args.output_dir, 'metrics', f'metrics_{comp_count}.csv')
        components_file = os.path.join(args.output_dir, 'components', f'components_{comp_count}.csv')
        reconstructed_file = os.path.join(args.output_dir, 'audio', f'reconstructed_{comp_count}.wav')
        
        metrics_files.append(metrics_file)
        components_files.append(components_file)
        reconstructed_files.append(reconstructed_file)
    
    # If requested, run deconvolution and decompression
    if args.run:
        # Run deconvolution for each component count
        for i, comp_count in enumerate(args.components):
            if not run_deconvolution(args.executable, args.input, args.threads, 
                                    comp_count, metrics_files[i], components_files[i]):
                print(f"Failed to run deconvolution for {comp_count} components")
        
        # Generate reconstructed audio for each component count
        for i, comp_count in enumerate(args.components):
            if not run_decompression(args.executable, args.input, components_files[i], 
                                   reconstructed_files[i], args.threads):
                print(f"Failed to decompress components for {comp_count} components")
    
    # Check which reconstructed files actually exist
    existing_reconstructed = []
    existing_components = []
    for i, file_path in enumerate(reconstructed_files):
        if os.path.exists(file_path):
            existing_reconstructed.append(file_path)
            existing_components.append(args.components[i])
        else:
            print(f"Warning: Reconstructed file {file_path} not found")
    
    # Check if we have reconstructed files
    if not existing_reconstructed:
        print("No reconstructed files available for analysis")
        return
    
    # Plot STFT comparison
    orig_S, recon_S_list = plot_stft_comparison(
        args.input, existing_reconstructed, existing_components, args.output_dir
    )
    
    # Plot spectral difference
    plot_spectral_difference(orig_S, recon_S_list, existing_components, args.output_dir)
    
    # Get sample rate
    sr = 48000
    
    # Plot frequency band coverage
    plot_frequency_band_coverage(orig_S, recon_S_list, existing_components, args.output_dir, sr)
    
    print("Spectral coverage analysis complete")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Creates a 3D surface plot showing the spectral reconstruction as a function of 
frequency and component count.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import librosa
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf
from scipy import signal

def create_3d_spectral_surface(original_file, reconstructed_files, component_counts, output_dir, time_slice=None):
    """
    Create a 3D surface plot showing spectral reconstruction across component counts.
    
    Args:
        original_file: Path to the original audio file
        reconstructed_files: List of paths to reconstructed audio files
        component_counts: List of component counts corresponding to each reconstructed file
        output_dir: Directory to save output visualizations
        time_slice: Optional time slice (in seconds) to use for visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original audio
    y_orig, sr = librosa.load(original_file, sr=None, mono=True)
    
    # If time slice is specified, extract portion of the audio
    if time_slice is not None:
        if isinstance(time_slice, (list, tuple)) and len(time_slice) == 2:
            start_sample = int(time_slice[0] * sr)
            end_sample = int(time_slice[1] * sr)
            y_orig = y_orig[start_sample:end_sample]
        elif isinstance(time_slice, (int, float)):
            # Use a single slice at specific time point
            center_sample = int(time_slice * sr)
            # Use 100ms window around the time point
            window_size = int(0.1 * sr)
            start_sample = max(0, center_sample - window_size//2)
            end_sample = min(len(y_orig), center_sample + window_size//2)
            y_orig = y_orig[start_sample:end_sample]
    
    # Compute magnitude spectrum of original
    n_fft = 2048
    S_orig = np.abs(librosa.stft(y_orig, n_fft=n_fft))
    
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # If audio is too long, take the average spectrum
    if S_orig.shape[1] > 1:
        S_orig = np.mean(S_orig, axis=1)
    else:
        S_orig = S_orig.flatten()
    
    # Apply log scaling to better visualize low amplitudes
    S_orig_db = librosa.amplitude_to_db(S_orig, ref=np.max)
    
    # Initialize arrays to hold reconstructed spectra
    recon_spectra = []
    
    # Process each reconstructed file
    for recon_file in reconstructed_files:
        y_recon, sr_recon = librosa.load(recon_file, sr=sr, mono=True)
        
        # Apply the same time slicing
        if time_slice is not None:
            if isinstance(time_slice, (list, tuple)) and len(time_slice) == 2:
                start_sample = int(time_slice[0] * sr)
                end_sample = int(time_slice[1] * sr)
                y_recon = y_recon[start_sample:end_sample]
            elif isinstance(time_slice, (int, float)):
                center_sample = int(time_slice * sr)
                window_size = int(0.1 * sr)
                start_sample = max(0, center_sample - window_size//2)
                end_sample = min(len(y_recon), center_sample + window_size//2)
                y_recon = y_recon[start_sample:end_sample]
        
        # Compute magnitude spectrum
        S_recon = np.abs(librosa.stft(y_recon, n_fft=n_fft))
        
        # Take average if needed
        if S_recon.shape[1] > 1:
            S_recon = np.mean(S_recon, axis=1)
        else:
            S_recon = S_recon.flatten()
        
        # Convert to dB scale
        S_recon_db = librosa.amplitude_to_db(S_recon, ref=np.max)
        
        recon_spectra.append(S_recon_db)
    
    # Sort the reconstructed spectra by component count
    sorted_indices = np.argsort(component_counts)
    sorted_spectra = [recon_spectra[i] for i in sorted_indices]
    sorted_components = [component_counts[i] for i in sorted_indices]
    
    # Create a matrix for the 3D surface
    # X: frequency bins, Y: component counts, Z: magnitude
    X = np.tile(freqs, (len(sorted_components), 1))
    Y = np.repeat(sorted_components, len(freqs)).reshape(len(sorted_components), len(freqs))
    Z = np.vstack(sorted_spectra)
    
    # Create an improved 3D surface plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a custom colormap with good color differentiation
    cmap = cm.viridis
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cmap(norm(Z))
    
    # Create the surface plot - use alpha to see through some surfaces
    surf = ax.plot_surface(
        X, Y, Z, 
        facecolors=colors, 
        linewidth=0,
        antialiased=True, 
        shade=True,
        alpha=0.9
    )
    
    # Add color bar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1)
    cbar.set_label('Magnitude (dB)', fontsize=12)
    
    # Set log scale for frequency axis
    ax.set_xscale('log')
    
    # Set nice labels and title
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Component Count', fontsize=14)
    ax.set_zlabel('Magnitude (dB)', fontsize=14)
    ax.set_title('Spectral Reconstruction vs. Component Count', fontsize=16)
    
    # Use log ticks for the frequency axis
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Rotate the view for better visibility
    ax.view_init(elev=30, azim=-35)
    
    # Show grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3d_spectral_surface.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D spectral surface plot to {output_path}")
    
    # Create a second view from a different angle
    ax.view_init(elev=15, azim=-110)
    alt_output_path = os.path.join(output_dir, '3d_spectral_surface_alt_view.png')
    plt.savefig(alt_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved alternative view to {alt_output_path}")
    
    # Create a 2D visualization as well - a "waterfall" plot
    plt.figure(figsize=(12, 8))
    
    # Normalize all spectra to improve visualization
    normalized_spectra = []
    for spectrum in sorted_spectra:
        normalized_spectra.append(spectrum - np.min(spectrum))
    
    # Create waterfall plot
    for i, (spectrum, comp_count) in enumerate(zip(normalized_spectra, sorted_components)):
        offset = i * 15  # Offset each spectrum for the waterfall effect
        plt.plot(freqs, spectrum + offset, linewidth=1.5, label=f"{comp_count} Components")
        
        # Fill the area under the curve
        plt.fill_between(freqs, offset, spectrum + offset, alpha=0.2)
    
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Magnitude (dB) + Offset', fontsize=14)
    plt.title('Spectral Reconstruction Waterfall Plot', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Set x-axis limits to focus on the audible frequency range
    plt.xlim(20, sr/2)
    
    # Use log ticks for the frequency axis
    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    
    # Save the waterfall plot
    waterfall_output_path = os.path.join(output_dir, 'spectral_waterfall.png')
    plt.savefig(waterfall_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved waterfall plot to {waterfall_output_path}")
    
    # Create a third visualization: highlight key areas where components make a difference
    # Analyze which frequency bands see the most improvement with added components
    improvement_matrix = np.zeros((len(sorted_components)-1, len(freqs)))
    
    for i in range(len(sorted_components)-1):
        # Calculate improvement from adding more components
        improvement_matrix[i] = sorted_spectra[i+1] - sorted_spectra[i]
    
    # Plot the improvement heatmap
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(freqs, sorted_components[1:], improvement_matrix, cmap='coolwarm', shading='auto')
    plt.colorbar(label='Magnitude Improvement (dB)')
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Component Count (After Improvement)', fontsize=14)
    plt.title('Spectral Improvement with Additional Components', fontsize=16)
    plt.xlim(20, sr/2)
    
    # Use log ticks for the frequency axis
    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Save the improvement heatmap
    improvement_output_path = os.path.join(output_dir, 'spectral_improvement.png')
    plt.savefig(improvement_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved improvement heatmap to {improvement_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create 3D spectral surface plot')
    parser.add_argument('--original', type=str, required=True, help='Path to original audio file')
    parser.add_argument('--reconstructed', type=str, nargs='+', required=True, 
                        help='Paths to reconstructed audio files')
    parser.add_argument('--components', type=int, nargs='+', required=True,
                        help='Component counts for each reconstructed file')
    parser.add_argument('--output_dir', type=str, default='spectral_surface_plots',
                        help='Directory to save output plots')
    parser.add_argument('--time_slice', type=float, nargs='+', 
                        help='Time slice (in seconds) to use for visualization. Can be a single value or start,end range')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.reconstructed) != len(args.components):
        parser.error("Number of reconstructed files must match number of component counts")
    
    time_slice = None
    if args.time_slice:
        if len(args.time_slice) == 1:
            time_slice = args.time_slice[0]
        elif len(args.time_slice) == 2:
            time_slice = (args.time_slice[0], args.time_slice[1])
        else:
            parser.error("Time slice must be a single value or start,end range")
    
    create_3d_spectral_surface(
        args.original,
        args.reconstructed,
        args.components,
        args.output_dir,
        time_slice
    )

if __name__ == "__main__":
    main()
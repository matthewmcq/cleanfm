#!/usr/bin/env python3
"""
Visualization script for audio processing accuracy metrics with debug enhancements.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import sys

def debug_print_csv_info(metrics_files):
    """Print debug information about CSV files"""
    print("\n===== CSV FILES DEBUG INFO =====")
    for length, filepath in metrics_files.items():
        try:
            print(f"\nFile: {filepath}")
            df = pd.read_csv(filepath)
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape}")
            print(f"First 3 rows:")
            print(df.head(3))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    print("================================\n")

def plot_metrics_grid(metrics_files, output_dir, debug=False):
    """
    Create a grid visualizing metrics across different sample lengths.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print debug info if requested
    if debug:
        debug_print_csv_info(metrics_files)
    
    # Sample lengths and metrics to plot
    sample_lengths = sorted(metrics_files.keys())
    
    # Check which metrics are available in all files
    available_metrics = set()
    first_file = True
    
    for length, filepath in metrics_files.items():
        try:
            df = pd.read_csv(filepath)
            if first_file:
                available_metrics = set(df.columns)
                first_file = False
            else:
                available_metrics &= set(df.columns)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # Remove non-metric columns
    for col in ['batch', 'components']:
        if col in available_metrics:
            available_metrics.remove(col)
    
    # Filter only our primary metrics (not naive ones)
    primary_metrics = [m for m in available_metrics if not m.startswith('naive_')]
    
    print(f"Available metrics across all files: {primary_metrics}")
    
    if not primary_metrics:
        print("Error: No common metrics found across files")
        return
    
    # Now we know exactly which metrics we can plot
    metrics = primary_metrics
    metric_titles = {
        'lsd': 'Log Spectral Distance (dB)',
        'rmse': 'Root Mean Squared Error',
        'cosine_sim': 'Cosine Similarity'
    }
    
    # Determine grid dimensions based on available metrics
    num_metrics = len(metrics)
    num_lengths = len(sample_lengths)
    
    print(f"Creating {num_lengths}x{num_metrics} grid for {num_lengths} sample lengths and {num_metrics} metrics")
    
    # Create figure with appropriate dimensions
    fig = plt.figure(figsize=(4 * num_metrics, 5 * num_lengths))
    gs = GridSpec(num_lengths, num_metrics, figure=fig)
    
    # Plot each sample length and metric
    for i, length in enumerate(sample_lengths):
        for j, metric in enumerate(metrics):
            print(f"Plotting {metric} for {length}s audio")
            
            
            ax = fig.add_subplot(gs[i, j])
            
            # Load metrics data for this sample length
            try:
                df = pd.read_csv(metrics_files[length])
                
                # Check for the metric column
                if metric not in df.columns:
                    ax.text(0.5, 0.5, f"Metric '{metric}'\nnot found in data", 
                            ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot our algorithm's performance
                ax.plot(df['components'], df[metric], 
                        color='blue', linewidth=2, marker='o', markersize=4,
                        label='Dirichlet Kernel Deconvolution')
                
                # Plot the naive baseline
                naive_metric = f'naive_{metric}'
                if naive_metric in df.columns:
                    ax.plot(df['components'], df[naive_metric], 
                            color='black', linestyle='--', linewidth=1.5,
                            label='Naive FFT Selection')
                
                # Set titles and labels
                if i == 0:
                    title = metric_titles.get(metric, metric.capitalize())
                    ax.set_title(title, fontsize=12)
                
                if j == 0:
                    ax.set_ylabel(f'{length}s Audio\nMetric Value', fontsize=12)
                else:
                    ax.set_ylabel('Metric Value', fontsize=10)
                
                if i == len(sample_lengths) - 1:
                    ax.set_xlabel('Number of Components', fontsize=10)
                
                # Add legend for the first plot only
                if i == 0 and j == 0:
                    ax.legend(loc='best', fontsize=10)
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3)
                
                # Customize Y-axis scale based on metric
                if metric == 'lsd':
                    ax.set_yscale('log')
                    # ax.set_ylim(bottom=0.1)
                elif metric == 'rmse':
                    ax.set_yscale('log')
                    # ax.set_ylim(bottom=1e-6)
                # elif metric == 'cosine_sim':
                    # ax.set_ylim(0, 1.05)  # Cosine similarity is between 0 and 1
                
            except Exception as e:
                print(f"Error plotting {metric} for {length}s: {e}")
                ax.text(0.5, 0.5, f"Error:\n{str(e)}", 
                        ha='center', va='center', transform=ax.transAxes)
    
    # Add a single main title
    fig.suptitle('Accuracy Metrics Across Audio Lengths and Component Counts', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path = os.path.join(output_dir, 'accuracy_metrics_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Generated figure: {output_path}")
    
    # Create individual plots for each metric
    for metric in metrics:
        try:
            plt.figure(figsize=(10, 8))
            
            for i, length in enumerate(sample_lengths):
                try:
                    df = pd.read_csv(metrics_files[length])
                    
                    if metric not in df.columns:
                        continue
                    
                    # Plot our algorithm's performance
                    plt.plot(df['components'], df[metric], 
                            marker='o', markersize=3, linewidth=2,
                            label=f'{length}s - DKD')
                    
                    # Plot the naive baseline
                    naive_metric = f'naive_{metric}'
                    if naive_metric in df.columns:
                        plt.plot(df['components'], df[naive_metric], 
                                linestyle='--', linewidth=1,
                                label=f'{length}s - Naive')
                    
                except Exception as e:
                    print(f"Error processing {length}s for {metric}: {e}")
            
            title = metric_titles.get(metric, metric.capitalize())
            plt.title(f'{title} vs. Component Count', fontsize=14)
            plt.xlabel('Number of Components', fontsize=12)
            plt.ylabel(title, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            if metric == 'lsd' or metric == 'rmse':
                plt.yscale('log')
            # elif metric == 'cosine_sim':
            #     plt.ylim(0, 1.05)
            
            output_path = os.path.join(output_dir, f'{metric}_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Generated figure: {output_path}")
        except Exception as e:
            print(f"Error creating individual plot for {metric}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize audio processing accuracy metrics')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                      help='Directory to save visualization results')
    parser.add_argument('--metrics_files', type=str, nargs='+', required=True,
                      help='List of metrics CSV files in format: length:file.csv')
    parser.add_argument('--debug', action='store_true',
                      help='Print debugging information')
    
    args = parser.parse_args()
    
    # Parse metrics_files argument
    metrics_files = {}
    for file_spec in args.metrics_files:
        try:
            length, filepath = file_spec.split(':')
            metrics_files[int(length)] = filepath
        except ValueError:
            print(f"Error parsing file spec: {file_spec}")
            print("Format should be 'length:filepath'")
            continue
    
    if not metrics_files:
        print("No valid metrics files specified")
        sys.exit(1)
    
    # Create visualizations
    plot_metrics_grid(metrics_files, args.output_dir, args.debug)

if __name__ == '__main__':
    main()
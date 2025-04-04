#!/usr/bin/env python3
"""
Detailed analysis of sample rate benchmark results for thesis.
Creates advanced visualizations and LaTeX tables.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

def load_metrics_data(metrics_dir, sample_rates):
    """Load metrics data for all sample rates"""
    data = {}
    for sr in sample_rates:
        file_path = os.path.join(metrics_dir, f'metrics_{sr}Hz.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                data[sr] = df
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return data

def create_component_efficiency_plot(data, output_dir):
    """Create plot showing component efficiency across sample rates"""
    plt.figure(figsize=(12, 8))
    
    # Calculate information per component
    for sr, df in data.items():
        if sr != 48000:  # Skip 48kHz for now
            continue
        if 'components' in df.columns and 'shannon_info' and 'naive_shannon_info' in df.columns:
            
            # Calculate information per component (efficiency)
            efficiency = df['shannon_info'] / df['components']
            
            naive_efficiency = df['naive_shannon_info'] / df['components']
            
            plt.plot(df['components'], efficiency, 
                    marker='o', markersize=3, linewidth=2,
                    label=f'{sr}Hz')
            
            
            
            ax = plt.gca()
            ax.set_xscale('log')
    
    plt.title('Information Efficiency vs. Component Count', fontsize=14)
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Shannon Information per Component', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    output_path = os.path.join(output_dir, 'component_efficiency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")

def create_frequency_bandwidth_plot(data, sample_rates, output_dir):
    """Create plot showing how bandwidth affects component count"""
    bandwidths = [sr/2 for sr in sample_rates]  # Nyquist frequencies
    
    # Get component counts at fixed error thresholds
    error_thresholds = [0.01, 0.001, 0.0001]
    comp_counts = {threshold: [] for threshold in error_thresholds}
    
    for sr in sample_rates:
        if sr in data:
            df = data[sr]
            for threshold in error_thresholds:
                # Find component count where MSE drops below threshold
                if 'mse' in df.columns:
                    below_threshold = df[df['mse'] < threshold]
                    if not below_threshold.empty:
                        comp_counts[threshold].append((sr, below_threshold['components'].iloc[0]))
                    else:
                        # If never drops below threshold, use max components
                        comp_counts[threshold].append((sr, df['components'].max()))
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    for threshold, points in comp_counts.items():
        if points:
            x = [p[0] for p in points]  # Sample rates
            y = [p[1] for p in points]  # Component counts
            
            plt.plot(x, y, marker='o', markersize=5, linewidth=2,
                    label=f'MSE < {threshold}')
    
    plt.title('Required Components vs. Sample Rate', fontsize=14)
    plt.xlabel('Sample Rate (Hz)', fontsize=12)
    plt.ylabel('Number of Components Required', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.xticks(sample_rates)
    
    # Add second x-axis for bandwidth
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(sample_rates)
    ax2.set_xticklabels([f"{sr/2/1000:.1f}kHz" for sr in sample_rates])
    ax2.set_xlabel("Bandwidth (kHz)", fontsize=12)
    
    output_path = os.path.join(output_dir, 'required_components.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")

def create_heatmap(data, sample_rates, output_dir):
    """Create heatmap showing relationship between components, sample rate, and error"""
    component_counts = [1024, 2048, 4096, 8192, 16384, 32768]
    
    # Extract MSE values for each sample rate and component count
    mse_matrix = np.zeros((len(sample_rates), len(component_counts)))
    mse_matrix.fill(np.nan)  # Fill with NaN initially
    
    for i, sr in enumerate(sample_rates):
        if sr in data:
            df = data[sr]
            for j, comp_count in enumerate(component_counts):
                if 'components' in df.columns and 'mse' in df.columns:
                    # Find closest component count
                    closest_idx = (df['components'] - comp_count).abs().idxmin()
                    actual_comp = df.loc[closest_idx, 'components']
                    
                    # Only use if within 10% of target
                    if abs(actual_comp - comp_count) / comp_count < 0.1:
                        mse_matrix[i, j] = df.loc[closest_idx, 'mse']
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    
    # Use log scale for MSE values
    log_mse = np.log10(mse_matrix)
    mask = np.isnan(log_mse)
    
    # Create heatmap with masked values
    ax = sns.heatmap(log_mse, annot=True, cmap='viridis', mask=mask, 
                    xticklabels=component_counts, 
                    yticklabels=sample_rates,
                    cbar_kws={'label': 'log10(MSE)'})
    
    plt.title('MSE Values Across Sample Rates and Component Counts', fontsize=14)
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Sample Rate (Hz)', fontsize=12)
    
    output_path = os.path.join(output_dir, 'mse_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")

def create_latex_tables(data, sample_rates, output_dir):
    """Create LaTeX tables for thesis"""
    component_counts = [1024, 2048, 4096, 8192, 16384, 32768]
    metrics = ['mse', 'lsd', 'cosine_sim', 'shannon_info']
    metric_names = {
        'mse': 'MSE',
        'lsd': 'LSD (dB)',
        'cosine_sim': 'Cosine Sim.',
        'shannon_info': 'Shannon Info.'
    }
    
    # Create a table for each metric
    for metric in metrics:
        with open(os.path.join(output_dir, f'{metric}_table.tex'), 'w') as f:
            f.write('\\begin{table}[htbp]\n')
            f.write('\\centering\n')
            f.write(f'\\caption{{{metric_names.get(metric, metric)} across Sample Rates and Component Counts}}\n')
            f.write('\\begin{tabular}{|l|')
            for _ in component_counts:
                f.write('c|')
            f.write('}\n')
            f.write('\\hline\n')
            
            # Header row
            f.write('Sample Rate (Hz) & ')
            f.write(' & '.join([f"{comp} Comp." for comp in component_counts]))
            f.write(' \\\\ \\hline\n')
            
            # Data rows
            for sr in sample_rates:
                row = f"{sr} & "
                values = []
                
                if sr in data:
                    df = data[sr]
                    for comp_count in component_counts:
                        if 'components' in df.columns and metric in df.columns:
                            # Find closest component count
                            closest_idx = (df['components'] - comp_count).abs().idxmin()
                            actual_comp = df.loc[closest_idx, 'components']
                            
                            # Only use if within 10% of target
                            if abs(actual_comp - comp_count) / comp_count < 0.1:
                                if metric == 'mse':
                                    # Scientific notation for MSE
                                    values.append(f"{df.loc[closest_idx, metric]:.2e}")
                                else:
                                    values.append(f"{df.loc[closest_idx, metric]:.4f}")
                            else:
                                values.append("--")
                        else:
                            values.append("--")
                else:
                    values = ["--"] * len(component_counts)
                
                row += " & ".join(values) + " \\\\ \\hline\n"
                f.write(row)
            
            f.write('\\end{tabular}\n')
            f.write('\\end{table}\n')
        
        print(f"Saved LaTeX table to {os.path.join(output_dir, f'{metric}_table.tex')}")

def main():
    parser = argparse.ArgumentParser(description='Analyze sample rate benchmark results')
    parser.add_argument('--metrics_dir', type=str, required=True,
                      help='Directory containing metrics CSV files')
    parser.add_argument('--output_dir', type=str, default='sample_rate_analysis',
                      help='Directory for output files')
    parser.add_argument('--sample_rates', type=int, nargs='+', 
                      default=[8000, 16000, 22050, 32000, 44100, 48000],
                      help='Sample rates tested')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_metrics_data(args.metrics_dir, args.sample_rates)
    
    if not data:
        print("No data found, exiting")
        return
    
    # Create visualizations
    create_component_efficiency_plot(data, args.output_dir)
    create_frequency_bandwidth_plot(data, args.sample_rates, args.output_dir)
    create_heatmap(data, args.sample_rates, args.output_dir)
    
    # Create LaTeX tables
    create_latex_tables(data, args.sample_rates, args.output_dir)
    
    print("Analysis complete. Files saved to", args.output_dir)

if __name__ == '__main__':
    main()
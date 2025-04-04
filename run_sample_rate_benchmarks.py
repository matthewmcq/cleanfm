#!/usr/bin/env python3
"""
Script to benchmark Dirichlet kernel deconvolution across different sample rates.
Takes an original audio file, resamples it to different rates, and runs the deconvolution.
"""

import os
import subprocess
import argparse
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def resample_audio(input_file, output_file, target_sr):
    """Resample audio to the specified sample rate"""
    try:
        # Load audio with librosa (automatically resamples)
        y, orig_sr = librosa.load(input_file, sr=None)
        
        print(f"Loaded audio: {len(y)} samples at {orig_sr}Hz")
        
        # Resample to target sample rate
        y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        
        print(f"Resampled to {target_sr}Hz: {len(y_resampled)} samples")
        
        # Save resampled audio
        sf.write(output_file, y_resampled, target_sr)
        print(f"Saved resampled audio to {output_file}")
        
        return True
    except Exception as e:
        print(f"Error resampling audio: {e}")
        return False

def run_deconvolution(executable, input_file, threads, metrics_file):
    """Run Dirichlet kernel deconvolution on the input file"""
    cmd = [executable, "-f", input_file, "-t", str(threads), "-m", metrics_file]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running deconvolution: {e}")
        print(f"stderr: {e.stderr}")
        return False

def plot_metrics_vs_sample_rate(metrics_files, sample_rates, output_dir):
    """Create plots comparing metrics across different sample rates with component count on x-axis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics data for each sample rate
    dfs = {}
    for sr, file_path in zip(sample_rates, metrics_files):
        try:
            df = pd.read_csv(file_path)
            dfs[sr] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not dfs:
        print("No metrics data found")
        return
    
    # Determine common metrics across all files
    metrics = set()
    for df in dfs.values():
        if metrics:
            metrics &= set(df.columns)
        else:
            metrics = set(df.columns)
    
    # Remove non-metric columns
    for col in ['batch', 'components']:
        if col in metrics:
            metrics.remove(col)
    
    # Filter out naive metrics
    main_metrics = [m for m in metrics if not m.startswith('naive_')]
    
    print(f"Common metrics: {main_metrics}")
    
    # Metric titles
    metric_titles = {
        'lsd': 'Log Spectral Distance (dB)',
        'mse': 'Mean Squared Error',
        'shannon_info': 'Shannon Information (bits)',
        'cosine_sim': 'Cosine Similarity'
    }
    
    # For each metric, plot vs component count with different sample rates as lines
    for metric in main_metrics:
        plt.figure(figsize=(12, 8))
        
        for sr, df in dfs.items():
            if 'components' in df.columns and metric in df.columns:
                plt.plot(df['components'], df[metric], 
                        marker='o', markersize=4, linewidth=2,
                        label=f'{sr} Hz')
        
        title = metric_titles.get(metric, metric.capitalize())
        plt.title(f'{title} vs. Component Count', fontsize=16)
        plt.xlabel('Number of Components', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        
        if metric == 'lsd' or metric == 'mse':
            plt.yscale('log')
        elif metric == 'cosine_sim' or metric == 'shannon_info':
            plt.xscale('log')
        
        # Add X and Y grid lines
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Add annotations for key insights (optional)
        if metric == 'mse':
            plt.annotate('Higher sample rates require\nmore components for same error',
                       xy=(1500, dfs[sample_rates[2]]['mse'].iloc[10]),
                       xytext=(1800, dfs[sample_rates[2]]['mse'].iloc[10] * 5),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                       fontsize=12)
        
        output_path = os.path.join(output_dir, f'sample_rate_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    
    # Combined plot for MSE, LSD, and cosine_sim
    if all(m in main_metrics for m in ['mse', 'lsd', 'cosine_sim']):
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        metrics_to_plot = ['mse', 'lsd', 'cosine_sim']
        
        for i, metric in enumerate(metrics_to_plot):
            for sr, df in dfs.items():
                if 'components' in df.columns and metric in df.columns:
                    axs[i].plot(df['components'], df[metric], 
                              marker='o', markersize=3, linewidth=2,
                              label=f'{sr} Hz')
            
            title = metric_titles.get(metric, metric.capitalize())
            axs[i].set_title(title, fontsize=14)
            axs[i].set_ylabel(title, fontsize=12)
            axs[i].grid(True, alpha=0.3)
            
            if metric in ['mse', 'lsd']:
                axs[i].set_yscale('log')
            
                
            # elif metric == 'cosine_sim':
            #     axs[i].set_ylim(0, 1.05)
            
            # Add legend to first subplot only
            if i == 0:
                axs[i].legend(loc='best', fontsize=10)
        
        # Add common X axis label
        fig.text(0.5, 0.04, 'Number of Components', ha='center', fontsize=14)
        
        # Add overall title
        fig.suptitle('Error Metrics vs. Component Count for Different Sample Rates', 
                    fontsize=18, y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        output_path = os.path.join(output_dir, 'combined_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    
    # Create summary tables
    component_thresholds = [1024, 2048, 4096, 8192, 16384, 32768]
    
    # Summary table 1: For each sample rate and component threshold, show metrics
    summary_data = []
    
    for sr in sample_rates:
        if sr in dfs:
            df = dfs[sr]
            
            for comp_threshold in component_thresholds:
                if 'components' in df.columns and len(df) > 0:
                    row = {'Sample Rate': sr, 'Components': comp_threshold}
                    
                    # Find row with closest component count
                    closest_idx = (df['components'] - comp_threshold).abs().idxmin()
                    actual_components = df.loc[closest_idx, 'components']
                    
                    # Only use if within 10% of target
                    if abs(actual_components - comp_threshold) / comp_threshold < 0.1:
                        for metric in main_metrics:
                            if metric in df.columns:
                                row[metric] = df.loc[closest_idx, metric]
                        
                        summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'metrics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary table to {summary_path}")
        
        # Also create a LaTeX table
        latex_path = os.path.join(output_dir, 'metrics_latex_table.tex')
        
        with open(latex_path, 'w') as f:
            f.write('\\begin{table}[htbp]\n')
            f.write('\\centering\n')
            f.write('\\caption{Error Metrics Across Sample Rates and Component Counts}\n')
            f.write('\\begin{tabular}{|c|c|c|c|c|}\n')
            f.write('\\hline\n')
            f.write('Sample Rate (Hz) & Components & MSE & LSD (dB) & Cosine Sim. \\\\ \\hline\n')
            
            for _, row in summary_df.iterrows():
                sr = int(row['Sample Rate'])
                comp = int(row['Components'])
                
                # Format metrics
                mse = f"{row['mse']:.2e}" if 'mse' in row else "--"
                lsd = f"{row['lsd']:.4f}" if 'lsd' in row else "--"
                cosine = f"{row['cosine_sim']:.4f}" if 'cosine_sim' in row else "--"
                
                f.write(f"{sr} & {comp} & {mse} & {lsd} & {cosine} \\\\ \\hline\n")
            
            f.write('\\end{tabular}\n')
            f.write('\\end{table}\n')
            
        print(f"Saved LaTeX table to {latex_path}")

def main():
    parser = argparse.ArgumentParser(description='Run Dirichlet kernel deconvolution at different sample rates')
    parser.add_argument('--input', type=str, required=True,
                      help='Input audio file (original sample rate)')
    parser.add_argument('--executable', type=str, required=True,
                      help='Path to cleanfm executable')
    parser.add_argument('--threads', type=int, default=12,
                      help='Number of threads to use')
    parser.add_argument('--output_dir', type=str, default='sample_rate_results',
                      help='Directory for output files')
    parser.add_argument('--sample_rates', type=int, nargs='+', 
                      default=[8000, 11025, 16000, 22050, 32000, 44100, 48000],
                      help='Sample rates to test')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Resample audio to different sample rates
    resampled_files = []
    for sr in args.sample_rates:
        output_file = os.path.join(args.output_dir, 'audio', f'resampled_{sr}Hz.wav')
        if resample_audio(args.input, output_file, sr):
            resampled_files.append((sr, output_file))
    
    if not resampled_files:
        print("No resampled files created, exiting")
        return
    
    # Run deconvolution on each resampled file
    metrics_files = []
    for sr, audio_file in resampled_files:
        metrics_file = os.path.join(args.output_dir, 'metrics', f'metrics_{sr}Hz.csv')
        if run_deconvolution(args.executable, audio_file, args.threads, metrics_file):
            metrics_files.append(metrics_file)
    
    if not metrics_files:
        print("No metrics files created, exiting")
        return
    
    # Create visualizations
    plot_metrics_vs_sample_rate(
        metrics_files, 
        [sr for sr, _ in resampled_files], 
        os.path.join(args.output_dir, 'plots')
    )
    
    print("Sample rate benchmark complete")

if __name__ == '__main__':
    main()
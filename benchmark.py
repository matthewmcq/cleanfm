import os
import subprocess
import csv
import time
import argparse
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys
import platform

# Configuration
THREAD_COUNTS = [1, 2, 4, 8, 12, 16]
DURATIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
DEFAULT_TRIALS = 5
EXECUTABLE = "./build/cleanfm"  
FILE_PATTERN = "examples/test_runtime/TEST_{duration}s.wav"
RESULTS_DIR = "benchmark_results"

def parse_benchmark_output(output):
    """Parse the benchmark output from the C++ program."""
    results = {}
    for line in output.strip().split('\n'):
        if ',' in line:
            key, value = line.split(',', 1)
            results[key] = value
    return results

def run_benchmark(duration, threads, trials=DEFAULT_TRIALS, dry_run=False, verbose=False):
    """Run benchmark with specified parameters for multiple trials."""
    input_file = FILE_PATTERN.format(duration=duration)
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Warning: File {input_file} does not exist, skipping.")
        return None
    
    all_results = []
    
    for trial in range(1, trials + 1):
        print(f"Running trial {trial}/{trials} for {duration}s audio, {threads} threads...")
        
        # Construct command
        cmd = [EXECUTABLE, "-f", input_file, "-t", str(threads), "-b"]
        cmd_str = ' '.join(cmd)
        
        if verbose:
            print(f"Executing: {cmd_str}")
        
        if dry_run:
            print(f"[DRY RUN] Would execute: {cmd_str}")
            # Simulate some results for dry run
            result = {
                'INPUT_FILE': input_file,
                'THREADS': str(threads),
                'PARALLEL': '1' if threads > 1 else '0',
                'WALL_TIME': str(duration * 0.1 * (1 if threads == 1 else 1 / min(threads, 8))),
                'CPU_TIME': str(duration * 0.1 * min(threads, 8)),
                'USER_TIME': str(duration * 0.08 * (1 if threads == 1 else 1 / min(threads, 8))),
                'SYSTEM_TIME': str(duration * 0.02 * (1 if threads == 1 else 1 / min(threads, 8))),
                'CLOCK_TIME': str(duration * 0.09 * (1 if threads == 1 else 1 / min(threads, 8))),
                'TOTAL_THREAD_TIME': str(duration * 0.1 * threads),
                'NUM_PEAKS': str(int(duration * 100)),
                'NUM_COMPONENTS': str(int(duration * 80)),
                'SAMPLE_RATE': '48000',
                'DURATION': str(duration)
            }
        else:
            # Execute the benchmark
            try:
                process = subprocess.run(cmd, check=True, capture_output=True, text=True)
                result = parse_benchmark_output(process.stdout)
                
                if verbose:
                    print(f"Command output:\n{process.stdout}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error running benchmark: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                continue
        
        # Add metadata
        result['DURATION_SEC'] = duration
        result['TRIAL'] = trial
        result['TIMESTAMP'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['SYSTEM'] = platform.platform()
        result['CPU'] = platform.processor()
        
        all_results.append(result)
        
        # Allow system to cool down between trials
        if not dry_run and trial < trials:
            cooling_time = 0.25
            # print(f"Cooling down for {cooling_time:.1f} seconds...")
            time.sleep(cooling_time)
    
    return all_results

def save_results(all_results, filename):
    """Save benchmark results to CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get all possible fields from results
    fields = set()
    for result_set in all_results:
        for result in result_set:
            fields.update(result.keys())
    
    fields = sorted(list(fields))
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for result_set in all_results:
            for result in result_set:
                writer.writerow(result)
    
    print(f"Results saved to {filename}")

def generate_report(csv_file):
    """Generate a report with statistics and graphs from benchmark results."""
    print("Generating benchmark report...")
    
    # Load results
    df = pd.read_csv(csv_file)
    
    # Convert string columns to numeric where possible
    numeric_columns = ['THREADS', 'WALL_TIME', 'CPU_TIME', 'USER_TIME', 'SYSTEM_TIME', 
                       'CLOCK_TIME', 'TOTAL_THREAD_TIME', 'NUM_PEAKS', 'NUM_COMPONENTS', 
                       'SAMPLE_RATE', 'DURATION', 'DURATION_SEC', 'TRIAL']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create output directory
    report_dir = os.path.join(RESULTS_DIR, 'report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Summary statistics
    summary_metrics = ['WALL_TIME', 'CPU_TIME', 'USER_TIME', 'SYSTEM_TIME', 'CLOCK_TIME', 'TOTAL_THREAD_TIME']
    
    summary = df.groupby(['DURATION_SEC', 'THREADS']).agg({
        metric: ['mean', 'std', 'min', 'max'] for metric in summary_metrics
    }).reset_index()
    
    summary.to_csv(os.path.join(report_dir, 'summary_stats.csv'))
    
    # Calculate speedup and efficiency
    baseline = df[df['THREADS'] == 1].groupby('DURATION_SEC')['USER_TIME'].mean().reset_index()
    baseline.columns = ['DURATION_SEC', 'BASELINE_TIME']
    
    # Merge with main dataframe
    merged = pd.merge(df, baseline, on='DURATION_SEC')
    merged['SPEEDUP'] = merged['BASELINE_TIME'] / merged['USER_TIME']
    merged['EFFICIENCY'] = merged['SPEEDUP'] / merged['THREADS'] * 100
    
    if merged.empty:
        max_threads = max(df['THREADS'])
    else:
        max_threads = max(merged['THREADS'])
    
    # Average speedup and efficiency by threads and duration
    speedup_stats = merged.groupby(['DURATION_SEC', 'THREADS']).agg({
        'SPEEDUP': ['mean', 'std'],
        'EFFICIENCY': ['mean', 'std']
    }).reset_index()
    
    
    
    speedup_stats.to_csv(os.path.join(report_dir, 'speedup_stats.csv'))
    
    # Create visualization plots
    
    # 1. User Time vs Duration for different thread counts
    plt.figure(figsize=(12, 8))
    
    for threads in sorted(df['THREADS'].unique()):
        subset = df[df['THREADS'] == threads]
        means = subset.groupby('DURATION_SEC')['USER_TIME'].mean()
        std_devs = subset.groupby('DURATION_SEC')['USER_TIME'].std()
        plt.errorbar(means.index, means.values, yerr=std_devs.values, 
                    label=f'{threads} threads', marker='o')
    
    plt.xlabel('Audio Duration (seconds)')
    plt.ylabel('User Time (seconds)')
    plt.title('User Processing Time vs. Duration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'user_time_vs_duration.png'), dpi=300)
    
    # 2. Wall time vs Duration for different thread counts
    plt.figure(figsize=(12, 8))
    
    for threads in sorted(df['THREADS'].unique()):
        subset = df[df['THREADS'] == threads]
        means = subset.groupby('DURATION_SEC')['WALL_TIME'].mean()
        std_devs = subset.groupby('DURATION_SEC')['WALL_TIME'].std()
        plt.errorbar(means.index, means.values, yerr=std_devs.values, 
                    label=f'{threads} threads', marker='o')
    
    plt.xlabel('Audio Duration (seconds)')
    plt.ylabel('Wall Time (seconds)')
    plt.title('Audio Processing Time vs. Duration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'wall_time_vs_duration.png'), dpi=300)
    
    
    
    if not merged.empty:
        plt.figure(figsize=(12, 8))
        
        for duration in sorted(merged['DURATION_SEC'].unique()):
            subset = merged[merged['DURATION_SEC'] == duration]
            means = subset.groupby('THREADS')['SPEEDUP'].mean()
            std_devs = subset.groupby('THREADS')['SPEEDUP'].std()
            plt.errorbar(means.index, means.values, yerr=std_devs.values, 
                        label=f'{duration}s audio', marker='o')
        
        # Ideal speedup line
        max_threads = max(merged['THREADS'])
        plt.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Speedup')
        
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup')
        plt.title('Speedup vs. Thread Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(report_dir, 'speedup_vs_threads.png'), dpi=300)
    else:
        print("Warning: Cannot create speedup plot - no baseline data available")
    
    # 4. Efficiency vs. Thread Count
    plt.figure(figsize=(12, 8))
    
    for duration in sorted(merged['DURATION_SEC'].unique()):
        subset = merged[merged['DURATION_SEC'] == duration]
        means = subset.groupby('THREADS')['EFFICIENCY'].mean()
        std_devs = subset.groupby('THREADS')['EFFICIENCY'].std()
        plt.errorbar(means.index, means.values, yerr=std_devs.values, 
                    label=f'{duration}s audio', marker='o')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency (%)')
    plt.title('Parallel Efficiency vs. Thread Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'efficiency_vs_threads.png'), dpi=300)
    
    # 5. Time Measurement Comparison (USER vs CPU vs WALL)
    plt.figure(figsize=(15, 10))
    
    # Select a representative duration
    rep_duration = 10  # 10s audio
    if rep_duration not in df['DURATION_SEC'].unique():
        rep_duration = df['DURATION_SEC'].median()
    
    subset = df[df['DURATION_SEC'] == rep_duration]
    
    # Get means for each timing method across thread counts
    wall_means = subset.groupby('THREADS')['WALL_TIME'].mean()
    cpu_means = subset.groupby('THREADS')['CPU_TIME'].mean()
    user_means = subset.groupby('THREADS')['USER_TIME'].mean()
    clock_means = subset.groupby('THREADS')['CLOCK_TIME'].mean()
    
    thread_counts = sorted(subset['THREADS'].unique())
    x = np.arange(len(thread_counts))
    width = 0.2
    
    plt.bar(x - 1.5*width, wall_means.values, width, label='Wall Time')
    plt.bar(x - 0.5*width, user_means.values, width, label='User Time')
    plt.bar(x + 0.5*width, cpu_means.values, width, label='CPU Time')
    plt.bar(x + 1.5*width, clock_means.values, width, label='Clock Time')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Time (seconds)')
    plt.title(f'Comparison of Time Measurements ({rep_duration}s Audio)')
    plt.xticks(x, thread_counts)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'time_measurement_comparison.png'), dpi=300)
    
    # 6. Linear regression for each thread count
    plt.figure(figsize=(12, 8))
    
    from scipy import stats
    
    for threads in sorted(df['THREADS'].unique()):
        subset = df[df['THREADS'] == threads]
        x = subset['DURATION_SEC'].values
        y = subset['USER_TIME'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        plt.scatter(x, y, alpha=0.6, label=f'{threads} threads')
        x_sorted = np.array(sorted(x))  # Convert to numpy array
        plt.plot(x_sorted, intercept + slope*x_sorted, '--')
        
        print(f"Threads: {threads}, Linear regression (USER TIME): y = {slope:.4f}x + {intercept:.4f}, R²: {r_value**2:.4f}")
    
    plt.xlabel('Audio Duration (seconds)')
    plt.ylabel('User Time (seconds)')
    plt.title('Linear Regression: User Processing Time vs. Duration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'linear_regression_user_time.png'), dpi=300)
    
    # 7. Thread count impact on component efficiency (NUM_COMPONENTS / NUM_PEAKS)
    if 'NUM_COMPONENTS' in df.columns and 'NUM_PEAKS' in df.columns:
        plt.figure(figsize=(12, 8))
        
        df['COMPRESSION_RATIO'] = df['NUM_PEAKS'] / df['NUM_COMPONENTS']
        
        for duration in sorted(df['DURATION_SEC'].unique()):
            subset = df[df['DURATION_SEC'] == duration]
            means = subset.groupby('THREADS')['COMPRESSION_RATIO'].mean()
            std_devs = subset.groupby('THREADS')['COMPRESSION_RATIO'].std()
            plt.errorbar(means.index, means.values, yerr=std_devs.values, 
                        label=f'{duration}s audio', marker='o')
        
        plt.xlabel('Number of Threads')
        plt.ylabel('Compression Ratio (Peaks/Components)')
        plt.title('Component Extraction Efficiency vs. Thread Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(report_dir, 'compression_ratio_vs_threads.png'), dpi=300)
    
    # 8. CPU Utilization Chart (TOTAL_THREAD_TIME / (THREADS * WALL_TIME))
    if 'CPU_TIME' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter out single-thread runs
        multi_thread_df = df[df['THREADS'] > 1].copy()
        multi_thread_df['CPU_UTILIZATION'] = multi_thread_df['CPU_TIME'] / (multi_thread_df['THREADS'] * multi_thread_df['WALL_TIME']) * 100
        
        for duration in sorted(multi_thread_df['DURATION_SEC'].unique()):
            subset = multi_thread_df[multi_thread_df['DURATION_SEC'] == duration]
            means = subset.groupby('THREADS')['CPU_UTILIZATION'].mean()
            std_devs = subset.groupby('THREADS')['CPU_UTILIZATION'].std()
            plt.errorbar(means.index, means.values, yerr=std_devs.values, 
                        label=f'{duration}s audio', marker='o')
        
        plt.xlabel('Number of Threads')
        plt.ylabel('CPU Utilization (%)')
        plt.title('CPU Utilization vs. Thread Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(report_dir, 'cpu_utilization_vs_threads.png'), dpi=300)
    
    # 9. If Log Spectral Distance data is available, plot it
    if 'LOG_SPECTRAL_DISTANCE' in df.columns:
        plt.figure(figsize=(12, 8))
        
        for duration in sorted(df['DURATION_SEC'].unique()):
            subset = df[df['DURATION_SEC'] == duration]
            means = subset.groupby('THREADS')['LOG_SPECTRAL_DISTANCE'].mean()
            std_devs = subset.groupby('THREADS')['LOG_SPECTRAL_DISTANCE'].std()
            plt.errorbar(means.index, means.values, yerr=std_devs.values, 
                        label=f'{duration}s audio', marker='o')
        
        plt.xlabel('Number of Threads')
        plt.ylabel('Log Spectral Distance (dB)')
        plt.title('Audio Quality vs. Thread Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(report_dir, 'lsd_vs_threads.png'), dpi=300)
    
    # Create summary table for thesis
    avg_results = df.groupby(['THREADS']).agg({
        'USER_TIME': ['mean', 'std'],
        'WALL_TIME': ['mean', 'std'],
        'CPU_TIME': ['mean', 'std'],
        'NUM_COMPONENTS': ['mean', 'std']
    }).reset_index()
    
    # Format the table for LaTeX
    with open(os.path.join(report_dir, 'thesis_table.tex'), 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance of Dirichlet Kernel Deconvolution with Varying Thread Counts}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Threads & User Time (s) & Wall Time (s) & CPU Time (s) & Components \\\\ \\hline\n")
        
        for _, row in avg_results.iterrows():
            threads = int(row['THREADS'])
            user_time = f"{row[('USER_TIME', 'mean')]:.2f} ± {row[('USER_TIME', 'std')]:.2f}"
            wall_time = f"{row[('WALL_TIME', 'mean')]:.2f} ± {row[('WALL_TIME', 'std')]:.2f}"
            cpu_time = f"{row[('CPU_TIME', 'mean')]:.2f} ± {row[('CPU_TIME', 'std')]:.2f}"
            components = f"{row[('NUM_COMPONENTS', 'mean')]:.1f} ± {row[('NUM_COMPONENTS', 'std')]:.1f}"
            
            f.write(f"{threads} & {user_time} & {wall_time} & {cpu_time} & {components} \\\\ \\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\end{table}")
    
    print(f"Report generated in {report_dir}")
    return report_dir



def main():
    parser = argparse.ArgumentParser(description='Run audio processing benchmarks.')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'Number of trials per configuration (default: {DEFAULT_TRIALS})')
    parser.add_argument('--threads', type=int, nargs='+', default=THREAD_COUNTS,
                        help=f'Thread counts to benchmark (default: {THREAD_COUNTS})')
    parser.add_argument('--durations', type=int, nargs='+', default=DURATIONS,
                        help=f'Audio durations in seconds to benchmark (default: {DURATIONS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output during benchmarking')
    parser.add_argument('--report-only', type=str, default=None,
                        help='Generate report from existing CSV file without running benchmarks')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output filename for results CSV')
    
    args = parser.parse_args()
    
    if args.report_only:
        if os.path.exists(args.report_only):
            report_dir = generate_report(args.report_only)
            print(f"Report generated using existing data from {args.report_only}")
            return
        else:
            print(f"Error: File {args.report_only} does not exist.")
            return
    
    # Check if executable exists
    if not args.dry_run and not os.path.exists(EXECUTABLE):
        print(f"Error: Executable {EXECUTABLE} not found. Please compile it first.")
        return
    
    # Create timestamped filename for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.output if args.output else os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.csv")
    
    all_results = []
    total_configs = len(args.durations) * len(args.threads)
    completed = 0
    
    print(f"Starting benchmarks with {args.trials} trials per configuration")
    print(f"Thread counts: {args.threads}")
    print(f"Audio durations: {args.durations}")
    print(f"Total configurations to test: {total_configs}")
    
    # Run all benchmarks
    for duration in args.durations:
        for threads in args.threads:
            print(f"\nConfiguration {completed+1}/{total_configs}: {duration}s audio, {threads} threads")
            results = run_benchmark(duration, threads, args.trials, args.dry_run, args.verbose)
            if results:
                all_results.append(results)
            completed += 1
            
            # Show progress
            progress = completed / total_configs * 100
            print(f"Overall progress: {progress:.1f}% complete")
    
    # Save and analyze results
    if all_results:
        save_results(all_results, results_file)
        report_dir = generate_report(results_file)
        print(f"Benchmark complete. Data saved to {results_file}")
        print(f"Report generated in {report_dir}")
    else:
        print("No benchmark results collected.")

if __name__ == "__main__":
    main()
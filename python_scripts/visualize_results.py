#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_compression_performance(csv_file='compression_results.csv'):
    """Plot compression ratio and time (Figure 2 from paper)"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Compression Ratio vs File Size
    ax1.plot(df['File_Size'], df['Compression_Ratio'], 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('File Size (bytes)', fontsize=12)
    ax1.set_ylabel('Compression Ratio (%)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Compression Performance', fontsize=14, fontweight='bold')
    
    # Plot 2: Processing Time
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['File_Size'], df['Processing_Time_ms'], 'r-s', linewidth=2, markersize=8)
    ax1_twin.set_ylabel('Processing Time (ms)', fontsize=12, color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(['Compression Ratio'], loc='upper left')
    ax1_twin.legend(['Processing Time'], loc='upper right')
    
    # Plot 2: Compression Efficiency
    ax2.bar(range(len(df)), df['Compression_Ratio'], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Data Sample', fontsize=12)
    ax2.set_ylabel('Compression Ratio (%)', fontsize=12)
    ax2.set_title('Compression Ratio Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('compression_performance.png', dpi=300, bbox_inches='tight')
    print("Saved: compression_performance.png")
    plt.show()

def plot_ckks_benchmark(csv_file='ckks_benchmark_results.csv'):
    """Plot CKKS operation performance (Table I from paper)"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    operations = df['Operation']
    times = df['Time_ms']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.barh(operations, times, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (op, time) in enumerate(zip(operations, times)):
        ax.text(time + max(times)*0.02, i, f'{time:.2f} ms', 
               va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('CKKS Operation Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('ckks_benchmark.png', dpi=300, bbox_inches='tight')
    print("Saved: ckks_benchmark.png")
    plt.show()

def plot_pipeline_comparison(csv_file='pipeline_results.csv'):
    """Plot full pipeline metrics"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    
    df = pd.read_csv(csv_file)
    
    # Extract metrics
    metrics = {}
    for _, row in df.iterrows():
        metrics[row['Metric']] = row['Value']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time breakdown
    operations = ['Compression', 'Encryption', 'Decryption', 'Decompression']
    times = [
        metrics.get('Compression_time_ms', 0),
        metrics.get('Encryption_time_ms', 0),
        metrics.get('Decryption_time_ms', 0),
        metrics.get('Decompression_time_ms', 0)
    ]
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    ax1.pie(times, labels=operations, autopct='%1.1f%%', colors=colors, 
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Pipeline Time Distribution', fontsize=14, fontweight='bold')
    
    # Plot 2: Data size comparison
    categories = ['Original', 'Compressed']
    sizes = [
        metrics.get('Original_size_bytes', 0),
        metrics.get('Compressed_size_bytes', 0)
    ]
    
    bars = ax2.bar(categories, sizes, color=['#95a5a6', '#27ae60'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Size (bytes)', fontsize=12, fontweight='bold')
    ax2.set_title('Data Size Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(size)} bytes',
                ha='center', va='bottom', fontweight='bold')
    
    # Add compression ratio text
    ratio = metrics.get('Compression_ratio_percent', 0)
    ax2.text(0.5, max(sizes)*0.5, f'Compression:\n{ratio:.1f}%', 
            ha='center', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('pipeline_results.png', dpi=300, bbox_inches='tight')
    print("Saved: pipeline_results.png")
    plt.show()

def plot_paper_comparison():
    """Recreate plots from the paper"""
    print("\n" + "="*60)
    print("Generating Paper-Style Plots")
    print("="*60)
    
    # Simulated data matching paper's Figure 2
    # Text compression
    file_sizes = np.array([10000, 20000, 40000, 60000, 80000])
    compression_ratios = np.array([23.24, 30.15, 38.42, 45.67, 48.83])
    processing_times = np.array([0.75, 2.34, 5.12, 8.45, 12.16])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Text compression
    color1 = '#1f77b4'
    color2 = '#ff7f0e'
    
    ax1.plot(file_sizes, compression_ratios, 'o-', color=color1, linewidth=2, 
            markersize=10, label='Compression Ratio (%)')
    ax1.set_xlabel('File Size (bytes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Compression Ratio (%)', fontsize=12, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(file_sizes, processing_times, 's-', color=color2, linewidth=2, 
                 markersize=10, label='Processing Time (ms)')
    ax1_twin.set_ylabel('Processing Time (ms)', fontsize=12, fontweight='bold', color=color2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('a. Text Data Compression', fontsize=13, fontweight='bold')
    
    # Image compression
    image_pixels = np.array([5000, 10000, 15000, 20000])
    img_compression_ratios = np.array([60.46, 75.23, 85.67, 90.97])
    img_processing_times = np.array([0.0366, 0.5234, 2.1456, 6.3945])
    
    ax2.plot(image_pixels, img_compression_ratios, 'o-', color=color1, linewidth=2,
            markersize=10, label='Compression Ratio (%)')
    ax2.set_xlabel('Image Size (n*n pixels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Compression Ratio (%)', fontsize=12, fontweight='bold', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(image_pixels, img_processing_times, 's-', color=color2, linewidth=2,
                 markersize=10, label='Processing Time (s)')
    ax2_twin.set_ylabel('Processing Time (s)', fontsize=12, fontweight='bold', color=color2)
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    
    ax2.set_title('b. Image Data Compression', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper_figure2_recreation.png', dpi=300, bbox_inches='tight')
    print("Saved: paper_figure2_recreation.png")
    plt.show()

def generate_summary_report(compression_csv='compression_results.csv',
                           ckks_csv='ckks_benchmark_results.csv',
                           pipeline_csv='pipeline_results.csv'):
    """Generate text summary report"""
    report = []
    report.append("="*60)
    report.append("EXPERIMENTAL RESULTS SUMMARY")
    report.append("="*60)
    
    # Compression results
    if os.path.exists(compression_csv):
        df = pd.read_csv(compression_csv)
        report.append("\n--- Compression Performance ---")
        report.append(f"Average compression ratio: {df['Compression_Ratio'].mean():.2f}%")
        report.append(f"Best compression ratio: {df['Compression_Ratio'].max():.2f}%")
        report.append(f"Average processing time: {df['Processing_Time_ms'].mean():.2f} ms")
    
    # CKKS results
    if os.path.exists(ckks_csv):
        df = pd.read_csv(ckks_csv)
        report.append("\n--- CKKS Performance ---")
        for _, row in df.iterrows():
            report.append(f"{row['Operation']}: {row['Time_ms']:.3f} ms")
    
    # Pipeline results
    if os.path.exists(pipeline_csv):
        df = pd.read_csv(pipeline_csv)
        report.append("\n--- Full Pipeline ---")
        for _, row in df.iterrows():
            metric = row['Metric'].replace('_', ' ').title()
            report.append(f"{metric}: {row['Value']:.2f}")
    
    report.append("\n" + "="*60)
    
    # Print and save
    report_text = '\n'.join(report)
    print(report_text)
    
    with open('experiment_summary.txt', 'w') as f:
        f.write(report_text)
    print("\nSaved: experiment_summary.txt")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize compression and CKKS results')
    parser.add_argument('--type', choices=['compression', 'ckks', 'pipeline', 'paper', 'all'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    if args.type == 'all' or args.type == 'compression':
        plot_compression_performance()
    
    if args.type == 'all' or args.type == 'ckks':
        plot_ckks_benchmark()
    
    if args.type == 'all' or args.type == 'pipeline':
        plot_pipeline_comparison()
    
    if args.type == 'paper':
        plot_paper_comparison()
    
    if args.type == 'all':
        generate_summary_report()
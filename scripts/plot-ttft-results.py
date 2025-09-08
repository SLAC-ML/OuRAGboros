#!/usr/bin/env python3

"""
Generate TTFT visualization plots from benchmark results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob

def find_latest_results():
    """Find the latest benchmark results"""
    results_dirs = glob.glob("benchmark-results-streaming/quick_*")
    if not results_dirs:
        print("❌ No quick benchmark results found. Please run:")
        print("   python3 scripts/quick-ttft-benchmark.py")
        return None
    
    latest_dir = sorted(results_dirs)[-1]
    return latest_dir

def load_results(results_dir):
    """Load benchmark results"""
    json_file = f"{results_dir}/quick_results.json"
    csv_file = f"{results_dir}/quick_results.csv"
    
    if not os.path.exists(json_file):
        print(f"❌ Results file not found: {json_file}")
        return None, None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # Handle both old format (ttft_ms) and new format (ttft_s)
        if 'ttft_s' in df.columns:
            df['ttft_value'] = df['ttft_s']
            df['ttft_units'] = 'seconds'
        elif 'ttft_ms' in df.columns:
            df['ttft_value'] = df['ttft_ms'] / 1000.0  # Convert to seconds
            df['ttft_units'] = 'seconds (converted from ms)'
        else:
            raise ValueError("No TTFT column found in CSV")
    else:
        # Create DataFrame from JSON data
        rows = []
        for result in data:
            for i, ttft in enumerate(result['ttfts']):
                response_time = result.get('response_times', [0] * len(result['ttfts']))[i] if i < len(result.get('response_times', [])) else 0
                rows.append({
                    'concurrency': result['concurrency'], 
                    'ttft_value': ttft,  # Already in seconds from new format
                    'response_time_ms': response_time * 1000,
                    'ttft_units': 'seconds'
                })
        df = pd.DataFrame(rows)
    
    return data, df

def create_plots(data, df, output_dir):
    """Create visualization plots"""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('OuRAGboros Streaming API - TTFT Concurrency Analysis', fontsize=16)
    
    # 1. Box plot of TTFT distribution by concurrency
    ax1.set_title('Time To First Token Distribution by Concurrency')
    concurrency_levels = sorted(df['concurrency'].unique())
    ttft_data = [df[df['concurrency'] == c]['ttft_value'].values for c in concurrency_levels]
    
    box_plot = ax1.boxplot(ttft_data, labels=concurrency_levels, patch_artist=True)
    ax1.set_xlabel('Concurrency Level')
    ax1.set_ylabel('TTFT (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
    
    # 2. Mean TTFT vs Concurrency
    ax2.set_title('Average TTFT vs Concurrency Level')
    concurrencies = [result['concurrency'] for result in data]
    mean_ttfts = [result['mean_ttft'] for result in data]
    min_ttfts = [result['min_ttft'] for result in data]
    max_ttfts = [result['max_ttft'] for result in data]
    
    ax2.plot(concurrencies, mean_ttfts, 'o-', linewidth=2, markersize=8, label='Mean TTFT')
    ax2.fill_between(concurrencies, min_ttfts, max_ttfts, alpha=0.3, label='Min-Max Range')
    ax2.set_xlabel('Concurrency Level')
    ax2.set_ylabel('TTFT (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(concurrencies)
    
    # 3. Histogram of TTFT values with common bins
    ax3.set_title('TTFT Distribution Histogram')
    
    # Calculate common bins across all data
    all_ttft_values = df['ttft_value'].values
    min_ttft = all_ttft_values.min()
    max_ttft = all_ttft_values.max()
    common_bins = np.linspace(min_ttft, max_ttft, 6)  # 5 bins
    
    for i, concurrency in enumerate(concurrency_levels):
        ttft_values = df[df['concurrency'] == concurrency]['ttft_value']
        ax3.hist(ttft_values, bins=common_bins, histtype='step', linewidth=2,
                label=f'Concurrency {concurrency}', 
                color=colors[i % len(colors)])
    
    ax3.set_xlabel('TTFT (seconds)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Success Rate and Performance Summary
    ax4.set_title('Success Rate by Concurrency Level')
    success_rates = [(result['successful'] / result['total']) * 100 for result in data]
    
    bars = ax4.bar(concurrencies, success_rates, alpha=0.8, color=colors[:len(concurrencies)])
    ax4.set_xlabel('Concurrency Level')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plots
    png_file = f"{output_dir}/ttft_analysis.png"
    pdf_file = f"{output_dir}/ttft_analysis.pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    
    print(f"📊 Plots saved:")
    print(f"  - {png_file}")
    print(f"  - {pdf_file}")
    
    # Create detailed violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, x='concurrency', y='ttft_value')
    plt.title('TTFT Distribution - Violin Plot by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Time To First Token (seconds)')
    plt.grid(True, alpha=0.3)
    
    violin_file = f"{output_dir}/ttft_violin_plot.png"
    plt.savefig(violin_file, dpi=300, bbox_inches='tight')
    print(f"  - {violin_file}")
    
    # Don't show plots interactively in headless environment
    plt.close()

def print_analysis(data, df):
    """Print detailed analysis"""
    print("\n" + "="*60)
    print("📈 TTFT CONCURRENCY ANALYSIS")
    print("="*60)
    
    print(f"\n{'Concurrency':<12} {'Success%':<9} {'Mean TTFT':<11} {'Min TTFT':<10} {'Max TTFT':<10} {'Std Dev':<10}")
    print("-" * 62)
    
    for result in data:
        concurrency = result['concurrency']
        success_rate = (result['successful'] / result['total']) * 100
        
        # Calculate std dev for this concurrency level
        ttft_values = df[df['concurrency'] == concurrency]['ttft_value'].values
        std_dev = np.std(ttft_values) if len(ttft_values) > 1 else 0
        
        print(f"{concurrency:<12} {success_rate:<9.1f} {result['mean_ttft']:<11.2f} "
              f"{result['min_ttft']:<10.2f} {result['max_ttft']:<10.2f} {std_dev:<10.3f}")
    
    # Performance insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # TTFT degradation
    mean_ttfts = [result['mean_ttft'] for result in data]
    ttft_increase = ((mean_ttfts[-1] - mean_ttfts[0]) / mean_ttfts[0]) * 100
    print(f"  • TTFT increases by {ttft_increase:.1f}% from concurrency 1 to {data[-1]['concurrency']}")
    
    # Find optimal concurrency
    best_concurrency = min(data, key=lambda x: x['mean_ttft'])['concurrency']
    worst_concurrency = max(data, key=lambda x: x['mean_ttft'])['concurrency']
    print(f"  • Best performance: Concurrency {best_concurrency}")
    print(f"  • Worst performance: Concurrency {worst_concurrency}")
    
    # Variability analysis
    variabilities = []
    for result in data:
        if result['mean_ttft'] > 0:
            variability = (result['max_ttft'] - result['min_ttft']) / result['mean_ttft']
            variabilities.append(variability)
    
    avg_variability = np.mean(variabilities) * 100
    print(f"  • Average TTFT variability: {avg_variability:.1f}%")
    
    # Success rate analysis
    success_rates = [(r['successful'] / r['total']) * 100 for r in data]
    if all(rate == 100.0 for rate in success_rates):
        print(f"  • ✅ All concurrency levels achieved 100% success rate")
    else:
        min_success = min(success_rates)
        print(f"  • ⚠️  Success rate drops to {min_success:.1f}% at high concurrency")

def main():
    print("📊 TTFT Visualization Generator")
    print("=" * 40)
    
    # Find latest results
    results_dir = find_latest_results()
    if not results_dir:
        return 1
    
    print(f"📁 Using results from: {results_dir}")
    
    # Load data
    data, df = load_results(results_dir)
    if data is None:
        return 1
    
    print(f"📊 Loaded {len(data)} concurrency levels with {len(df)} total measurements")
    
    # Create plots
    create_plots(data, df, results_dir)
    
    # Print analysis
    print_analysis(data, df)
    
    print(f"\n✅ Analysis complete! Check {results_dir}/ for all outputs")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
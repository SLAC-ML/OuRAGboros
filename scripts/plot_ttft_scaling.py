#!/usr/bin/env python3
"""
TTFT Scaling Analysis Visualization

This script analyzes and plots TTFT (Time To First Token) scaling behavior
across different concurrency levels for OuRAGboros API performance analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import sys

def load_ttft_data(filenames):
    """Load TTFT data from JSON files"""
    data = {}
    
    for filename in filenames:
        try:
            with open(filename, 'r') as f:
                result = json.load(f)
            
            # Extract concurrency from filename (e.g., ttft_c5.json -> 5 or ttft_mock_c5.json -> 5)
            if 'ttft_mock_c' in filename:
                concurrency = int(filename.split('ttft_mock_c')[1].split('.json')[0])
            elif 'ttft_c' in filename:
                concurrency = int(filename.split('ttft_c')[1].split('.json')[0])
            else:
                print(f"Warning: Could not extract concurrency from {filename}")
                continue
                
            if 'ttft_milliseconds' in result:
                data[concurrency] = {
                    'total_samples': result['total_samples'],
                    'successful': result['successful'],
                    'failed': result['failed'],
                    'success_rate': result['success_rate'],
                    'mean_ms': result['ttft_milliseconds']['mean'],
                    'median_ms': result['ttft_milliseconds']['median'],
                    'min_ms': result['ttft_milliseconds']['min'],
                    'max_ms': result['ttft_milliseconds']['max'],
                    'p95_ms': result['ttft_milliseconds']['p95'],
                    'p99_ms': result['ttft_milliseconds']['p99'],
                    'stdev_ms': result['ttft_milliseconds']['stdev']
                }
            else:
                print(f"Warning: No TTFT data found in {filename}")
                
        except FileNotFoundError:
            print(f"Warning: File {filename} not found")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from {filename}")
        except Exception as e:
            print(f"Warning: Error processing {filename}: {e}")
    
    return data

def create_scaling_plot(data, output_file='ttft_scaling_analysis.png'):
    """Create comprehensive TTFT scaling analysis plot"""
    if not data:
        print("Error: No data to plot")
        return
    
    # Sort data by concurrency level
    concurrency_levels = sorted(data.keys())
    
    # Extract metrics
    p99_values = [data[c]['p99_ms'] for c in concurrency_levels]
    p95_values = [data[c]['p95_ms'] for c in concurrency_levels]
    mean_values = [data[c]['mean_ms'] for c in concurrency_levels]
    median_values = [data[c]['median_ms'] for c in concurrency_levels]
    success_rates = [data[c]['success_rate'] for c in concurrency_levels]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TTFT Scaling Analysis - OuRAGboros API Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: P99 vs Concurrency (main plot)
    ax1.plot(concurrency_levels, p99_values, 'ro-', linewidth=2, markersize=8, label='P99 TTFT')
    ax1.plot(concurrency_levels, p95_values, 'bo-', linewidth=2, markersize=6, label='P95 TTFT')
    ax1.set_xlabel('Concurrency Level')
    ax1.set_ylabel('Time To First Token (ms)')
    ax1.set_title('P99/P95 TTFT vs Concurrency', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')  # Log scale to show exponential growth
    
    # Add performance thresholds
    ax1.axhline(y=800, color='orange', linestyle='--', alpha=0.7, label='Good (<800ms)')
    ax1.axhline(y=500, color='green', linestyle='--', alpha=0.7, label='Excellent (<500ms)')
    ax1.legend()
    
    # Plot 2: All percentiles comparison
    ax2.plot(concurrency_levels, mean_values, 'g^-', label='Mean', markersize=6)
    ax2.plot(concurrency_levels, median_values, 'y*-', label='Median', markersize=8)
    ax2.plot(concurrency_levels, p95_values, 'bo-', label='P95', markersize=6)
    ax2.plot(concurrency_levels, p99_values, 'ro-', label='P99', markersize=6)
    ax2.set_xlabel('Concurrency Level')
    ax2.set_ylabel('TTFT (ms)')
    ax2.set_title('TTFT Percentiles Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # Plot 3: Success Rate
    ax3.bar(concurrency_levels, success_rates, color=['green' if sr >= 95 else 'orange' if sr >= 80 else 'red' for sr in success_rates])
    ax3.set_xlabel('Concurrency Level')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate vs Concurrency', fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add success rate text on bars
    for i, (c, sr) in enumerate(zip(concurrency_levels, success_rates)):
        ax3.text(c, sr + 1, f'{sr:.1f}%', ha='center', fontweight='bold')
    
    # Plot 4: Scaling analysis (linear vs actual)
    # Calculate theoretical linear scaling (assuming c=1 as baseline)
    baseline_p99 = p99_values[0] if concurrency_levels[0] == 1 else p99_values[0]
    linear_scaling = [baseline_p99 * c for c in concurrency_levels]
    
    ax4.plot(concurrency_levels, p99_values, 'ro-', linewidth=3, markersize=8, label='Actual P99 TTFT')
    ax4.plot(concurrency_levels, linear_scaling, 'g--', linewidth=2, label='Linear Scaling (ideal)')
    ax4.set_xlabel('Concurrency Level')
    ax4.set_ylabel('P99 TTFT (ms)')
    ax4.set_title('Scaling Behavior: Actual vs Linear', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Scaling analysis plot saved to: {output_file}")
    
    return fig

def print_scaling_analysis(data):
    """Print detailed scaling analysis"""
    print("\n" + "="*60)
    print("📊 TTFT CONCURRENCY SCALING ANALYSIS")
    print("="*60)
    
    concurrency_levels = sorted(data.keys())
    
    print(f"\n{'Concurrency':<12} {'P99 (ms)':<10} {'P95 (ms)':<10} {'Mean (ms)':<10} {'Success %':<10}")
    print("-" * 58)
    
    for c in concurrency_levels:
        d = data[c]
        print(f"{c:<12} {d['p99_ms']:<10.0f} {d['p95_ms']:<10.0f} {d['mean_ms']:<10.0f} {d['success_rate']:<10.1f}")
    
    # Calculate scaling factors
    print(f"\n🔍 SCALING ANALYSIS:")
    baseline_p99 = data[concurrency_levels[0]]['p99_ms']
    
    for c in concurrency_levels:
        if c == concurrency_levels[0]:
            continue
        actual_p99 = data[c]['p99_ms']
        scaling_factor = actual_p99 / baseline_p99
        linear_expectation = c
        efficiency = linear_expectation / scaling_factor * 100
        
        print(f"  Concurrency {c}: {scaling_factor:.1f}x slower (vs {linear_expectation:.1f}x ideal) - {efficiency:.1f}% efficiency")
    
    # Key observations
    print(f"\n🎯 KEY OBSERVATIONS:")
    worst_concurrency = max(concurrency_levels, key=lambda c: data[c]['p99_ms'])
    worst_p99 = data[worst_concurrency]['p99_ms']
    worst_success = min(data[c]['success_rate'] for c in concurrency_levels)
    
    print(f"  • Worst P99 TTFT: {worst_p99:.0f}ms at concurrency {worst_concurrency}")
    print(f"  • Lowest success rate: {worst_success:.1f}%")
    
    # Identify bottleneck point
    bottleneck_point = None
    for i, c in enumerate(concurrency_levels[1:], 1):
        prev_c = concurrency_levels[i-1]
        p99_ratio = data[c]['p99_ms'] / data[prev_c]['p99_ms']
        if p99_ratio > 3:  # More than 3x degradation
            bottleneck_point = prev_c
            break
    
    if bottleneck_point:
        print(f"  • Severe bottleneck detected between concurrency {bottleneck_point} and {bottleneck_point*2}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if worst_p99 > 5000:
        print("  • CRITICAL: P99 TTFT >5s indicates severe performance issues")
    if worst_success < 90:
        print("  • CRITICAL: Success rate <90% indicates reliability issues")
    if bottleneck_point and bottleneck_point <= 5:
        print(f"  • Consider investigating resource limits or serialization bottlenecks")
        print(f"  • Check LLM API rate limits and connection pooling")
        print(f"  • Monitor CPU/memory usage during concurrent loads")

def main():
    parser = argparse.ArgumentParser(description='TTFT Scaling Analysis and Visualization')
    parser.add_argument('files', nargs='*', default=['ttft_c1.json', 'ttft_c2.json', 'ttft_c5.json', 'ttft_c10.json', 'ttft_c20.json'],
                       help='TTFT result JSON files (default: ttft_c*.json)')
    parser.add_argument('-o', '--output', default='ttft_scaling_analysis.png',
                       help='Output plot filename (default: ttft_scaling_analysis.png)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation, show analysis only')
    
    args = parser.parse_args()
    
    # Load data
    data = load_ttft_data(args.files)
    
    if not data:
        print("Error: No valid TTFT data found")
        sys.exit(1)
    
    # Print analysis
    print_scaling_analysis(data)
    
    # Generate plot
    if not args.no_plot:
        try:
            create_scaling_plot(data, args.output)
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
            print("Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
Streaming API Concurrency Benchmark with TTFT Analysis
Tests different concurrency levels and measures Time To First Token distributions
"""

import os
import sys
import time
import json
import requests
import concurrent.futures
import threading
from datetime import datetime
from typing import List, Dict, Any, Tuple
import statistics

# Configuration
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001")
TOTAL_REQUESTS = int(os.environ.get("TOTAL_REQUESTS", "50"))
CONCURRENCY_LEVELS = [1, 2, 5, 10, 20]

def create_results_directory():
    """Create benchmark results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"benchmark-results-streaming/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir, timestamp

def test_streaming_request(request_id: int, query: str) -> Dict[str, Any]:
    """Make a single streaming request and measure TTFT"""
    
    payload = {
        "query": f"{query} (Request #{request_id})",
        "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "stanford:gpt-4o",
        "prompt": "You are a helpful assistant. Please provide a comprehensive answer.",
        "use_rag": False
    }
    
    start_time = time.time()
    first_token_time = None
    last_token_time = None
    total_tokens = 0
    total_chars = 0
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask/stream",
            json=payload,
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return {
                "request_id": request_id,
                "success": False,
                "error": f"HTTP {response.status_code}",
                "start_time": start_time
            }
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                current_time = time.time()
                
                if first_token_time is None:
                    first_token_time = current_time
                
                if line.startswith(b'data: '):
                    data_str = line[6:].decode('utf-8')
                    if data_str == '[DONE]':
                        last_token_time = current_time
                        break
                    
                    try:
                        data = json.loads(data_str)
                        delta = data.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            total_tokens += 1
                            total_chars += len(content)
                            last_token_time = current_time
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.time()
        
        return {
            "request_id": request_id,
            "success": True,
            "start_time": start_time,
            "first_token_time": first_token_time,
            "last_token_time": last_token_time or end_time,
            "end_time": end_time,
            "ttft": (first_token_time - start_time) * 1000 if first_token_time else None,  # ms
            "total_time": (end_time - start_time) * 1000,  # ms
            "generation_time": ((last_token_time or end_time) - (first_token_time or start_time)) * 1000 if first_token_time else None,  # ms
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "tokens_per_second": total_tokens / ((last_token_time or end_time) - (first_token_time or start_time)) if first_token_time and total_tokens > 0 else 0
        }
        
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
            "start_time": start_time
        }

def run_concurrency_test(concurrency: int, total_requests: int) -> List[Dict[str, Any]]:
    """Run streaming requests with specified concurrency"""
    print(f"🚀 Testing concurrency level: {concurrency}")
    print(f"   Total requests: {total_requests}")
    
    # Generate diverse queries to simulate real usage
    base_queries = [
        "Explain quantum mechanics and its fundamental principles",
        "What are the key differences between machine learning and deep learning?",
        "Describe the process of photosynthesis in plants",
        "How does the stock market work and what factors influence it?",
        "What are the main causes and effects of climate change?",
        "Explain the theory of relativity in simple terms",
        "How do neural networks learn and make predictions?",
        "What is the significance of DNA in genetics?",
        "Describe the water cycle and its importance",
        "How do vaccines work to prevent diseases?"
    ]
    
    queries = [base_queries[i % len(base_queries)] for i in range(total_requests)]
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(test_streaming_request, i, queries[i])
            for i in range(total_requests)
        ]
        
        results = []
        completed = 0
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 10 == 0 or completed == total_requests:
                print(f"   Completed: {completed}/{total_requests}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    failed_count = len(results) - len(successful_results)
    
    ttfts = [r["ttft"] for r in successful_results if r["ttft"] is not None]
    total_times = [r["total_time"] for r in successful_results]
    tokens_per_sec = [r["tokens_per_second"] for r in successful_results if r["tokens_per_second"] > 0]
    
    stats = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(successful_results),
        "failed_requests": failed_count,
        "total_test_time": total_time,
        "throughput": len(successful_results) / total_time,  # requests per second
        "ttft_stats": {
            "count": len(ttfts),
            "mean": statistics.mean(ttfts) if ttfts else 0,
            "median": statistics.median(ttfts) if ttfts else 0,
            "min": min(ttfts) if ttfts else 0,
            "max": max(ttfts) if ttfts else 0,
            "stdev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            "p95": sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) >= 20 else (max(ttfts) if ttfts else 0),
            "p99": sorted(ttfts)[int(len(ttfts) * 0.99)] if len(ttfts) >= 100 else (max(ttfts) if ttfts else 0)
        },
        "total_time_stats": {
            "mean": statistics.mean(total_times) if total_times else 0,
            "median": statistics.median(total_times) if total_times else 0,
            "min": min(total_times) if total_times else 0,
            "max": max(total_times) if total_times else 0
        },
        "tokens_per_second_stats": {
            "mean": statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
            "median": statistics.median(tokens_per_sec) if tokens_per_sec else 0
        }
    }
    
    print(f"   ✅ Success: {len(successful_results)}/{total_requests}")
    print(f"   📊 Throughput: {stats['throughput']:.2f} req/s")
    print(f"   ⏱️  TTFT: {stats['ttft_stats']['mean']:.0f}ms ± {stats['ttft_stats']['stdev']:.0f}ms")
    print(f"   📈 P95 TTFT: {stats['ttft_stats']['p95']:.0f}ms")
    print()
    
    return results, stats

def save_results(results_dir: str, all_results: Dict[str, Any], timestamp: str):
    """Save benchmark results to files"""
    
    # Save raw data
    with open(f"{results_dir}/raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary statistics
    summary = {
        "timestamp": timestamp,
        "base_url": BASE_URL,
        "total_requests": TOTAL_REQUESTS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "summary_stats": all_results["summary_stats"]
    }
    
    with open(f"{results_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save CSV for easy analysis
    csv_lines = ["concurrency,ttft_ms,total_time_ms,tokens_per_sec,success,request_id"]
    
    for concurrency, data in all_results["detailed_results"].items():
        for result in data["results"]:
            if result["success"]:
                csv_lines.append(
                    f"{concurrency},{result.get('ttft', 0)},{result.get('total_time', 0)},"
                    f"{result.get('tokens_per_second', 0)},1,{result['request_id']}"
                )
            else:
                csv_lines.append(f"{concurrency},0,0,0,0,{result['request_id']}")
    
    with open(f"{results_dir}/results.csv", "w") as f:
        f.write("\n".join(csv_lines))

def create_plot_script(results_dir: str):
    """Create a Python script to generate visualization plots"""
    
    plot_script = f'''#!/usr/bin/env python3

"""
Generate visualization plots for streaming benchmark results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load results
with open("{results_dir}/raw_results.json", "r") as f:
    data = json.load(f)

# Load CSV data
df = pd.read_csv("{results_dir}/results.csv")
df_success = df[df['success'] == 1]

# Set up the plot style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('OuRAGboros Streaming API - Concurrency Benchmark Results', fontsize=16)

# 1. TTFT Distribution by Concurrency
ax1.set_title('Time To First Token Distribution')
concurrency_levels = sorted(df_success['concurrency'].unique())
ttft_data = [df_success[df_success['concurrency'] == c]['ttft_ms'].values for c in concurrency_levels]
box_plot = ax1.boxplot(ttft_data, labels=concurrency_levels, patch_artist=True)
ax1.set_xlabel('Concurrency Level')
ax1.set_ylabel('TTFT (ms)')
ax1.grid(True, alpha=0.3)

# Color the boxes
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
    patch.set_facecolor(color)

# 2. Throughput vs Concurrency
ax2.set_title('Throughput vs Concurrency Level')
summary_stats = data['summary_stats']
concurrencies = []
throughputs = []
for level, stats in summary_stats.items():
    concurrencies.append(int(level))
    throughputs.append(stats['throughput'])

ax2.plot(concurrencies, throughputs, 'o-', linewidth=2, markersize=8)
ax2.set_xlabel('Concurrency Level')
ax2.set_ylabel('Throughput (requests/sec)')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(concurrencies)

# 3. TTFT Mean and P95 by Concurrency
ax3.set_title('TTFT: Mean vs P95 by Concurrency')
means = []
p95s = []
for level, stats in summary_stats.items():
    means.append(stats['ttft_stats']['mean'])
    p95s.append(stats['ttft_stats']['p95'])

x = np.arange(len(concurrencies))
width = 0.35
ax3.bar(x - width/2, means, width, label='Mean TTFT', alpha=0.8)
ax3.bar(x + width/2, p95s, width, label='P95 TTFT', alpha=0.8)
ax3.set_xlabel('Concurrency Level')
ax3.set_ylabel('TTFT (ms)')
ax3.set_xticks(x)
ax3.set_xticklabels(concurrencies)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Success Rate by Concurrency
ax4.set_title('Success Rate by Concurrency Level')
success_rates = []
for level, stats in summary_stats.items():
    success_rate = (stats['successful_requests'] / stats['total_requests']) * 100
    success_rates.append(success_rate)

bars = ax4.bar(concurrencies, success_rates, alpha=0.8)
ax4.set_xlabel('Concurrency Level')
ax4.set_ylabel('Success Rate (%)')
ax4.set_ylim(0, 105)
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{{:.1f}}%'.format(rate), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('{results_dir}/benchmark_plots.png', dpi=300, bbox_inches='tight')
plt.savefig('{results_dir}/benchmark_plots.pdf', bbox_inches='tight')

print("📊 Plots saved:")
print(f"  - {results_dir}/benchmark_plots.png")
print(f"  - {results_dir}/benchmark_plots.pdf")

# Create a detailed TTFT histogram
plt.figure(figsize=(12, 8))
for i, concurrency in enumerate(concurrency_levels):
    ttft_data = df_success[df_success['concurrency'] == concurrency]['ttft_ms']
    plt.hist(ttft_data, bins=20, alpha=0.6, label=f'Concurrency {{concurrency}}', 
             color=colors[i % len(colors)])

plt.xlabel('Time To First Token (ms)')
plt.ylabel('Frequency')
plt.title('TTFT Distribution Histogram by Concurrency Level')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('{results_dir}/ttft_histogram.png', dpi=300, bbox_inches='tight')

print(f"  - {results_dir}/ttft_histogram.png")

# Print summary statistics
print("\\n📈 BENCHMARK SUMMARY")
print("=" * 50)
for level, stats in summary_stats.items():
    print(f"\\nConcurrency {{level}}:")
    print(f"  Throughput: {{stats['throughput']:.2f}} req/s")
    print(f"  TTFT Mean: {{stats['ttft_stats']['mean']:.0f}}ms")
    print(f"  TTFT P95:  {{stats['ttft_stats']['p95']:.0f}}ms")
    print(f"  Success:   {{stats['successful_requests']}}/{{stats['total_requests']}}")
'''
    
    with open(f"{results_dir}/generate_plots.py", "w") as f:
        f.write(plot_script)
    
    # Make it executable
    os.chmod(f"{results_dir}/generate_plots.py", 0o755)

def main():
    print("🚀 OuRAGboros Streaming API Concurrency Benchmark")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Total Requests: {TOTAL_REQUESTS}")
    print(f"Concurrency Levels: {CONCURRENCY_LEVELS}")
    print()
    
    # Check if API is accessible
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not accessible: HTTP {response.status_code}")
            return 1
        print("✅ API is accessible")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure your services are running: docker compose up -d")
        return 1
    
    # Create results directory
    results_dir, timestamp = create_results_directory()
    print(f"📁 Results will be saved to: {results_dir}")
    print()
    
    # Run benchmark for each concurrency level
    all_results = {
        "timestamp": timestamp,
        "base_url": BASE_URL,
        "total_requests": TOTAL_REQUESTS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "detailed_results": {},
        "summary_stats": {}
    }
    
    for concurrency in CONCURRENCY_LEVELS:
        results, stats = run_concurrency_test(concurrency, TOTAL_REQUESTS)
        all_results["detailed_results"][str(concurrency)] = {
            "results": results,
            "stats": stats
        }
        all_results["summary_stats"][str(concurrency)] = stats
    
    # Save all results
    save_results(results_dir, all_results, timestamp)
    print(f"💾 Results saved to {results_dir}/")
    
    # Create plot generation script
    create_plot_script(results_dir)
    print(f"📊 Plot script created: {results_dir}/generate_plots.py")
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"{'Concurrency':<12} {'Throughput':<12} {'TTFT Mean':<12} {'TTFT P95':<12} {'Success':<10}")
    print("-" * 60)
    
    for concurrency in CONCURRENCY_LEVELS:
        stats = all_results["summary_stats"][str(concurrency)]
        print(f"{concurrency:<12} {stats['throughput']:<12.2f} {stats['ttft_stats']['mean']:<12.0f} "
              f"{stats['ttft_stats']['p95']:<12.0f} {stats['successful_requests']}/{stats['total_requests']}")
    
    print("\\n✅ Benchmark complete!")
    print("\\n💡 Next steps:")
    print(f"  1. Review detailed results: {results_dir}/")
    print(f"  2. Generate plots: python3 {results_dir}/generate_plots.py")
    print("  3. Analyze bottlenecks and optimize accordingly")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
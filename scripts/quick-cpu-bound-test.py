#!/usr/bin/env python3

"""
Quick CPU-bound vs I/O-bound Test
Demonstrates the difference between RAG-enabled (CPU-bound) and RAG-disabled (I/O-bound) performance
"""

import time
import json
import requests
import concurrent.futures
from datetime import datetime

BASE_URL = "http://localhost:8001"

def test_single_request(request_id, config_name, use_rag=False):
    """Test a single request and measure timing"""
    
    payload = {
        "query": f"Explain quantum mechanics in detail (Request #{request_id})",
        "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "stanford:gpt-4o",
        "prompt": "You are a helpful assistant.",
        "use_rag": use_rag,
        "use_opensearch": False,  # Force in-memory for CPU-bound test
        "max_documents": 3
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{BASE_URL}/ask/stream", json=payload, stream=True, timeout=60)
        
        if response.status_code != 200:
            return {"success": False, "ttft": None, "config": config_name}
        
        # Find first actual token
        for line in response.iter_lines():
            if line and line.startswith(b'data: '):
                data_str = line[6:].decode('utf-8')
                if data_str == '[DONE]':
                    break
                
                try:
                    data = json.loads(data_str)
                    if data.get('type') == 'token' and data.get('content'):
                        ttft = time.time() - start_time
                        return {"success": True, "ttft": ttft, "config": config_name, "use_rag": use_rag}
                except json.JSONDecodeError:
                    continue
        
        return {"success": False, "ttft": None, "config": config_name}
        
    except Exception as e:
        return {"success": False, "ttft": None, "config": config_name, "error": str(e)}

def test_concurrency(concurrency, config_name, use_rag, requests=5):
    """Test specific concurrency level"""
    print(f"Testing {config_name} - Concurrency {concurrency} (RAG={use_rag})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(test_single_request, i, config_name, use_rag)
            for i in range(requests)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    successful = [r for r in results if r.get("success", False)]
    ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
    
    if ttfts:
        mean_ttft = sum(ttfts) / len(ttfts)
        min_ttft = min(ttfts)
        max_ttft = max(ttfts)
        std_ttft = (sum((x - mean_ttft) ** 2 for x in ttfts) / len(ttfts)) ** 0.5 if len(ttfts) > 1 else 0
    else:
        mean_ttft = min_ttft = max_ttft = std_ttft = 0
    
    return {
        "config": config_name,
        "concurrency": concurrency,
        "use_rag": use_rag,
        "successful": len(successful),
        "total": len(results),
        "mean_ttft": mean_ttft,
        "min_ttft": min_ttft,
        "max_ttft": max_ttft,
        "std_ttft": std_ttft,
        "failed": [r for r in results if not r.get("success", False)]
    }

def main():
    print("🚀 Quick CPU-bound vs I/O-bound Test")
    print("=" * 50)
    print(f"Testing endpoint: {BASE_URL}")
    print()
    
    # Test configurations
    test_configs = [
        ("NO_RAG (I/O-bound)", False),
        ("WITH_RAG (CPU-bound)", True)
    ]
    
    concurrency_levels = [1, 5, 10]
    all_results = []
    
    for config_name, use_rag in test_configs:
        print(f"\\n📊 {config_name}")
        print("-" * 30)
        
        for concurrency in concurrency_levels:
            result = test_concurrency(concurrency, config_name, use_rag, requests=5)
            all_results.append(result)
            
            if result["successful"] > 0:
                degradation = ""
                if concurrency > 1 and len(all_results) > 1:
                    # Find baseline (concurrency 1) for this config
                    baseline = next((r for r in all_results if r["config"] == config_name and r["concurrency"] == 1), None)
                    if baseline and baseline["mean_ttft"] > 0:
                        degradation_pct = ((result["mean_ttft"] - baseline["mean_ttft"]) / baseline["mean_ttft"]) * 100
                        degradation = f" ({degradation_pct:+.1f}% vs C1)"
                
                print(f"  Concurrency {concurrency:2d}: "
                      f"{result['successful']}/{result['total']} success, "
                      f"TTFT: {result['mean_ttft']:.2f}s ± {result['std_ttft']:.2f}s "
                      f"({result['min_ttft']:.2f}-{result['max_ttft']:.2f}s){degradation}")
            else:
                print(f"  Concurrency {concurrency:2d}: ALL FAILED")
                for fail in result["failed"][:2]:  # Show first 2 failures
                    print(f"    Error: {fail.get('error', 'Unknown')}")
    
    # Analysis
    print("\\n" + "=" * 50)
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Compare I/O-bound vs CPU-bound
    no_rag_results = [r for r in all_results if not r["use_rag"]]
    rag_results = [r for r in all_results if r["use_rag"]]
    
    print("\\n🔍 Key Insights:")
    
    # Baseline comparison
    no_rag_baseline = next((r for r in no_rag_results if r["concurrency"] == 1), None)
    rag_baseline = next((r for r in rag_results if r["concurrency"] == 1), None)
    
    if no_rag_baseline and rag_baseline and no_rag_baseline["mean_ttft"] > 0 and rag_baseline["mean_ttft"] > 0:
        rag_overhead = ((rag_baseline["mean_ttft"] - no_rag_baseline["mean_ttft"]) / no_rag_baseline["mean_ttft"]) * 100
        print(f"• RAG overhead at C1: {rag_baseline['mean_ttft']:.2f}s vs {no_rag_baseline['mean_ttft']:.2f}s (+{rag_overhead:.1f}%)")
        
        if rag_overhead > 20:
            print("  → Confirms CPU-bound operations in RAG pipeline!")
        else:
            print("  → RAG overhead is minimal (unexpected)")
    
    # Concurrency scaling analysis
    for use_rag in [False, True]:
        config_results = [r for r in all_results if r["use_rag"] == use_rag and r["successful"] > 0]
        if len(config_results) >= 2:
            config_name = "RAG" if use_rag else "NO_RAG"
            baseline_result = min(config_results, key=lambda x: x["concurrency"])
            worst_result = max(config_results, key=lambda x: x["mean_ttft"])
            
            if baseline_result["mean_ttft"] > 0:
                scaling_degradation = ((worst_result["mean_ttft"] - baseline_result["mean_ttft"]) / baseline_result["mean_ttft"]) * 100
                print(f"• {config_name} scaling: {scaling_degradation:.1f}% degradation from C{baseline_result['concurrency']} to C{worst_result['concurrency']}")
                
                if use_rag and scaling_degradation > 50:
                    print("  → ThreadPool struggles with CPU-bound RAG operations!")
                elif not use_rag and scaling_degradation < 30:
                    print("  → ThreadPool handles I/O-bound operations well")
    
    print("\\n💡 Conclusions:")
    if rag_baseline and no_rag_baseline:
        if rag_baseline["mean_ttft"] > no_rag_baseline["mean_ttft"] * 1.2:
            print("✅ RAG introduces significant CPU-bound overhead")
            print("✅ This explains why ThreadPool doesn't scale linearly with RAG")
            print("💡 ProcessPoolExecutor should help with CPU-bound embedding tasks")
        else:
            print("⚠️  RAG overhead is smaller than expected")
            print("🤔 CPU bottleneck might be elsewhere or masked by I/O")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark-results-profiling/quick_cpu_test_{timestamp}.json"
    import os
    os.makedirs("benchmark-results-profiling", exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": all_results,
            "summary": {
                "no_rag_baseline": no_rag_baseline,
                "rag_baseline": rag_baseline
            }
        }, f, indent=2)
    
    print(f"\\n📁 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
Quick TTFT Benchmark - Simplified version for faster results
"""

import time
import json
import requests
import concurrent.futures
from datetime import datetime
import os

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001")

def test_single_request(request_id, concurrency_level):
    """Test a single streaming request and measure actual TTFT (time to first LLM token)"""
    payload = {
        "query": f"Explain quantum mechanics in simple terms (Request #{request_id})",
        "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2", 
        "llm_model": "stanford:gpt-4o",
        "prompt": "You are a helpful assistant.",
        "use_rag": False
    }
    
    start_time = time.time()
    first_response_time = None
    first_token_time = None
    
    try:
        response = requests.post(f"{BASE_URL}/ask/stream", json=payload, stream=True, timeout=60)
        
        if response.status_code != 200:
            return {"success": False, "ttft": None, "concurrency": concurrency_level, "error": f"HTTP {response.status_code}"}
        
        # Parse streaming response correctly
        for line in response.iter_lines():
            if line:
                current_time = time.time()
                
                # Mark first HTTP response time (not TTFT!)
                if first_response_time is None:
                    first_response_time = current_time
                
                if line.startswith(b'data: '):
                    data_str = line[6:].decode('utf-8')
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        data_type = data.get('type', '')
                        
                        # Only measure TTFT when we get actual LLM token
                        if data_type == 'token' and data.get('content'):
                            if first_token_time is None:
                                first_token_time = current_time
                                break  # We got our first token, stop measuring
                                
                    except json.JSONDecodeError:
                        continue
        
        if first_token_time is not None:
            ttft = first_token_time - start_time  # seconds
            time_to_first_response = first_response_time - start_time if first_response_time else 0
            
            return {
                "success": True, 
                "ttft": ttft,  # actual TTFT in seconds
                "time_to_first_response": time_to_first_response,  # HTTP response time
                "concurrency": concurrency_level
            }
        else:
            return {
                "success": False, 
                "ttft": None, 
                "concurrency": concurrency_level,
                "error": "No LLM tokens received"
            }
        
    except Exception as e:
        return {
            "success": False, 
            "ttft": None, 
            "concurrency": concurrency_level,
            "error": str(e)
        }

def test_concurrency(concurrency, requests_per_level=10):
    """Test specific concurrency level"""
    print(f"Testing concurrency {concurrency}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(test_single_request, i, concurrency) 
                  for i in range(requests_per_level)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    successful = [r for r in results if r["success"]]
    ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
    response_times = [r.get("time_to_first_response", 0) for r in successful]
    
    return {
        "concurrency": concurrency,
        "successful": len(successful),
        "total": requests_per_level,
        "ttfts": ttfts,
        "response_times": response_times,
        "mean_ttft": sum(ttfts) / len(ttfts) if ttfts else 0,
        "min_ttft": min(ttfts) if ttfts else 0,
        "max_ttft": max(ttfts) if ttfts else 0,
        "mean_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "failed_results": [r for r in results if not r["success"]]
    }

def main():
    print("⚡ Quick TTFT Benchmark")
    print("=" * 30)
    
    # Test different concurrency levels
    concurrency_levels = [1, 2, 5, 10, 20]
    results = []
    
    for concurrency in concurrency_levels:
        result = test_concurrency(concurrency, requests_per_level=10)
        results.append(result)
        
        print(f"  Concurrency {concurrency:2d}: "
              f"{result['successful']:2d}/{result['total']:2d} success, "
              f"TTFT: {result['mean_ttft']:6.1f}s avg "
              f"({result['min_ttft']:4.1f}-{result['max_ttft']:4.1f}s range), "
              f"HTTP: {result['mean_response_time']*1000:4.0f}ms")
    
    # Save quick results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"benchmark-results-streaming/quick_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save raw data
    with open(f"{results_dir}/quick_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create simple CSV
    csv_lines = ["concurrency,ttft_s,response_time_ms"]
    for result in results:
        for i, ttft in enumerate(result["ttfts"]):
            response_time = result["response_times"][i] if i < len(result["response_times"]) else 0
            csv_lines.append(f"{result['concurrency']},{ttft:.3f},{response_time*1000:.1f}")
    
    with open(f"{results_dir}/quick_results.csv", "w") as f:
        f.write("\n".join(csv_lines))
    
    print(f"\n✅ Quick results saved to: {results_dir}/")
    print("\n📊 Summary:")
    print("Concurrency | Success | Avg TTFT  | Range      | Failed Reasons")
    print("-" * 70)
    
    for result in results:
        success_rate = (result['successful'] / result['total']) * 100
        
        # Show failure reasons if any
        failure_reasons = []
        for failed in result["failed_results"]:
            if "error" in failed:
                failure_reasons.append(failed["error"])
        
        failure_summary = ", ".join(set(failure_reasons[:2])) if failure_reasons else "None"
        if len(failure_summary) > 20:
            failure_summary = failure_summary[:17] + "..."
            
        print(f"{result['concurrency']:10d} | {success_rate:6.1f}% | {result['mean_ttft']:8.1f}s | "
              f"{result['min_ttft']:4.1f}-{result['max_ttft']:4.1f}s | {failure_summary}")

if __name__ == "__main__":
    main()
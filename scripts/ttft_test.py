#!/usr/bin/env python3
"""
Time To First Token (TTFT) Benchmark Tool for OuRAGboros API

This tool measures the time from sending a request to receiving the first token
from the streaming API endpoint. It's designed to be lightweight, accurate, and
reusable across different deployments.

Usage:
    python ttft_test.py --endpoint http://localhost:8001 --samples 10 --concurrency 5
    python ttft_test.py --k8s-service ouragboros --namespace ouragboros --samples 20
"""

import asyncio
import argparse
import json
import statistics
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import httpx
from httpx_sse import aconnect_sse
import sys
import os

# Standard test configuration based on user requirements
DEFAULT_CONFIG = {
    "query": "What are the key principles of quantum mechanics?",
    "embedding_model": "huggingface:thellert/physbert_cased",
    "llm_model": "stanford:gpt-4.omini",  # Note: using gpt-4.omini (with typo) as per API error
    "prompt": "You are a helpful physics assistant.",
    "use_rag": True,
    "use_qdrant": True,
    "use_opensearch": False,
    "knowledge_base": "default",
    "max_documents": 3,
    "score_threshold": 0.0
}

class TTFTMeasurement:
    """Individual TTFT measurement result"""
    def __init__(self, ttft_seconds: float, success: bool, error: Optional[str] = None):
        self.ttft_seconds = ttft_seconds
        self.success = success
        self.error = error
        self.timestamp = time.time()

class TTFTBenchmark:
    """TTFT Benchmark tool for OuRAGboros streaming API"""
    
    def __init__(self, endpoint: str, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip('/') + '/ask/stream'
        self.timeout = timeout
        self.results: List[TTFTMeasurement] = []
        
    async def measure_single_ttft(self, session_id: int = 0) -> TTFTMeasurement:
        """Measure TTFT for a single request"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use regular streaming POST since server returns text/plain instead of text/event-stream
                async with client.stream(
                    "POST", 
                    self.endpoint,
                    json=DEFAULT_CONFIG,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status_code != 200:
                        return TTFTMeasurement(0, False, f"HTTP {response.status_code}: {await response.aread()}")
                    
                    async for chunk in response.aiter_lines():
                        if chunk and chunk.strip():
                            # Parse SSE format: "data: {json}"
                            if chunk.startswith("data: "):
                                json_data = chunk[6:]  # Remove "data: " prefix
                                try:
                                    data = json.loads(json_data)
                                    # Look for the first token event (API uses "content" field for token data)
                                    if data.get('type') == 'token' and data.get('content') is not None:
                                        ttft = time.time() - start_time
                                        return TTFTMeasurement(ttft, True)
                                        
                                except json.JSONDecodeError:
                                    continue
                    
                    # If we get here, no token was received
                    return TTFTMeasurement(0, False, "No token received")
                    
        except httpx.TimeoutException:
            return TTFTMeasurement(0, False, f"Timeout after {self.timeout}s")
        except Exception as e:
            return TTFTMeasurement(0, False, str(e))
    
    async def run_concurrent_benchmark(self, samples: int, concurrency: int) -> Dict[str, Any]:
        """Run concurrent TTFT measurements"""
        print(f"🚀 Starting TTFT benchmark: {samples} samples, {concurrency} concurrent")
        print(f"📡 Endpoint: {self.endpoint}")
        print(f"⚡ Model: {DEFAULT_CONFIG['llm_model']}")
        print(f"🧠 Embedding: {DEFAULT_CONFIG['embedding_model']}")
        print(f"📚 Knowledge Base: {DEFAULT_CONFIG['knowledge_base']}")
        print("")
        
        # Warm-up request
        print("🔥 Warming up...")
        warmup = await self.measure_single_ttft()
        if warmup.success:
            print(f"✅ Warm-up successful: {warmup.ttft_seconds:.3f}s")
        else:
            print(f"⚠️ Warm-up failed: {warmup.error}")
            return {"error": "Warm-up failed", "details": warmup.error}
        
        print(f"\n📊 Running {samples} measurements with concurrency {concurrency}...")
        
        # Run batches of concurrent requests
        all_results = []
        batch_size = concurrency
        batches = (samples + batch_size - 1) // batch_size
        
        for batch_num in range(batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, samples)
            batch_samples = batch_end - batch_start
            
            print(f"  Batch {batch_num + 1}/{batches}: {batch_samples} requests...", end="", flush=True)
            
            # Create concurrent tasks for this batch
            tasks = [
                self.measure_single_ttft(batch_start + i) 
                for i in range(batch_samples)
            ]
            
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            # Show progress
            successful = sum(1 for r in batch_results if r.success)
            print(f" {successful}/{batch_samples} successful")
            
            # Small delay between batches to avoid overwhelming the server
            if batch_num < batches - 1:
                await asyncio.sleep(0.5)
        
        self.results = all_results
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze TTFT measurement results"""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if not successful_results:
            return {
                "error": "No successful measurements",
                "total_samples": len(self.results),
                "successful": 0,
                "failed": len(failed_results),
                "failure_reasons": [r.error for r in failed_results]
            }
        
        ttft_values = [r.ttft_seconds for r in successful_results]
        
        # Calculate statistics
        stats = {
            "total_samples": len(self.results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "success_rate": len(successful_results) / len(self.results) * 100,
            
            "ttft_seconds": {
                "min": min(ttft_values),
                "max": max(ttft_values),
                "mean": statistics.mean(ttft_values),
                "median": statistics.median(ttft_values),
                "stdev": statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0,
                "p95": self._percentile(ttft_values, 95),
                "p99": self._percentile(ttft_values, 99)
            },
            
            "ttft_milliseconds": {
                "min": min(ttft_values) * 1000,
                "max": max(ttft_values) * 1000,
                "mean": statistics.mean(ttft_values) * 1000,
                "median": statistics.median(ttft_values) * 1000,
                "stdev": (statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0) * 1000,
                "p95": self._percentile(ttft_values, 95) * 1000,
                "p99": self._percentile(ttft_values, 99) * 1000
            }
        }
        
        if failed_results:
            failure_counts = {}
            for result in failed_results:
                error = result.error or "Unknown error"
                failure_counts[error] = failure_counts.get(error, 0) + 1
            stats["failure_reasons"] = failure_counts
        
        return stats
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c

def print_results(results: Dict[str, Any]):
    """Pretty print the benchmark results"""
    if "error" in results:
        print(f"\n❌ Benchmark failed: {results['error']}")
        if "details" in results:
            print(f"   Details: {results['details']}")
        return
    
    print(f"\n📊 TTFT Benchmark Results")
    print("=" * 50)
    
    print(f"Total Samples: {results['total_samples']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    if results['successful'] > 0:
        ttft_ms = results['ttft_milliseconds']
        print(f"\n⏱️  Time To First Token (milliseconds)")
        print(f"   Mean:   {ttft_ms['mean']:.1f} ms")
        print(f"   Median: {ttft_ms['median']:.1f} ms")
        print(f"   Min:    {ttft_ms['min']:.1f} ms")
        print(f"   Max:    {ttft_ms['max']:.1f} ms")
        print(f"   P95:    {ttft_ms['p95']:.1f} ms")
        print(f"   P99:    {ttft_ms['p99']:.1f} ms")
        print(f"   StdDev: {ttft_ms['stdev']:.1f} ms")
        
        # Performance assessment
        mean_ms = ttft_ms['mean']
        if mean_ms < 500:
            assessment = "🟢 Excellent"
        elif mean_ms < 800:
            assessment = "🟡 Good" 
        elif mean_ms < 1500:
            assessment = "🟠 Acceptable"
        else:
            assessment = "🔴 Needs Optimization"
        
        print(f"\n🎯 Performance Assessment: {assessment}")
        print(f"   (Industry good: <800ms, excellent: <500ms)")
    
    if results['failed'] > 0 and 'failure_reasons' in results:
        print(f"\n❌ Failure Breakdown:")
        for reason, count in results['failure_reasons'].items():
            print(f"   {reason}: {count}")

def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file"""
    # Add metadata
    results['metadata'] = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": results.get('endpoint', ''),
        "config": DEFAULT_CONFIG
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

async def main():
    parser = argparse.ArgumentParser(description='TTFT Benchmark for OuRAGboros API')
    parser.add_argument('--endpoint', default='http://localhost:8001', 
                       help='API endpoint (default: http://localhost:8001)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of TTFT measurements (default: 10)')
    parser.add_argument('--concurrency', type=int, default=5,
                       help='Concurrent requests (default: 5)')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--output', help='Output JSON file (optional)')
    parser.add_argument('--k8s-service', help='Kubernetes service name (implies port-forward)')
    parser.add_argument('--namespace', default='ouragboros', 
                       help='Kubernetes namespace (default: ouragboros)')
    
    args = parser.parse_args()
    
    # Handle k8s service with port-forward
    endpoint = args.endpoint
    if args.k8s_service:
        print(f"🚢 Kubernetes mode: {args.k8s_service} in namespace {args.namespace}")
        endpoint = "http://localhost:8001"  # Assume port-forward to 8001
        print(f"   Using endpoint: {endpoint}")
        print(f"   💡 Make sure to run: kubectl port-forward -n {args.namespace} svc/{args.k8s_service} 8001:8001")
        print()
    
    # Run benchmark
    benchmark = TTFTBenchmark(endpoint, args.timeout)
    results = await benchmark.run_concurrent_benchmark(args.samples, args.concurrency)
    results['endpoint'] = endpoint
    
    # Display results
    print_results(results)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    elif not "error" in results:
        # Auto-generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ttft_results_{timestamp}.json"
        save_results(results, output_file)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        sys.exit(1)
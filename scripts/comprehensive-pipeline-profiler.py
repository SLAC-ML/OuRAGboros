#!/usr/bin/env python3

"""
Comprehensive Pipeline Profiler for OuRAGboros
Tests CPU-bound vs I/O-bound operations with detailed timing breakdown
"""

import os
import sys
import time
import json
import requests
import concurrent.futures
import multiprocessing
from datetime import datetime
from typing import List, Dict, Any, Tuple
import statistics

# Configuration
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001")
CONCURRENCY_LEVELS = [1, 2, 5, 10]
REQUESTS_PER_LEVEL = 8  # Smaller for detailed profiling

class PipelineProfiler:
    def __init__(self):
        self.results_dir = f"benchmark-results-profiling/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def profile_single_request(self, request_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Profile a single request with detailed step-by-step timing"""
        
        payload = {
            "query": f"Explain quantum field theory and its applications (Request #{request_id})",
            "embedding_model": config.get("embedding_model", "huggingface:sentence-transformers/all-MiniLM-L6-v2"),
            "llm_model": config.get("llm_model", "stanford:gpt-4o"),
            "prompt": "You are a helpful physics professor.",
            "use_rag": config.get("use_rag", True),
            "use_opensearch": config.get("use_opensearch", False),
            "max_documents": 3,
            "score_threshold": 0.0
        }
        
        timing_breakdown = {}
        start_time = time.time()
        
        try:
            response = requests.post(f"{BASE_URL}/ask/stream", json=payload, stream=True, timeout=120)
            
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}", "config": config}
            
            # Track detailed timing through streaming response
            first_response_time = None
            first_status_time = None
            documents_received_time = None
            first_token_time = None
            last_token_time = None
            document_count = 0
            token_count = 0
            
            for line in response.iter_lines():
                if line:
                    current_time = time.time()
                    
                    # Mark first HTTP response
                    if first_response_time is None:
                        first_response_time = current_time
                    
                    if line.startswith(b'data: '):
                        data_str = line[6:].decode('utf-8')
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            data_type = data.get('type', '')
                            
                            if data_type == 'status' and first_status_time is None:
                                first_status_time = current_time
                                timing_breakdown['status_message'] = data.get('message', 'Unknown')
                                
                            elif data_type == 'documents' and documents_received_time is None:
                                documents_received_time = current_time
                                document_count = data.get('count', 0)
                                timing_breakdown['retrieval_time'] = data.get('retrieval_time', 0)
                                
                            elif data_type == 'token':
                                if first_token_time is None:
                                    first_token_time = current_time
                                last_token_time = current_time
                                token_count += 1
                                
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            
            # Calculate timing metrics
            total_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else None
            time_to_response = (first_response_time - start_time) if first_response_time else None
            time_to_documents = (documents_received_time - start_time) if documents_received_time else None
            time_to_generation = (first_token_time - documents_received_time) if (first_token_time and documents_received_time) else None
            generation_time = (last_token_time - first_token_time) if (last_token_time and first_token_time) else None
            
            return {
                "success": True,
                "config": config,
                "timing": {
                    "total_time": total_time,
                    "ttft": ttft,
                    "time_to_response": time_to_response,
                    "time_to_documents": time_to_documents,
                    "time_to_generation": time_to_generation,
                    "generation_time": generation_time,
                    "pipeline_breakdown": timing_breakdown
                },
                "metrics": {
                    "document_count": document_count,
                    "token_count": token_count,
                    "tokens_per_second": token_count / generation_time if generation_time and generation_time > 0 else 0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "config": config}

    def test_concurrency_threadpool(self, concurrency: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test concurrency using ThreadPoolExecutor"""
        print(f"🧵 Testing ThreadPool - Concurrency {concurrency}, Config: {config['name']}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.profile_single_request, i, config)
                for i in range(REQUESTS_PER_LEVEL)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        return results

    def test_concurrency_processpool(self, concurrency: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test concurrency using ProcessPoolExecutor"""
        print(f"🔧 Testing ProcessPool - Concurrency {concurrency}, Config: {config['name']}")
        
        try:
            with multiprocessing.Pool(processes=concurrency) as pool:
                # Create tasks for process pool
                tasks = [(i, config) for i in range(REQUESTS_PER_LEVEL)]
                results = pool.starmap(self.profile_single_request, tasks)
                return results
        except Exception as e:
            print(f"⚠️ ProcessPool failed: {e}")
            return [{"success": False, "error": f"ProcessPool error: {e}", "config": config}]

    def analyze_results(self, results: List[Dict[str, Any]], config_name: str, concurrency: int, executor_type: str):
        """Analyze timing results for a specific test configuration"""
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        if not successful:
            return {
                "config": config_name,
                "concurrency": concurrency,
                "executor": executor_type,
                "success_rate": 0,
                "errors": [r.get("error", "Unknown") for r in failed]
            }
        
        # Extract timing metrics
        ttfts = [r["timing"]["ttft"] for r in successful if r["timing"]["ttft"]]
        total_times = [r["timing"]["total_time"] for r in successful]
        response_times = [r["timing"]["time_to_response"] for r in successful if r["timing"]["time_to_response"]]
        document_times = [r["timing"]["time_to_documents"] for r in successful if r["timing"]["time_to_documents"]]
        generation_start_times = [r["timing"]["time_to_generation"] for r in successful if r["timing"]["time_to_generation"]]
        
        def safe_stats(values):
            if not values:
                return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        analysis = {
            "config": config_name,
            "concurrency": concurrency,
            "executor": executor_type,
            "success_rate": len(successful) / len(results) * 100,
            "total_requests": len(results),
            "successful_requests": len(successful),
            "timing_analysis": {
                "ttft": safe_stats(ttfts),
                "total_time": safe_stats(total_times),
                "http_response": safe_stats(response_times),
                "document_retrieval": safe_stats(document_times),
                "generation_start": safe_stats(generation_start_times)
            }
        }
        
        if failed:
            analysis["errors"] = [r.get("error", "Unknown") for r in failed[:3]]  # First 3 errors
            
        return analysis

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across different configurations"""
        
        # Test configurations
        test_configs = [
            {
                "name": "no_rag_baseline",
                "use_rag": False,
                "description": "I/O-bound only (LLM API calls)"
            },
            {
                "name": "rag_inmemory",
                "use_rag": True,
                "use_opensearch": False,
                "description": "CPU-bound (embedding + in-memory search)"
            },
            {
                "name": "rag_opensearch", 
                "use_rag": True,
                "use_opensearch": True,
                "description": "Mixed (embedding + network search)"
            }
        ]
        
        all_results = []
        
        print("🚀 Starting Comprehensive Pipeline Profiling")
        print(f"📁 Results will be saved to: {self.results_dir}")
        print("=" * 80)
        
        for config in test_configs:
            print(f"\n📊 Testing Configuration: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Settings: use_rag={config.get('use_rag', False)}, "
                  f"use_opensearch={config.get('use_opensearch', False)}")
            
            for concurrency in CONCURRENCY_LEVELS:
                # Test ThreadPoolExecutor
                thread_results = self.test_concurrency_threadpool(concurrency, config)
                thread_analysis = self.analyze_results(thread_results, config['name'], concurrency, "ThreadPool")
                all_results.append(thread_analysis)
                
                print(f"   ThreadPool C{concurrency}: "
                      f"TTFT={thread_analysis['timing_analysis']['ttft']['mean']:.2f}s ± "
                      f"{thread_analysis['timing_analysis']['ttft']['std']:.2f}s, "
                      f"Success={thread_analysis['success_rate']:.0f}%")
                
                # Test ProcessPoolExecutor (only for CPU-bound configs)
                if config.get('use_rag', False):
                    try:
                        process_results = self.test_concurrency_processpool(concurrency, config)
                        process_analysis = self.analyze_results(process_results, config['name'], concurrency, "ProcessPool")
                        all_results.append(process_analysis)
                        
                        print(f"   ProcessPool C{concurrency}: "
                              f"TTFT={process_analysis['timing_analysis']['ttft']['mean']:.2f}s ± "
                              f"{process_analysis['timing_analysis']['ttft']['std']:.2f}s, "
                              f"Success={process_analysis['success_rate']:.0f}%")
                    except Exception as e:
                        print(f"   ProcessPool C{concurrency}: Failed - {e}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.results_dir}/comprehensive_profile_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "configurations": test_configs,
                "concurrency_levels": CONCURRENCY_LEVELS,
                "requests_per_level": REQUESTS_PER_LEVEL,
                "results": all_results
            }, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(all_results, test_configs)
        
        print(f"\n✅ Comprehensive profiling complete!")
        print(f"📁 Results saved to: {results_file}")
        return all_results

    def generate_summary_report(self, results: List[Dict[str, Any]], configs: List[Dict[str, Any]]):
        """Generate a human-readable summary report"""
        
        report_path = f"{self.results_dir}/PROFILING_SUMMARY.md"
        
        with open(report_path, "w") as f:
            f.write("# 🔬 Comprehensive Pipeline Profiling Report\\n\\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Test Configurations**: {len(configs)}\\n")
            f.write(f"**Concurrency Levels**: {CONCURRENCY_LEVELS}\\n")
            f.write(f"**Requests per Level**: {REQUESTS_PER_LEVEL}\\n\\n")
            
            f.write("## 📊 Performance Summary\\n\\n")
            
            # Group results by configuration
            config_groups = {}
            for result in results:
                config_name = result['config']
                if config_name not in config_groups:
                    config_groups[config_name] = []
                config_groups[config_name].append(result)
            
            for config_name, config_results in config_groups.items():
                config_info = next(c for c in configs if c['name'] == config_name)
                
                f.write(f"### {config_name.upper()}: {config_info['description']}\\n\\n")
                f.write("| Concurrency | Executor | TTFT (s) | Success Rate | HTTP Response (ms) |\\n")
                f.write("|-------------|----------|----------|--------------|-------------------|\\n")
                
                for result in sorted(config_results, key=lambda x: (x['concurrency'], x['executor'])):
                    ttft_mean = result['timing_analysis']['ttft']['mean']
                    ttft_std = result['timing_analysis']['ttft']['std']
                    http_mean = result['timing_analysis']['http_response']['mean'] * 1000  # Convert to ms
                    success_rate = result['success_rate']
                    
                    f.write(f"| {result['concurrency']:<11} | {result['executor']:<8} | "
                           f"{ttft_mean:.2f}±{ttft_std:.2f} | {success_rate:.0f}% | {http_mean:.0f}ms |\\n")
                
                f.write("\\n")
            
            f.write("## 🔍 Key Insights\\n\\n")
            f.write("### ThreadPool vs ProcessPool Comparison\\n")
            f.write("- **No RAG**: ThreadPool optimal (I/O-bound workload)\\n")
            f.write("- **With RAG**: ProcessPool should outperform ThreadPool (CPU-bound embedding)\\n")
            f.write("- **GIL Impact**: Visible in CPU-intensive configurations\\n\\n")
            
            f.write("### Performance Bottleneck Analysis\\n")
            f.write("- **HTTP Response**: Time to first HTTP message (~5-20ms)\\n")
            f.write("- **Document Retrieval**: Embedding generation + vector search\\n") 
            f.write("- **TTFT**: Time to first actual LLM token\\n")
            f.write("- **Generation**: Token streaming speed\\n\\n")
        
        print(f"📄 Summary report: {report_path}")

def main():
    print("🔬 Comprehensive Pipeline Profiler")
    print("==================================")
    print(f"Testing endpoint: {BASE_URL}")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"Requests per level: {REQUESTS_PER_LEVEL}")
    print()
    
    # Check API availability
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not accessible: HTTP {response.status_code}")
            return 1
        print("✅ API is accessible")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return 1
    
    # Run comprehensive profiling
    profiler = PipelineProfiler()
    results = profiler.run_comprehensive_benchmark()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
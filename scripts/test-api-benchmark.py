#!/usr/bin/env python3

"""
API Benchmark Script for OuRAGboros
Tests the API endpoints directly without needing local Python dependencies
"""

import os
import sys
import time
import json
import requests
import concurrent.futures
from typing import List, Dict, Any

# Configuration
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://localhost:8001")
DOCKER_BASE_URL = os.environ.get("DOCKER_BASE_URL", "http://localhost:8001")

def test_api_health():
    """Test if API is accessible"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{LOCAL_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("  ✅ API is healthy and accessible")
            return True
        else:
            print(f"  ❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Failed to connect to API: {e}")
        return False

def test_mock_configuration():
    """Test different mock configurations"""
    print("\n🧪 Testing mock configurations...")
    
    configurations = [
        {"name": "baseline", "mock_embeddings": False, "mock_llm": False},
        {"name": "mock-embeddings", "mock_embeddings": True, "mock_llm": False},
        {"name": "mock-llm", "mock_embeddings": False, "mock_llm": True},
        {"name": "mock-both", "mock_embeddings": True, "mock_llm": True},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n  Testing: {config['name']}")
        
        # Set environment for mock configuration
        payload = {
            "query": "What is quantum mechanics?",
            "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "stanford:gpt-4o",
            "prompt": "You are a helpful assistant.",
            "use_rag": False,
            "use_mock_embeddings": config["mock_embeddings"],
            "use_mock_llm": config["mock_llm"]
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{LOCAL_BASE_URL}/ask", json=payload, timeout=30)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ Success in {elapsed:.2f}s")
                print(f"    📝 Response length: {len(result.get('answer', ''))} chars")
                
                # Check if mocks are working
                if config["mock_llm"] and "Mock response" in result.get('answer', ''):
                    print(f"    ✅ Mock LLM confirmed")
                elif config["mock_llm"]:
                    print(f"    ⚠️  Mock LLM may not be working")
                    
                results.append({
                    "config": config['name'],
                    "success": True,
                    "time": elapsed,
                    "response_length": len(result.get('answer', ''))
                })
            else:
                print(f"    ❌ Failed with status {response.status_code}")
                results.append({
                    "config": config['name'],
                    "success": False,
                    "error": f"Status {response.status_code}"
                })
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "config": config['name'],
                "success": False,
                "error": str(e)
            })
    
    return results

def test_streaming_endpoint():
    """Test the streaming endpoint"""
    print("\n🌊 Testing streaming endpoint...")
    
    payload = {
        "query": "Explain quantum entanglement",
        "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "stanford:gpt-4o-mini",
        "prompt": "You are a helpful physics teacher.",
        "use_rag": False
    }
    
    try:
        start_time = time.time()
        first_token_time = None
        tokens = []
        
        response = requests.post(
            f"{LOCAL_BASE_URL}/ask/stream",
            json=payload,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    if line.startswith(b'data: '):
                        data_str = line[6:].decode('utf-8')
                        if data_str != '[DONE]':
                            try:
                                data = json.loads(data_str)
                                token = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if token:
                                    tokens.append(token)
                            except json.JSONDecodeError:
                                pass
            
            total_time = time.time() - start_time
            print(f"  ✅ Streaming successful")
            print(f"  ⏱️  Time to first token: {first_token_time:.3f}s")
            print(f"  ⏱️  Total time: {total_time:.3f}s")
            print(f"  📝 Tokens received: {len(tokens)}")
            return True
        else:
            print(f"  ❌ Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_concurrent_requests(num_concurrent: int = 5):
    """Test concurrent request handling"""
    print(f"\n🔄 Testing {num_concurrent} concurrent requests...")
    
    def make_request(i: int) -> Dict[str, Any]:
        payload = {
            "query": f"Test query {i}: What is the speed of light?",
            "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "stanford:gpt-4o",
            "prompt": "You are a helpful assistant.",
            "use_rag": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(f"{LOCAL_BASE_URL}/ask", json=payload, timeout=30)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                return {
                    "request_id": i,
                    "success": True,
                    "time": elapsed,
                    "status": response.status_code
                }
            else:
                return {
                    "request_id": i,
                    "success": False,
                    "time": elapsed,
                    "status": response.status_code
                }
        except Exception as e:
            return {
                "request_id": i,
                "success": False,
                "time": time.time() - start_time,
                "error": str(e)
            }
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    avg_time = sum(r["time"] for r in results) / len(results)
    
    print(f"  ✅ Completed {successful}/{num_concurrent} requests")
    print(f"  ⏱️  Total time: {total_time:.2f}s")
    print(f"  ⏱️  Average time per request: {avg_time:.2f}s")
    print(f"  📊 Throughput: {num_concurrent/total_time:.2f} req/s")
    
    return results

def main():
    print("🚀 OuRAGboros API Benchmark")
    print("=" * 50)
    print(f"Testing endpoint: {LOCAL_BASE_URL}")
    print()
    
    # Test suite
    if not test_api_health():
        print("\n❌ API is not accessible. Please ensure services are running:")
        print("   docker compose up -d")
        return 1
    
    # Run tests
    mock_results = test_mock_configuration()
    streaming_success = test_streaming_endpoint()
    
    # Test different concurrency levels
    print("\n📊 Concurrency Scaling Test")
    print("-" * 40)
    for concurrent in [1, 5, 10, 20]:
        test_concurrent_requests(concurrent)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 50)
    
    print("\nMock Configuration Results:")
    for result in mock_results:
        status = "✅" if result.get("success") else "❌"
        print(f"  {result['config']:<20}: {status}")
    
    print(f"\nStreaming Endpoint: {'✅' if streaming_success else '❌'}")
    
    print("\n✅ Benchmark complete!")
    print("\n💡 Next steps:")
    print("  1. Review the results above")
    print("  2. Run systematic benchmark: ./scripts/test-concurrency-systematic.sh")
    print("  3. Test with K8s: kubectl port-forward -n ouragboros svc/ouragboros 8501:8501")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
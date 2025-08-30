#!/usr/bin/env python3
"""
Direct Stanford API Concurrency Test

This script tests Stanford AI API concurrency behavior by comparing:
1. Fresh client per request (current problematic approach) 
2. Shared client with connection pooling (proposed fix)

It measures TTFT for direct API calls without the OuRAGboros stack.
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict, Any
import json
import os
from datetime import datetime
import argparse

# Stanford AI API configuration
STANFORD_API_KEY = os.getenv('STANFORD_API_KEY')
STANFORD_BASE_URL = os.getenv('STANFORD_BASE_URL', 'https://aiapi-prod.stanford.edu/v1')

class StanfordAPITester:
    """Test Stanford API concurrency patterns"""
    
    def __init__(self):
        self.results = []
        
    async def test_fresh_clients_pattern(self, query: str, num_requests: int, concurrency: int):
        """Test with fresh aiohttp session per request (simulating current issue)"""
        
        async def single_request_fresh_client(session_id: int):
            start_time = time.time()
            
            # Create fresh session for each request (simulating the problem)
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {STANFORD_API_KEY}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': 'gpt-4.omini',
                    'messages': [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': query}
                    ],
                    'stream': True,
                    'max_tokens': 50
                }
                
                try:
                    async with session.post(
                        f'{STANFORD_BASE_URL}/chat/completions',
                        headers=headers,
                        json=payload
                    ) as response:
                        
                        if response.status != 200:
                            return {
                                'success': False,
                                'ttft': 0,
                                'error': f'HTTP {response.status}',
                                'session_id': session_id
                            }
                        
                        # Read streaming response to get TTFT
                        first_chunk_received = False
                        async for line in response.content:
                            if line and not first_chunk_received:
                                ttft = time.time() - start_time
                                first_chunk_received = True
                                return {
                                    'success': True, 
                                    'ttft': ttft,
                                    'session_id': session_id
                                }
                        
                        return {
                            'success': False,
                            'ttft': 0,
                            'error': 'No response received',
                            'session_id': session_id
                        }
                        
                except Exception as e:
                    return {
                        'success': False,
                        'ttft': 0,
                        'error': str(e),
                        'session_id': session_id
                    }
        
        print(f"🔴 Testing fresh clients: {num_requests} requests, {concurrency} concurrent")
        
        # Run requests in batches of 'concurrency'
        all_results = []
        batch_size = concurrency
        batches = (num_requests + batch_size - 1) // batch_size
        
        for batch_num in range(batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, num_requests)
            batch_requests = batch_end - batch_start
            
            print(f"  Batch {batch_num + 1}/{batches}: {batch_requests} requests...", end="", flush=True)
            
            tasks = [single_request_fresh_client(batch_start + i) for i in range(batch_requests)]
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            successful = sum(1 for r in batch_results if r['success'])
            print(f" {successful}/{batch_requests} successful")
        
        return all_results
    
    async def test_shared_client_pattern(self, query: str, num_requests: int, concurrency: int):
        """Test with shared aiohttp session (simulating the fix)"""
        
        async def single_request_shared_client(session: aiohttp.ClientSession, session_id: int):
            start_time = time.time()
            
            headers = {
                'Authorization': f'Bearer {STANFORD_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'gpt-4.omini', 
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': query}
                ],
                'stream': True,
                'max_tokens': 50
            }
            
            try:
                async with session.post(
                    f'{STANFORD_BASE_URL}/chat/completions',
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        return {
                            'success': False,
                            'ttft': 0,
                            'error': f'HTTP {response.status}',
                            'session_id': session_id
                        }
                    
                    # Read streaming response to get TTFT
                    first_chunk_received = False
                    async for line in response.content:
                        if line and not first_chunk_received:
                            ttft = time.time() - start_time
                            first_chunk_received = True
                            return {
                                'success': True,
                                'ttft': ttft,
                                'session_id': session_id
                            }
                    
                    return {
                        'success': False,
                        'ttft': 0,
                        'error': 'No response received',
                        'session_id': session_id
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'ttft': 0,
                    'error': str(e),
                    'session_id': session_id
                }
        
        print(f"🟢 Testing shared client: {num_requests} requests, {concurrency} concurrent")
        
        # Create shared session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=20,  # Max connections per host
            keepalive_timeout=30,  # Keep connections alive
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Run requests in batches of 'concurrency'
            all_results = []
            batch_size = concurrency
            batches = (num_requests + batch_size - 1) // batch_size
            
            for batch_num in range(batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, num_requests)
                batch_requests = batch_end - batch_start
                
                print(f"  Batch {batch_num + 1}/{batches}: {batch_requests} requests...", end="", flush=True)
                
                tasks = [single_request_shared_client(session, batch_start + i) for i in range(batch_requests)]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)
                
                successful = sum(1 for r in batch_results if r['success'])
                print(f" {successful}/{batch_requests} successful")
        
        return all_results
    
    def analyze_results(self, results: List[Dict[str, Any]], test_name: str):
        """Analyze and print results"""
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if not successful_results:
            print(f"\n❌ {test_name}: No successful requests")
            return None
        
        ttft_values = [r['ttft'] for r in successful_results]
        
        analysis = {
            'test_name': test_name,
            'total_requests': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'ttft_stats': {
                'mean_ms': statistics.mean(ttft_values) * 1000,
                'median_ms': statistics.median(ttft_values) * 1000,
                'min_ms': min(ttft_values) * 1000,
                'max_ms': max(ttft_values) * 1000,
                'p95_ms': self._percentile(ttft_values, 95) * 1000,
                'p99_ms': self._percentile(ttft_values, 99) * 1000,
                'stdev_ms': statistics.stdev(ttft_values) * 1000 if len(ttft_values) > 1 else 0
            }
        }
        
        print(f"\n📊 {test_name} Results:")
        print(f"  Total: {analysis['total_requests']}, Success: {analysis['successful']}, Failed: {analysis['failed']}")
        print(f"  Success Rate: {analysis['success_rate']:.1f}%")
        print(f"  TTFT Mean: {analysis['ttft_stats']['mean_ms']:.1f}ms")
        print(f"  TTFT P99:  {analysis['ttft_stats']['p99_ms']:.1f}ms")
        
        if failed_results:
            error_counts = {}
            for r in failed_results:
                error = r.get('error', 'Unknown')
                error_counts[error] = error_counts.get(error, 0) + 1
            print(f"  Errors: {error_counts}")
        
        return analysis
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c

async def main():
    parser = argparse.ArgumentParser(description='Stanford API Concurrency Test')
    parser.add_argument('--query', default='What is 2+2?', help='Test query')
    parser.add_argument('--requests', type=int, default=10, help='Number of requests per test')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrent requests')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    if not STANFORD_API_KEY:
        print("❌ STANFORD_API_KEY environment variable not set")
        return
    
    tester = StanfordAPITester()
    
    print("🧪 Stanford API Concurrency Test")
    print(f"Query: {args.query}")
    print(f"Requests: {args.requests}, Concurrency: {args.concurrency}")
    print("")
    
    # Test 1: Fresh clients (current problematic approach)
    fresh_results = await tester.test_fresh_clients_pattern(
        args.query, args.requests, args.concurrency
    )
    fresh_analysis = tester.analyze_results(fresh_results, "Fresh Clients")
    
    print("\n" + "="*50)
    
    # Test 2: Shared client (proposed fix)
    shared_results = await tester.test_shared_client_pattern(
        args.query, args.requests, args.concurrency
    )
    shared_analysis = tester.analyze_results(shared_results, "Shared Client")
    
    # Comparison
    if fresh_analysis and shared_analysis:
        fresh_p99 = fresh_analysis['ttft_stats']['p99_ms']
        shared_p99 = shared_analysis['ttft_stats']['p99_ms']
        improvement = fresh_p99 / shared_p99 if shared_p99 > 0 else 0
        
        print(f"\n🎯 COMPARISON:")
        print(f"  Fresh Client P99:  {fresh_p99:.1f}ms")  
        print(f"  Shared Client P99: {shared_p99:.1f}ms")
        print(f"  Improvement: {improvement:.1f}x faster with shared client")
        
        if improvement > 2:
            print(f"  🎉 SIGNIFICANT IMPROVEMENT! Shared client is much faster.")
        elif improvement > 1.5:
            print(f"  ✅ GOOD IMPROVEMENT! Shared client helps performance.")
        else:
            print(f"  🤔 MARGINAL DIFFERENCE: May need more investigation.")
    
    # Save results
    if args.output:
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'query': args.query,
                'requests': args.requests, 
                'concurrency': args.concurrency
            },
            'fresh_client_results': fresh_analysis,
            'shared_client_results': shared_analysis
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
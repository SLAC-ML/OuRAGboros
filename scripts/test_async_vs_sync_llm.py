#!/usr/bin/env python3
"""
Test Async vs Sync LLM Client Behavior

This tests whether the bottleneck is in using synchronous LangChain OpenAI client
in an async context vs using a native async OpenAI client.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import os
import json
from datetime import datetime

# Test both approaches
try:
    from langchain_openai import OpenAI as LangChainOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    langchain_available = True
except ImportError:
    langchain_available = False
    print("⚠️ LangChain not available, testing native OpenAI only")

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False
    print("❌ OpenAI client not available")

STANFORD_API_KEY = os.getenv('STANFORD_API_KEY')
STANFORD_BASE_URL = os.getenv('STANFORD_BASE_URL', 'https://aiapi-prod.stanford.edu/v1')

class LLMConcurrencyTester:
    def __init__(self):
        self.results = {}
    
    async def test_langchain_sync_client(self, query: str, num_requests: int, concurrency: int):
        """Test LangChain OpenAI client (synchronous) - current approach"""
        if not langchain_available:
            return None
        
        def single_langchain_request(request_id: int):
            start_time = time.time()
            
            try:
                # This is what we currently do - create fresh client each time
                llm = LangChainOpenAI(
                    openai_api_key=STANFORD_API_KEY,
                    openai_api_base=STANFORD_BASE_URL,
                    model_name='gpt-4.omini',
                    max_tokens=50
                )
                
                # This streaming call is SYNCHRONOUS and blocks!
                stream = llm.stream([
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=query)
                ])
                
                # Get first token
                for token in stream:
                    ttft = time.time() - start_time
                    return {
                        'success': True,
                        'ttft': ttft,
                        'request_id': request_id
                    }
                
                return {
                    'success': False,
                    'ttft': 0,
                    'error': 'No tokens received',
                    'request_id': request_id
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'ttft': 0,
                    'error': str(e),
                    'request_id': request_id
                }
        
        print(f"🔴 Testing LangChain sync client: {num_requests} requests, {concurrency} concurrent")
        
        # Run in batches - but this will still block on sync calls!
        all_results = []
        batch_size = concurrency
        batches = (num_requests + batch_size - 1) // batch_size
        
        for batch_num in range(batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, num_requests)
            batch_requests = batch_end - batch_start
            
            print(f"  Batch {batch_num + 1}/{batches}: {batch_requests} requests...", end="", flush=True)
            
            # Even though we're using asyncio.gather, the sync LangChain calls will block!
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, single_langchain_request, batch_start + i)
                for i in range(batch_requests)
            ]
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            successful = sum(1 for r in batch_results if r['success'])
            print(f" {successful}/{batch_requests} successful")
        
        return all_results
    
    async def test_native_async_client(self, query: str, num_requests: int, concurrency: int):
        """Test native OpenAI async client - proposed fix"""
        if not openai_available:
            return None
        
        # Create shared async client (connection pooling!)
        async_client = openai.AsyncOpenAI(
            api_key=STANFORD_API_KEY,
            base_url=STANFORD_BASE_URL
        )
        
        async def single_async_request(request_id: int):
            start_time = time.time()
            
            try:
                # Native async streaming call
                stream = await async_client.chat.completions.create(
                    model='gpt-4.omini',
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}
                    ],
                    stream=True,
                    max_tokens=50
                )
                
                # Get first chunk - truly async!
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        ttft = time.time() - start_time
                        return {
                            'success': True,
                            'ttft': ttft,
                            'request_id': request_id
                        }
                
                return {
                    'success': False,
                    'ttft': 0,
                    'error': 'No content received',
                    'request_id': request_id
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'ttft': 0,
                    'error': str(e),
                    'request_id': request_id
                }
        
        print(f"🟢 Testing native async client: {num_requests} requests, {concurrency} concurrent")
        
        try:
            # Run in batches with true async concurrency
            all_results = []
            batch_size = concurrency
            batches = (num_requests + batch_size - 1) // batch_size
            
            for batch_num in range(batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, num_requests)
                batch_requests = batch_end - batch_start
                
                print(f"  Batch {batch_num + 1}/{batches}: {batch_requests} requests...", end="", flush=True)
                
                # Truly concurrent async calls!
                tasks = [single_async_request(batch_start + i) for i in range(batch_requests)]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)
                
                successful = sum(1 for r in batch_results if r['success'])
                print(f" {successful}/{batch_requests} successful")
            
            return all_results
        
        finally:
            await async_client.close()
    
    def analyze_results(self, results: List[Dict[str, Any]], test_name: str):
        """Analyze results"""
        if not results:
            print(f"\n❌ {test_name}: No results")
            return None
        
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if not successful_results:
            print(f"\n❌ {test_name}: No successful requests")
            for r in failed_results[:3]:  # Show first 3 errors
                print(f"   Error: {r.get('error', 'Unknown')}")
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
    if not STANFORD_API_KEY:
        print("❌ STANFORD_API_KEY environment variable not set")
        return
    
    tester = LLMConcurrencyTester()
    
    print("🧪 Async vs Sync LLM Client Test")
    print("Query: What is 2+2?")
    print("Testing concurrent behavior of LangChain sync vs Native async clients")
    print("")
    
    # Test parameters
    num_requests = 10
    concurrency = 5
    
    # Test 1: LangChain synchronous client (current approach)
    langchain_results = await tester.test_langchain_sync_client("What is 2+2?", num_requests, concurrency)
    langchain_analysis = tester.analyze_results(langchain_results, "LangChain Sync Client")
    
    print("\n" + "="*50)
    
    # Test 2: Native async OpenAI client (proposed fix)
    async_results = await tester.test_native_async_client("What is 2+2?", num_requests, concurrency)
    async_analysis = tester.analyze_results(async_results, "Native Async Client")
    
    # Comparison
    if langchain_analysis and async_analysis:
        langchain_p99 = langchain_analysis['ttft_stats']['p99_ms']
        async_p99 = async_analysis['ttft_stats']['p99_ms']
        improvement = langchain_p99 / async_p99 if async_p99 > 0 else 0
        
        print(f"\n🎯 COMPARISON:")
        print(f"  LangChain Sync P99:  {langchain_p99:.1f}ms")  
        print(f"  Native Async P99:    {async_p99:.1f}ms")
        print(f"  Improvement: {improvement:.1f}x faster with native async client")
        
        if improvement > 3:
            print(f"  🎉 MAJOR IMPROVEMENT! Async client is much faster.")
        elif improvement > 2:
            print(f"  ✅ SIGNIFICANT IMPROVEMENT! Async client helps a lot.")
        elif improvement > 1.5:
            print(f"  ✅ GOOD IMPROVEMENT! Async client helps performance.")
        else:
            print(f"  🤔 MARGINAL DIFFERENCE: Both approaches similar.")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_config': {
            'query': "What is 2+2?",
            'requests': num_requests,
            'concurrency': concurrency
        },
        'langchain_results': langchain_analysis,
        'async_results': async_analysis
    }
    
    with open('async_vs_sync_llm_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: async_vs_sync_llm_test.json")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
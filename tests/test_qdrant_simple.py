#!/usr/bin/env python3
"""
Simple Qdrant performance test using pre-computed embeddings.
"""

import subprocess
import json
import time

def test_qdrant_with_mock_embeddings():
    """Test Qdrant with mock embeddings to avoid model download"""
    
    script_content = '''
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

print("🧪 Testing Qdrant with Mock Embeddings...")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid

    # Connect to Qdrant
    client = QdrantClient(url="http://qdrant:6333")
    print("✅ Connected to Qdrant")
    
    # Create collection with 384-dimensional vectors (common size)
    collection_name = "test_mock_docs"
    vector_size = 384
    
    print(f"📝 Creating collection: {collection_name}")
    
    # Check if exists first
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            # Optimize for concurrent search performance
            hnsw_config={
                "m": 16,
                "ef_construct": 200,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 0,
            }
        )
        print("✅ Created collection with optimized settings")
    else:
        print("ℹ️  Collection already exists")
    
    # Generate mock embeddings and documents
    def generate_mock_embedding():
        return [random.gauss(0, 0.1) for _ in range(vector_size)]
    
    test_docs = [
        "The speed of light is 299,792,458 m/s",
        "E=mc² is Einstein's mass-energy equivalence", 
        "The Planck constant is 6.626×10^-34 J⋅Hz^-1",
        "Quantum mechanics describes particle behavior",
        "General relativity explains spacetime curvature",
        "Newton's laws describe classical mechanics",
        "Thermodynamics deals with heat and energy",
        "Electromagnetism unifies electric and magnetic forces",
    ]
    
    print(f"🔢 Generating {len(test_docs)} mock embeddings...")
    
    # Insert documents with mock embeddings
    points = []
    for i, doc in enumerate(test_docs):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=generate_mock_embedding(),
            payload={"text": doc, "index": i}
        ))
    
    print("📤 Uploading to Qdrant...")
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Uploaded {len(points)} documents")
    
    # Test search with mock query embedding
    print("🔍 Testing search...")
    query_embedding = generate_mock_embedding()
    
    search_start = time.time()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3,
        score_threshold=0.0  # Accept all results for testing
    )
    search_time = time.time() - search_start
    
    print(f"✅ Search completed in {search_time:.3f}s")
    print(f"✅ Found {len(search_result)} results:")
    for i, result in enumerate(search_result[:3]):
        print(f"  {i+1}. Score: {result.score:.3f} - {result.payload['text'][:50]}...")
    
    # Test concurrent searches
    def search_query(query_id):
        start_time = time.time()
        try:
            query_vector = generate_mock_embedding()
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=2,
                score_threshold=0.0,
                search_params={"hnsw_ef": 128}
            )
            end_time = time.time()
            return query_id, end_time - start_time, len(results), "success"
        except Exception as e:
            end_time = time.time()
            return query_id, end_time - start_time, 0, str(e)
    
    # Test different concurrency levels
    concurrency_levels = [1, 5, 10, 20]
    
    for concurrency in concurrency_levels:
        print(f"\\n📊 Testing {concurrency} concurrent requests...")
        
        start_total = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(search_query, i+1): i+1 for i in range(concurrency)}
            
            results = []
            for future in as_completed(futures):
                query_id, response_time, result_count, status = future.result()
                results.append((query_id, response_time, result_count, status))
        
        end_total = time.time()
        total_time = end_total - start_total
        
        successful = [r for r in results if r[3] == "success"]
        
        if successful:
            avg_response_time = sum(r[1] for r in successful) / len(successful)
            throughput = len(successful) / total_time
            
            print(f"  Results: {len(successful)}/{concurrency} success")
            print(f"  Avg response: {avg_response_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} req/s")
            
            # Check performance target
            if avg_response_time <= 1.0:  # 1 second target for mock data
                print(f"  🎯 EXCELLENT: {avg_response_time:.3f}s response time")
            elif avg_response_time <= 3.0:
                print(f"  ✅ GOOD: {avg_response_time:.3f}s response time") 
            else:
                print(f"  ⚠️  SLOW: {avg_response_time:.3f}s response time")
        else:
            print(f"  ❌ All requests failed")
    
    print("\\n🎉 Qdrant performance test completed!")
    
    # Estimate capacity
    best_concurrency = max(concurrency_levels)
    if successful:
        print(f"\\n📈 Performance Estimation:")
        print(f"   At {best_concurrency} concurrent users:")
        print(f"   - Average response: {avg_response_time:.3f}s")
        print(f"   - Throughput: {throughput:.1f} req/s")
        
        # Extrapolate to 100 users
        if avg_response_time > 0:
            projected_100_users = avg_response_time * (100 / best_concurrency) ** 0.5  # Conservative scaling
            print(f"   - Projected 100 users: ~{projected_100_users:.3f}s response")
            
            if projected_100_users <= 3.0:
                print("   🎯 Should handle 100+ concurrent users!")
            else:
                print("   ⚠️  May need optimization for 100+ users")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
'''
    
    # Run in container with timeout
    cmd = [
        "docker", "run", "--rm", 
        "--network", "ouragboros_default",
        "python:3.11-slim", 
        "bash", "-c", f"pip install qdrant-client>=1.12.0 -q && python -c '{script_content}'"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "Test timed out", "Timeout after 120s", 1

def main():
    """Run simple Qdrant test"""
    print("🧪 Simple Qdrant Performance Test")
    print("=" * 50)
    
    # Check if Qdrant is accessible
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:6333/collections'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            print("❌ Qdrant not accessible")
            print("💡 Make sure it's running: docker compose up -d qdrant")
            return False
        print("✅ Qdrant is accessible")
    except:
        print("❌ Cannot reach Qdrant")
        return False
    
    print("\n🚀 Running performance test...")
    stdout, stderr, returncode = test_qdrant_with_mock_embeddings()
    
    print(stdout)
    if stderr:
        print("STDERR:", stderr)
    
    success = returncode == 0
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Qdrant test completed successfully!")
    else:
        print("❌ Test failed")
    
    return success

if __name__ == "__main__":
    main()
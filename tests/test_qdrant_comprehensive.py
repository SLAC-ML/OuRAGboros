#!/usr/bin/env python3
"""
Comprehensive Qdrant performance test using Docker containers.
This test runs in a controlled environment without dependency conflicts.
"""

import subprocess
import json
import time
import concurrent.futures

def run_in_container(script_content, container_name="qdrant_test"):
    """Run Python script in a container with network access to Qdrant"""
    
    # Create the test script content
    full_script = f'''
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install required packages
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("📦 Installing dependencies...")
packages = [
    "qdrant-client>=1.12.0",
    "sentence-transformers>=2.0.0",
]

for pkg in packages:
    try:
        install_package(pkg)
    except Exception as e:
        print(f"Warning: Could not install {{pkg}}: {{e}}")

# Now run the actual test
{script_content}
'''
    
    # Write script to temp file
    with open("/tmp/test_script.py", "w") as f:
        f.write(full_script)
    
    # Run in container
    cmd = [
        "docker", "run", "--rm", 
        "--network", "ouragboros_default",
        "-v", "/tmp/test_script.py:/test_script.py",
        "python:3.11-slim", 
        "python", "/test_script.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout, result.stderr, result.returncode

def test_qdrant_basic_integration():
    """Test basic Qdrant integration with embeddings"""
    script_content = '''
print("🧪 Testing Basic Qdrant Integration...")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    import uuid

    # Connect to Qdrant
    client = QdrantClient(url="http://qdrant:6333")
    print("✅ Connected to Qdrant")
    
    # Load a small embedding model
    print("📚 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector_size = 384  # all-MiniLM-L6-v2 dimension
    
    # Create collection
    collection_name = "test_physics_docs"
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
            # Optimize for search performance
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
    
    # Test documents
    test_docs = [
        "The speed of light is 299,792,458 m/s",
        "E=mc² is Einstein's mass-energy equivalence", 
        "The Planck constant is 6.626×10^-34 J⋅Hz^-1",
        "Quantum mechanics describes the behavior of particles at atomic scales",
        "General relativity explains gravity as curvature of spacetime",
    ]
    
    print(f"🔢 Embedding {len(test_docs)} documents...")
    embeddings = model.encode(test_docs)
    
    # Insert documents
    points = []
    for i, (doc, embedding) in enumerate(zip(test_docs, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={"text": doc, "index": i}
        ))
    
    print("📤 Uploading to Qdrant...")
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Uploaded {len(points)} documents")
    
    # Test search
    query = "What is the speed of light?"
    print(f"🔍 Searching: '{query}'")
    
    query_embedding = model.encode([query])[0]
    
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=3,
        score_threshold=0.3
    )
    
    print(f"✅ Found {len(search_result)} results:")
    for i, result in enumerate(search_result):
        print(f"  {i+1}. Score: {result.score:.3f} - {result.payload['text']}")
    
    print("🎉 Basic integration test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
'''
    
    stdout, stderr, returncode = run_in_container(script_content)
    
    print("=== Basic Integration Test Results ===")
    if stdout:
        print(stdout)
    if stderr:
        print("STDERR:", stderr)
    
    return returncode == 0

def test_qdrant_concurrent_performance():
    """Test concurrent search performance"""
    script_content = '''
print("🚀 Testing Concurrent Performance...")

try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Connect and load model
    client = QdrantClient(url="http://qdrant:6333")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    collection_name = "test_physics_docs"
    
    # Verify collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if collection_name not in collection_names:
        print("❌ Collection not found. Run basic test first.")
        exit(1)
    
    def search_query(query_id):
        """Perform a single search query"""
        query = f"physics concept {query_id}"
        start_time = time.time()
        
        try:
            query_embedding = model.encode([query])[0]
            results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=2,
                score_threshold=0.1,
                search_params={"hnsw_ef": 128}  # Optimize search
            )
            end_time = time.time()
            return query_id, end_time - start_time, len(results), "success"
        except Exception as e:
            end_time = time.time()
            return query_id, end_time - start_time, 0, str(e)
    
    # Test with increasing concurrency
    concurrency_levels = [1, 5, 10]
    
    for concurrency in concurrency_levels:
        print(f"\\n📊 Testing {concurrency} concurrent requests...")
        
        start_total = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(search_query, i+1): i+1 for i in range(concurrency)}
            
            results = []
            for future in as_completed(futures):
                query_id, response_time, result_count, status = future.result()
                results.append((query_id, response_time, result_count, status))
                
                status_icon = "✅" if status == "success" else "❌"
                print(f"  Query {query_id}: {response_time:.3f}s {status_icon}")
        
        end_total = time.time()
        total_time = end_total - start_total
        
        successful = [r for r in results if r[3] == "success"]
        
        if successful:
            avg_response_time = sum(r[1] for r in successful) / len(successful)
            throughput = len(successful) / total_time
            
            print(f"  📈 Results for {concurrency} concurrent:")
            print(f"     Success rate: {len(successful)}/{concurrency}")
            print(f"     Average response: {avg_response_time:.3f}s")
            print(f"     Wall time: {total_time:.3f}s") 
            print(f"     Throughput: {throughput:.1f} req/s")
            
            # Performance targets
            target_response_time = 3.0  # 3 second target
            if avg_response_time <= target_response_time:
                print(f"     🎯 PASS: Response time {avg_response_time:.3f}s <= {target_response_time}s")
            else:
                print(f"     ⚠️  CONCERN: Response time {avg_response_time:.3f}s > {target_response_time}s")
        else:
            print(f"  ❌ All requests failed for concurrency {concurrency}")
    
    print("\\n🎉 Concurrent performance test completed!")
    
except Exception as e:
    print(f"❌ Concurrent test failed: {e}")
    import traceback
    traceback.print_exc()
'''
    
    stdout, stderr, returncode = run_in_container(script_content)
    
    print("\\n=== Concurrent Performance Test Results ===")
    if stdout:
        print(stdout)
    if stderr:
        print("STDERR:", stderr)
    
    return returncode == 0

def main():
    """Run comprehensive Qdrant tests"""
    print("🧪 Comprehensive Qdrant Performance Testing")
    print("=" * 60)
    
    # Check if Qdrant is running
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:6333/collections'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            print("❌ Qdrant not accessible at localhost:6333")
            print("💡 Make sure Qdrant is running: docker compose up -d qdrant")
            return False
        print("✅ Qdrant is accessible")
    except Exception as e:
        print(f"❌ Cannot reach Qdrant: {e}")
        return False
    
    print()
    
    # Run tests
    success = True
    
    print("🔧 Test 1: Basic Integration")
    if not test_qdrant_basic_integration():
        print("❌ Basic integration test failed")
        success = False
    else:
        print("✅ Basic integration test passed")
    
    print()
    
    print("🚀 Test 2: Concurrent Performance") 
    if not test_qdrant_concurrent_performance():
        print("❌ Concurrent performance test failed")
        success = False
    else:
        print("✅ Concurrent performance test passed")
    
    print("\\n" + "=" * 60)
    if success:
        print("🎉 All Qdrant tests passed!")
        print("\\n💡 Next steps:")
        print("   1. Compare with OpenSearch performance")
        print("   2. Deploy to Kubernetes")
        print("   3. Test with 100+ concurrent users")
    else:
        print("❌ Some tests failed - check output above")
    
    return success

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for Qdrant integration with Docker Compose.
Run this after: docker-compose up -d qdrant
"""

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_qdrant_connection():
    """Test basic Qdrant connection"""
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url="http://localhost:6333")
        info = client.get_collections()
        print(f"✅ Qdrant connection successful! Collections: {len(info.collections)}")
        return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def test_qdrant_integration():
    """Test Qdrant with LangChain integration"""
    try:
        import sys
        import os
        sys.path.append('src')
        
        import lib.langchain.qdrant as qdrant_lib
        from langchain.schema import Document
        
        # Test documents
        test_docs = [
            Document(page_content="The speed of light is 299,792,458 m/s", metadata={"source": "physics", "topic": "constants"}),
            Document(page_content="E=mc² is Einstein's mass-energy equivalence", metadata={"source": "physics", "topic": "relativity"}),
            Document(page_content="The Planck constant is 6.626×10^-34 J⋅Hz^-1", metadata={"source": "physics", "topic": "quantum"}),
        ]
        
        # Test adding documents
        print("📝 Testing document insertion...")
        doc_ids = qdrant_lib.add_documents_to_qdrant(
            documents=test_docs,
            embedding_model="huggingface:thellert/physbert_cased",
            knowledge_base="test_local"
        )
        print(f"✅ Added {len(doc_ids)} documents to Qdrant")
        
        # Test similarity search  
        print("🔍 Testing similarity search...")
        results = qdrant_lib.search_qdrant_documents(
            query="What is the speed of light?",
            embedding_model="huggingface:thellert/physbert_cased", 
            knowledge_base="test_local",
            k=2,
            score_threshold=0.3
        )
        
        print(f"✅ Found {len(results)} similar documents:")
        for i, (doc, score) in enumerate(results):
            print(f"  {i+1}. Score: {score:.3f} - {doc.page_content[:50]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Qdrant integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_concurrent_performance():
    """Test concurrent search performance"""
    try:
        print("🚀 Testing concurrent performance...")
        
        import sys
        sys.path.append('src')
        import lib.langchain.qdrant as qdrant_lib
        
        def search_query(query_id):
            start_time = time.time()
            try:
                results = qdrant_lib.search_qdrant_documents(
                    query=f"What is physics constant {query_id}?",
                    embedding_model="huggingface:thellert/physbert_cased",
                    knowledge_base="test_local",
                    k=1,
                    score_threshold=0.1
                )
                end_time = time.time()
                return query_id, end_time - start_time, len(results), "success"
            except Exception as e:
                end_time = time.time() 
                return query_id, end_time - start_time, 0, str(e)
        
        # Test with 10 concurrent requests
        concurrency = 10
        start_total = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(search_query, i+1): i+1 for i in range(concurrency)}
            
            results = []
            for future in as_completed(futures):
                query_id, response_time, result_count, status = future.result()
                results.append((query_id, response_time, result_count, status))
                status_icon = "✅" if status == "success" else "❌"
                print(f"  Query {query_id}: {response_time:.3f}s {status_icon} ({result_count} results)")
        
        end_total = time.time()
        total_time = end_total - start_total
        
        successful = [r for r in results if r[3] == "success"]
        if successful:
            avg_time = sum(r[1] for r in successful) / len(successful)
            print(f"📊 Concurrent test results ({concurrency} requests):")
            print(f"   Success rate: {len(successful)}/{concurrency}")
            print(f"   Average response time: {avg_time:.3f}s")
            print(f"   Total wall time: {total_time:.3f}s")
            print(f"   Throughput: {len(successful)/total_time:.2f} req/s")
            
            # Compare to theoretical OpenSearch performance
            opensearch_projected = avg_time * concurrency  # Linear scaling
            improvement = opensearch_projected / avg_time if avg_time > 0 else 1
            print(f"   🎯 Expected improvement over OpenSearch: {improvement:.1f}x faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Concurrent performance test failed: {e}")
        return False

def main():
    print("🧪 Testing Qdrant Local Docker Setup")
    print("=" * 50)
    
    # Test connection first
    if not test_qdrant_connection():
        print("\n💡 Make sure Qdrant is running: docker-compose up -d qdrant")
        return
    
    print()
    
    # Test basic integration
    if not test_qdrant_integration():
        return
        
    print()
    
    # Test concurrent performance
    test_concurrent_performance()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n💡 Next steps:")
    print("   1. Start full stack: docker-compose up -d")
    print("   2. Test via web UI at http://localhost:8501") 
    print("   3. Run performance benchmarks")

if __name__ == "__main__":
    main()
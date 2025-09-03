#!/usr/bin/env python3

"""
Mock System Validation Script
Tests that USE_MOCK_EMBEDDINGS and USE_MOCK_LLM work correctly
"""

import os
import sys
import time
import requests
import json

def test_mock_embeddings():
    """Test that mock embeddings work and are fast"""
    print("🧪 Testing mock embeddings...")
    
    # Set mock embeddings environment variable
    os.environ['USE_MOCK_EMBEDDINGS'] = 'true'
    os.environ['USE_MOCK_LLM'] = 'false'
    
    # Import embedding function after setting env vars
    try:
        sys.path.append('src')
        from lib.langchain import embeddings
        
        # Clear any existing cache to force reload
        embeddings.clear_embedding_cache()
        
        # Test embedding generation speed
        start_time = time.time()
        embedding_model = "huggingface:thellert/physbert_cased"
        embed_instance = embeddings.get_embedding(embedding_model)
        
        # Generate embedding for a sample query
        test_query = "What is quantum mechanics?"
        embedding_vector = embed_instance.embed_query(test_query)
        
        embed_time = time.time() - start_time
        
        print(f"  ✅ Mock embedding generated in {embed_time:.3f}s")
        print(f"  📏 Vector dimension: {len(embedding_vector)}")
        print(f"  🎯 Vector type: {type(embedding_vector)}")
        
        # Validate it's actually using mock
        if embed_time < 0.1:  # Mock should be very fast
            print("  ✅ Mock embeddings are working (very fast generation)")
            return True
        else:
            print("  ⚠️  Embedding took longer than expected - might not be using mock")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing mock embeddings: {e}")
        return False

def test_mock_llm():
    """Test that mock LLM works and is fast"""
    print("\n🤖 Testing mock LLM...")
    
    # Set mock LLM environment variable  
    os.environ['USE_MOCK_EMBEDDINGS'] = 'false'
    os.environ['USE_MOCK_LLM'] = 'true'
    
    try:
        sys.path.append('src')
        from lib.langchain import llm
        
        # Test LLM query speed
        start_time = time.time()
        llm_model = "stanford:gpt-4o-mini"
        
        # Generate response
        response_tokens = list(llm.query_llm(
            llm_model=llm_model,
            question="What is quantum mechanics?",
            system_message="You are a helpful assistant."
        ))
        
        llm_time = time.time() - start_time
        response_text = ''.join(response_tokens)
        
        print(f"  ✅ Mock LLM responded in {llm_time:.3f}s")
        print(f"  📝 Response length: {len(response_text)} chars")
        print(f"  🎯 Response preview: {response_text[:100]}...")
        
        # Validate it's actually using mock
        if llm_time < 0.5 and "Mock response" in response_text:
            print("  ✅ Mock LLM is working (fast response with mock content)")
            return True
        else:
            print("  ⚠️  LLM response doesn't look like mock - might be using real API")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing mock LLM: {e}")
        return False

def test_both_mocks():
    """Test both mocks enabled together"""
    print("\n🔄 Testing both mocks enabled...")
    
    os.environ['USE_MOCK_EMBEDDINGS'] = 'true'
    os.environ['USE_MOCK_LLM'] = 'true'
    
    try:
        sys.path.append('src')
        from lib import rag_service
        
        start_time = time.time()
        
        # Run a full RAG query with both mocks
        answer, docs = rag_service.answer_query(
            query="What is quantum mechanics?",
            embedding_model="huggingface:thellert/physbert_cased",
            llm_model="stanford:gpt-4o-mini",
            k=3,
            score_threshold=0.0,
            use_opensearch=False,  # Use in-memory
            prompt_template="You are a helpful physics assistant.",
            user_files=[],
            history=[],
            use_rag=True,
            knowledge_base="default",
            use_qdrant=False
        )
        
        total_time = time.time() - start_time
        
        print(f"  ✅ Full RAG query completed in {total_time:.3f}s")
        print(f"  📚 Retrieved {len(docs)} documents")
        print(f"  📝 Answer length: {len(answer)} chars")
        print(f"  🎯 Answer preview: {answer[:100]}...")
        
        # With both mocks, this should be very fast
        if total_time < 2.0:
            print("  ✅ Both mocks working - full pipeline very fast")
            return True
        else:
            print("  ⚠️  Pipeline slower than expected with both mocks")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing both mocks: {e}")
        return False

def test_api_endpoint():
    """Test the API endpoint responds (if running)"""
    print("\n🌐 Testing API endpoint availability...")
    
    api_url = "http://localhost:8001"
    
    try:
        # Test basic connectivity
        response = requests.get(f"{api_url}/docs", timeout=5)
        if response.status_code == 200:
            print("  ✅ API endpoint is running and accessible")
            
            # Test ask endpoint with mock
            test_payload = {
                "query": "Test query",
                "embedding_model": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
                "llm_model": "stanford:gpt-4o-mini",
                "prompt": "You are a helpful assistant.",
                "use_rag": False
            }
            
            ask_response = requests.post(f"{api_url}/ask", json=test_payload, timeout=10)
            if ask_response.status_code == 200:
                print("  ✅ /ask endpoint working")
                return True
            else:
                print(f"  ⚠️  /ask endpoint returned {ask_response.status_code}")
                return False
        else:
            print(f"  ⚠️  API returned {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ⚠️  API endpoint not running (this is OK for testing)")
        print("  💡 To start: uv run uvicorn src.app_api:app --reload --port 8001")
        return False
    except Exception as e:
        print(f"  ❌ Error testing API: {e}")
        return False

def main():
    print("🧪 OuRAGboros Mock System Validation")
    print("===================================")
    
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    results = []
    
    # Test individual components
    results.append(("Mock Embeddings", test_mock_embeddings()))
    results.append(("Mock LLM", test_mock_llm()))  
    results.append(("Both Mocks", test_both_mocks()))
    results.append(("API Endpoint", test_api_endpoint()))
    
    # Summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed! Mock system is ready for benchmarking.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
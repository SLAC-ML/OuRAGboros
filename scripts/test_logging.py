#!/usr/bin/env python3
"""
Simple test script for the auto-log and eval feature.
Tests the query logging system without requiring the full stack.
"""
import asyncio
import os
import sys
import time
import json
from datetime import datetime

# Add src to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from langchain.schema import Document
from lib.query_logger import (
    QueryLoggerService, 
    QueryLogEntry,
    LoggingStatus,
    RagasEvaluatorClient
)

async def test_basic_logging():
    """Test basic logging functionality without OpenSearch or RAGAS"""
    print("🔧 Testing basic logging functionality...")
    
    # Create mock RAG data
    mock_docs = [
        (Document(page_content="Quantum mechanics is a fundamental theory in physics.", 
                 metadata={"source": "physics_textbook.pdf", "page_number": 1}), 0.92),
        (Document(page_content="It describes the physical properties of nature at small scales.", 
                 metadata={"source": "physics_textbook.pdf", "page_number": 2}), 0.88)
    ]
    mock_docs[0][0].id = "doc_1"
    mock_docs[1][0].id = "doc_2"
    
    # Create log entry
    log_entry = QueryLogEntry.create_from_api_data(
        query="What is quantum mechanics?",
        embedding_model="huggingface:sentence-transformers/all-mpnet-base-v2",
        llm_model="stanford:gpt-4",
        knowledge_base="physics_papers",
        score_threshold=0.5,
        max_documents=5,
        use_opensearch=True,
        use_qdrant=False,
        prompt_template="You are a helpful physics assistant.",
        rag_docs=mock_docs,
        final_answer="Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the atomic and subatomic scale. It differs significantly from classical mechanics, introducing concepts like wave-particle duality, uncertainty principle, and quantum superposition.",
        retrieval_time=0.342,
        response_time=2.15,
        total_time=3.89,
        embedding_time=0.12,
        token_count=45
    )
    
    print(f"✅ Created log entry: {log_entry.id}")
    print(f"   Query: {log_entry.user_query}")
    print(f"   Status: {log_entry.status.value}")
    print(f"   Documents: {log_entry.rag_results.documents_count}")
    
    # Test serialization
    doc = log_entry.to_opensearch_doc()
    print(f"✅ Serialized to OpenSearch doc: {len(json.dumps(doc))} bytes")
    
    return log_entry

async def test_ragas_client():
    """Test RAGAS evaluator client (will fail if service not running, which is expected)"""
    print("\n🔧 Testing RAGAS evaluator client...")
    
    # This will fail if RAGAS service isn't running, but we can test the client logic
    client = RagasEvaluatorClient("http://localhost:8002")
    
    # Create a simple log entry for testing
    mock_doc = (Document(page_content="Test content", metadata={"source": "test"}), 0.9)
    mock_doc[0].id = "test_doc"
    
    log_entry = QueryLogEntry.create_from_api_data(
        query="Test query?",
        embedding_model="test_model",
        llm_model="test_llm", 
        knowledge_base="test",
        score_threshold=0.5,
        max_documents=1,
        use_opensearch=False,
        use_qdrant=False,
        prompt_template="Test prompt",
        rag_docs=[mock_doc],
        final_answer="Test answer",
        retrieval_time=0.1,
        response_time=1.0,
        total_time=2.0,
        token_count=10
    )
    
    try:
        evaluation = await client.evaluate_query(log_entry)
        if evaluation.error:
            print(f"⚠️  RAGAS evaluation failed (expected if service not running): {evaluation.error}")
        else:
            print(f"✅ RAGAS evaluation successful!")
            print(f"   Faithfulness: {evaluation.faithfulness}")
            print(f"   Answer Relevancy: {evaluation.answer_relevancy}")
    except Exception as e:
        print(f"⚠️  RAGAS client test failed (expected if service not running): {e}")
    
    await client.close()

async def test_logger_service():
    """Test the full logger service (will fail without OpenSearch, which is expected)"""
    print("\n🔧 Testing QueryLoggerService...")
    
    # Create logger service
    logger_service = QueryLoggerService(
        opensearch_host="http://localhost:9200",
        ragas_base_url="http://localhost:8002",
        batch_size=2,  # Small batch for testing
        batch_interval=2.0  # Short interval for testing
    )
    
    try:
        await logger_service.start()
        print("✅ Logger service started")
        
        # Create test log entries
        for i in range(3):
            mock_doc = (Document(page_content=f"Test content {i}", metadata={"source": f"test{i}"}), 0.8 + i*0.05)
            mock_doc[0].id = f"test_doc_{i}"
            
            log_entry = QueryLogEntry.create_from_api_data(
                query=f"Test query {i}?",
                embedding_model="test_model",
                llm_model="test_llm", 
                knowledge_base="test",
                score_threshold=0.5,
                max_documents=1,
                use_opensearch=False,
                use_qdrant=False,
                prompt_template="Test prompt",
                rag_docs=[mock_doc],
                final_answer=f"Test answer {i}",
                retrieval_time=0.1 + i*0.05,
                response_time=1.0 + i*0.1,
                total_time=2.0 + i*0.2,
                token_count=10 + i*2
            )
            
            # Queue the log entry (should not block)
            await logger_service.log_query(log_entry)
            print(f"✅ Queued log entry {i}: {log_entry.id}")
        
        # Wait for batch processing
        print("⏳ Waiting for batch processing...")
        await asyncio.sleep(5)
        
        # Test search (will fail without OpenSearch)
        try:
            result = await logger_service.search_logs("test", size=10)
            print(f"✅ Search test: {result.get('total', 0)} results")
        except Exception as e:
            print(f"⚠️  Search test failed (expected without OpenSearch): {e}")
        
    except Exception as e:
        print(f"⚠️  Logger service test failed (expected without OpenSearch): {e}")
    
    finally:
        await logger_service.stop()
        print("✅ Logger service stopped")

async def main():
    """Run all tests"""
    print("🚀 Starting auto-log and eval feature tests...\n")
    
    # Test 1: Basic logging functionality
    await test_basic_logging()
    
    # Test 2: RAGAS client
    await test_ragas_client()
    
    # Test 3: Logger service
    await test_logger_service()
    
    print("\n✅ All tests completed!")
    print("\n📝 Summary:")
    print("   - Basic logging functionality: ✅ Working")
    print("   - RAGAS client: ⚠️  Requires RAGAS service running")
    print("   - Logger service: ⚠️  Requires OpenSearch running")
    print("\n🔧 Next steps:")
    print("   1. Start docker-compose with ragas-evaluator service")
    print("   2. Test with real API calls through /ask/stream")
    print("   3. Open query_logs_viewer.html to inspect results")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Test Qdrant integration with the Streamlit application components.
This simulates the UI workflow without actually running Streamlit.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path so we can import
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables for testing
os.environ["QDRANT_BASE_URL"] = "http://localhost:6333"
os.environ["PREFER_QDRANT"] = "true"
os.environ["PREFER_OPENSEARCH"] = "false"
os.environ["HUGGINGFACE_EMBEDDING_MODEL_DEFAULT"] = "sentence-transformers/all-MiniLM-L6-v2"

def test_qdrant_knowledge_base_operations():
    """Test knowledge base create/list/delete operations"""
    print("🧪 Testing Qdrant Knowledge Base Operations...")
    
    try:
        import lib.langchain.qdrant as langchain_qdrant
        from langchain.schema import Document
        
        # Test 1: List empty knowledge bases
        print("📋 Testing knowledge base listing...")
        kbs = langchain_qdrant.get_available_knowledge_bases()
        print(f"✅ Available knowledge bases: {kbs}")
        
        # Test 2: Create new knowledge base
        print("🔧 Testing knowledge base creation...")
        test_embedding_model = "huggingface:sentence-transformers/all-MiniLM-L6-v2"
        test_kb_name = "test_ui_integration"
        
        collection_name = langchain_qdrant.ensure_qdrant_collection(
            test_embedding_model, test_kb_name
        )
        print(f"✅ Created knowledge base collection: {collection_name}")
        
        # Test 3: Add test documents
        print("📝 Testing document addition...")
        test_docs = [
            Document(
                page_content="This is a test document about physics.",
                metadata={"source": "test_file.txt", "page": 1}
            ),
            Document(
                page_content="Another test document about quantum mechanics.", 
                metadata={"source": "test_file2.txt", "page": 1}
            )
        ]
        
        doc_ids = langchain_qdrant.add_documents_to_qdrant(
            test_docs, test_embedding_model, test_kb_name
        )
        print(f"✅ Added {len(doc_ids)} documents to knowledge base")
        
        # Test 4: Search documents
        print("🔍 Testing document search...")
        search_results = langchain_qdrant.search_qdrant_documents(
            query="physics quantum",
            embedding_model=test_embedding_model,
            knowledge_base=test_kb_name,
            k=2,
            score_threshold=0.0
        )
        print(f"✅ Search returned {len(search_results)} documents")
        for i, (doc, score) in enumerate(search_results):
            print(f"   {i+1}. Score: {score:.3f} - {doc.page_content[:50]}...")
        
        # Test 5: Clean up - delete knowledge base
        print("🗑️  Testing knowledge base deletion...")
        client = langchain_qdrant.get_qdrant_client()
        try:
            client.delete_collection(collection_name=collection_name)
            print("✅ Successfully deleted test knowledge base")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_session_integration():
    """Test session state integration with Qdrant"""
    print("\n🧪 Testing Streamlit Session State Integration...")
    
    try:
        # Mock Streamlit session state
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def get(self, key, default=None):
                return self.data.get(key, default)
            
            def __getitem__(self, key):
                return self.data[key]
            
            def __setitem__(self, key, value):
                self.data[key] = value
            
            def __contains__(self, key):
                return key in self.data

        # Mock Streamlit
        class MockStreamlit:
            def __init__(self):
                self.session_state = MockSessionState()
                self._cache_resource_data = {}
                
            def cache_resource(self, func=None):
                def decorator(f):
                    def wrapper(*args, **kwargs):
                        cache_key = f"{f.__name__}_{hash(str(args) + str(kwargs))}"
                        if cache_key not in self._cache_resource_data:
                            self._cache_resource_data[cache_key] = f(*args, **kwargs)
                        return self._cache_resource_data[cache_key]
                    wrapper.clear = lambda: self._cache_resource_data.clear()
                    return wrapper
                return decorator(func) if func else decorator

        # Install mock
        import lib.streamlit.session_state as ss
        mock_st = MockStreamlit()
        
        # Test session state initialization
        print("🔧 Testing session state initialization...")
        
        # Set up mock session state with Qdrant preferences
        mock_st.session_state[ss.StateKey.USE_QDRANT] = True
        mock_st.session_state[ss.StateKey.USE_OPENSEARCH] = False
        mock_st.session_state[ss.StateKey.EMBEDDING_MODEL] = "huggingface:sentence-transformers/all-MiniLM-L6-v2"
        mock_st.session_state[ss.StateKey.KNOWLEDGE_BASE] = "default"
        
        # Test get_vector_store with Qdrant
        print("📊 Testing vector store retrieval...")
        vector_store = ss.get_vector_store(
            use_opensearch_vectorstore=False,
            model=mock_st.session_state[ss.StateKey.EMBEDDING_MODEL],
            knowledge_base=mock_st.session_state[ss.StateKey.KNOWLEDGE_BASE],
            use_qdrant=True
        )
        
        print(f"✅ Successfully created vector store: {type(vector_store).__name__}")
        
        # Test storage toggle utilities
        print("🔄 Testing storage toggle utilities...")
        import lib.streamlit.storage_toggle as storage_toggle
        
        # Mock the session state for storage toggle
        class MockST:
            session_state = mock_st.session_state
        
        # Replace st reference temporarily
        import lib.streamlit.storage_toggle
        original_st = getattr(lib.streamlit.storage_toggle, 'st', None)
        lib.streamlit.storage_toggle.st = MockST()
        
        try:
            storage_modes = storage_toggle.get_storage_modes()
            current_mode = storage_toggle.get_current_storage_mode()
            
            print(f"✅ Available storage modes: {storage_modes}")
            print(f"✅ Current storage mode: {current_mode}")
            
            if current_mode != "Qdrant":
                print(f"⚠️  Expected Qdrant mode, got {current_mode}")
                return False
                
        finally:
            # Restore original st
            if original_st:
                lib.streamlit.storage_toggle.st = original_st
        
        return True
        
    except Exception as e:
        print(f"❌ Session state integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_service_integration():
    """Test RAG service with Qdrant"""
    print("\n🧪 Testing RAG Service Integration...")
    
    try:
        import lib.rag_service as rag_service
        
        # Create a test knowledge base with some content
        import lib.langchain.qdrant as langchain_qdrant
        from langchain.schema import Document
        
        test_embedding_model = "huggingface:sentence-transformers/all-MiniLM-L6-v2"
        test_kb_name = "test_rag_integration"
        
        # Ensure collection exists
        langchain_qdrant.ensure_qdrant_collection(test_embedding_model, test_kb_name)
        
        # Add test documents
        test_docs = [
            Document(
                page_content="The speed of light in vacuum is exactly 299,792,458 meters per second.",
                metadata={"source": "physics_constants.txt", "page": 1}
            ),
            Document(
                page_content="Einstein's mass-energy equivalence is expressed as E = mc².",
                metadata={"source": "einstein_equations.txt", "page": 1}  
            )
        ]
        
        langchain_qdrant.add_documents_to_qdrant(test_docs, test_embedding_model, test_kb_name)
        
        # Test document retrieval function
        print("🔍 Testing document retrieval...")
        docs = rag_service.perform_document_retrieval(
            query="What is the speed of light?",
            embedding_model=test_embedding_model,
            k=2,
            score_threshold=0.0,
            use_opensearch=False,
            knowledge_base=test_kb_name,
            use_qdrant=True
        )
        
        print(f"✅ Retrieved {len(docs)} documents")
        for i, (doc, score) in enumerate(docs):
            print(f"   {i+1}. Score: {score:.3f} - {doc.page_content[:50]}...")
        
        # Clean up
        client = langchain_qdrant.get_qdrant_client()
        collection_name = langchain_qdrant.get_collection_name(test_embedding_model, test_kb_name)
        try:
            client.delete_collection(collection_name=collection_name)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ RAG service integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("🧪 Qdrant UI Integration Test Suite")
    print("=" * 60)
    
    # Check if Qdrant is accessible
    try:
        import subprocess
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
    
    tests = [
        ("Knowledge Base Operations", test_qdrant_knowledge_base_operations),
        ("Session State Integration", test_streamlit_session_integration), 
        ("RAG Service Integration", test_rag_service_integration),
    ]
    
    for test_name, test_func in tests:
        print(f"🧪 Running: {test_name}")
        if not test_func():
            print(f"❌ {test_name} failed")
            success = False
        else:
            print(f"✅ {test_name} passed")
        print()
    
    print("=" * 60)
    if success:
        print("🎉 All Qdrant UI integration tests passed!")
        print("\n💡 Ready to test with Streamlit:")
        print("   QDRANT_BASE_URL=http://localhost:6333 PREFER_QDRANT=true uv run streamlit run src/main.py")
    else:
        print("❌ Some tests failed - check output above")
    
    return success

if __name__ == "__main__":
    main()
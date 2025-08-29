#!/usr/bin/env python3
"""
Basic Qdrant connection test without heavy dependencies.
"""

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

if __name__ == "__main__":
    print("🧪 Testing Qdrant Basic Connection")
    print("=" * 40)
    
    if test_qdrant_connection():
        print("🎉 Basic connection test passed!")
        print("\n💡 Next: Run full integration test with uv")
    else:
        print("💡 Make sure Qdrant is running: docker-compose up -d qdrant")
# lib/langchain/qdrant.py
"""
Qdrant vector store integration for high-performance concurrent vector search.
This module provides Qdrant-specific functionality to replace OpenSearch for better concurrency.
"""

import os
from typing import List, Optional, Dict, Any, Tuple
import threading
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest
import uuid

import lib.config as config
import lib.langchain.embeddings as langchain_embeddings

# Thread-safe cache for Qdrant clients and collections
_qdrant_clients = {}
_qdrant_clients_lock = threading.Lock()

def get_qdrant_client(url: str = None) -> QdrantClient:
    """
    Get a thread-safe cached Qdrant client instance.
    
    :param url: Qdrant server URL (defaults to config value)
    :return: Cached QdrantClient instance
    """
    if url is None:
        url = getattr(config, 'qdrant_base_url', 'http://localhost:6333')
    
    if url not in _qdrant_clients:
        with _qdrant_clients_lock:
            # Double-check locking pattern
            if url not in _qdrant_clients:
                _qdrant_clients[url] = QdrantClient(url=url)
    
    return _qdrant_clients[url]

def get_collection_name(embedding_model: str, knowledge_base: str = "default") -> str:
    """
    Generate a consistent collection name for the given parameters.
    
    :param embedding_model: Embedding model identifier
    :param knowledge_base: Knowledge base name
    :return: Collection name
    """
    # Clean up names for Qdrant collection naming requirements
    clean_model = embedding_model.replace(":", "_").replace("/", "_")
    clean_kb = knowledge_base.replace(":", "_").replace("/", "_")
    return f"ouragboros_{clean_model}_{clean_kb}"

def ensure_qdrant_collection(embedding_model: str, knowledge_base: str = "default") -> str:
    """
    Ensure a Qdrant collection exists for the given model and knowledge base.
    
    :param embedding_model: Embedding model identifier  
    :param knowledge_base: Knowledge base name
    :return: Collection name
    """
    client = get_qdrant_client()
    collection_name = get_collection_name(embedding_model, knowledge_base)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name not in collection_names:
        # Get vector size from embedding model
        embeddings = langchain_embeddings.get_embedding(embedding_model)
        vector_size = langchain_embeddings.get_vector_size(embedding_model)
        
        # Create collection with optimized settings for concurrent search
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,  # Cosine similarity for semantic search
            ),
            # Optimizations for concurrent performance
            optimizers_config={
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 2,  # More segments for better parallelization
                "max_segment_size": 20000,
                "memmap_threshold": 50000,
                "indexing_threshold": 10000,
                "flush_interval_sec": 5,
            },
            hnsw_config={
                "m": 16,  # Number of bi-directional links created for each new element
                "ef_construct": 200,  # Size of the dynamic candidate list
                "full_scan_threshold": 10000,  # Threshold for switching to full scan
                "max_indexing_threads": 0,  # Use all available threads
            }
        )
        print(f"✅ Created Qdrant collection: {collection_name}")
    
    return collection_name

def qdrant_vector_store(embedding_model: str, knowledge_base: str = "default") -> VectorStore:
    """
    Create a Qdrant vector store instance for the given model and knowledge base.
    
    :param embedding_model: Embedding model identifier
    :param knowledge_base: Knowledge base name  
    :return: Configured Qdrant vector store
    """
    client = get_qdrant_client()
    collection_name = ensure_qdrant_collection(embedding_model, knowledge_base)
    embeddings = langchain_embeddings.get_embedding(embedding_model)
    
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )

def get_available_knowledge_bases() -> List[str]:
    """
    Get list of available knowledge bases from Qdrant collections.
    Knowledge bases are ordered by creation time (earliest document uploaded first), with "default" always first.
    
    :return: List of knowledge base names ordered by creation time
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        
        # Extract knowledge base names and their creation times from collection names
        import lib.langchain.embeddings as langchain_embeddings
        
        # Get available embedding models to help with parsing
        available_embeddings = langchain_embeddings.get_available_embeddings()
        
        kb_timestamps = {}  # kb_name -> earliest_timestamp
        
        for collection in collections:
            if collection.name.startswith("ouragboros_"):
                remainder = collection.name[11:]  # Remove "ouragboros_" prefix
                
                # Try to match against known embedding models
                matched_kb = None
                for embedding_model in available_embeddings:
                    clean_model = embedding_model.replace(":", "_").replace("/", "_")
                    if remainder.startswith(clean_model + "_"):
                        # Found matching model, extract KB name
                        kb_name = remainder[len(clean_model) + 1:]  # +1 for the underscore
                        matched_kb = kb_name
                        break
                
                if not matched_kb:
                    # Fallback: assume last part after underscore is KB name
                    if "_" in remainder:
                        kb_name = remainder.rsplit("_", 1)[-1]
                        matched_kb = kb_name
                
                if matched_kb:
                    # Get the earliest uploaded timestamp from this collection
                    try:
                        collection_info = client.get_collection(collection.name)
                        if collection_info.points_count > 0:
                            # Sample points to find earliest timestamp
                            points, _ = client.scroll(
                                collection_name=collection.name, 
                                limit=min(50, collection_info.points_count), 
                                with_payload=True
                            )
                            
                            earliest_timestamp = None
                            for point in points:
                                if point.payload and 'metadata' in point.payload:
                                    uploaded = point.payload['metadata'].get('uploaded')
                                    if uploaded:
                                        if earliest_timestamp is None or uploaded < earliest_timestamp:
                                            earliest_timestamp = uploaded
                            
                            # Use the earliest timestamp found, or current time if none found
                            if earliest_timestamp is not None:
                                # Use the earliest timestamp for each knowledge base 
                                if matched_kb not in kb_timestamps or earliest_timestamp < kb_timestamps[matched_kb]:
                                    kb_timestamps[matched_kb] = earliest_timestamp
                            else:
                                # No timestamps found, use current time as fallback
                                import time
                                if matched_kb not in kb_timestamps:
                                    kb_timestamps[matched_kb] = time.time()
                        else:
                            # Empty collection, use current time
                            import time
                            if matched_kb not in kb_timestamps:
                                kb_timestamps[matched_kb] = time.time()
                                
                    except Exception as e:
                        print(f"Warning: Could not get timestamp for collection {collection.name}: {e}")
                        # Fallback to current time
                        import time
                        if matched_kb not in kb_timestamps:
                            kb_timestamps[matched_kb] = time.time()
        
        if not kb_timestamps:
            return ["default"]
        
        # Sort by creation time (earliest first)
        sorted_kbs = sorted(kb_timestamps.items(), key=lambda x: x[1])
        kb_list = [kb for kb, _ in sorted_kbs]
        
        # Ensure 'default' is always first
        if "default" in kb_list:
            kb_list.remove("default")
        kb_list.insert(0, "default")
        
        print(f"Knowledge bases ordered by creation time: {kb_list}")
        return kb_list
        
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant: {e}")
        return ["default"]

def add_documents_to_qdrant(
    documents: List[Document], 
    embedding_model: str, 
    knowledge_base: str = "default"
) -> List[str]:
    """
    Add documents to Qdrant collection with optimized batch processing.
    
    :param documents: List of documents to add
    :param embedding_model: Embedding model identifier
    :param knowledge_base: Knowledge base name
    :return: List of document IDs
    """
    vectorstore = qdrant_vector_store(embedding_model, knowledge_base)
    
    # Use batch processing for better performance
    batch_size = 100
    all_ids = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_ids = vectorstore.add_documents(batch)
        all_ids.extend(batch_ids)
        print(f"✅ Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} to Qdrant ({len(batch)} documents)")
    
    return all_ids

def search_qdrant_documents(
    query: str,
    embedding_model: str,
    knowledge_base: str = "default",
    k: int = 5,
    score_threshold: float = 0.5,
    search_params: Optional[Dict[str, Any]] = None
) -> List[Tuple[Document, float]]:
    """
    Search documents in Qdrant with optimized concurrent performance.
    
    :param query: Search query
    :param embedding_model: Embedding model identifier
    :param knowledge_base: Knowledge base name
    :param k: Number of results to return
    :param score_threshold: Minimum similarity score threshold
    :param search_params: Additional search parameters
    :return: List of (document, score) tuples
    """
    vectorstore = qdrant_vector_store(embedding_model, knowledge_base)
    
    # Use similarity_search_with_score for better performance control
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        score_threshold=score_threshold,
        search_params=search_params or {
            "hnsw_ef": 128,  # Size of the dynamic candidate list (higher = more accurate)
            "exact": False,  # Use approximate search for speed
        }
    )
    
    return results

def clear_qdrant_cache():
    """Clear all cached Qdrant clients."""
    with _qdrant_clients_lock:
        _qdrant_clients.clear()
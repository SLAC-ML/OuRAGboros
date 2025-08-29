import os.path
import threading
import numpy as np
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import lib.config as config

import lib.langchain.llm as langchain_llms
import lib.langchain.util as langchain_utils

# Thread-safe caches for embedding instances and vector stores
_embedding_cache = {}
_embedding_cache_lock = threading.Lock()
_vector_size_cache = {}
_vector_size_cache_lock = threading.Lock()
_in_memory_vector_stores = {}
_in_memory_vector_stores_lock = threading.Lock()


class MockEmbeddings(Embeddings):
    """
    Mock embeddings class that returns instant pre-computed vectors.
    Used for performance testing to isolate embedding generation bottlenecks.
    
    Compatible with PhysBERT (768-dimensional vectors).
    """
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # Pre-computed normalized vector for consistent similarity scores
        # This simulates a realistic PhysBERT embedding for query "What's this docs about?"
        np.random.seed(42)  # Reproducible vector
        self._cached_vector = np.random.normal(0, 1, dimension)
        # Normalize to unit vector (like real embeddings)
        self._cached_vector = self._cached_vector / np.linalg.norm(self._cached_vector)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return cached vector for all documents instantly."""
        # Return slightly varied vectors to simulate realistic diversity
        vectors = []
        for i, text in enumerate(texts):
            # Add small random variation (but keep reproducible)
            np.random.seed(42 + i)
            variation = np.random.normal(0, 0.01, self.dimension)  # Small noise
            varied_vector = self._cached_vector + variation
            varied_vector = varied_vector / np.linalg.norm(varied_vector)  # Renormalize
            vectors.append(varied_vector.tolist())
        return vectors
    
    def embed_query(self, text: str) -> List[float]:
        """Return cached vector for query instantly (no computation)."""
        # This eliminates the PhysBERT bottleneck completely
        return self._cached_vector.tolist()


def get_in_memory_vector_store(embedding_model: str, knowledge_base: str = "default") -> VectorStore:
    """
    Returns a basic in-memory vector store for performing a vector search. Useful for
    development. Now supports knowledge base isolation.

    :param embedding_model: The embedding model to use (e.g., "huggingface:bert-base")
    :param knowledge_base: The knowledge base name for isolation
    :return: Thread-safe cached vector store instance
    """
    # Use both embedding model and knowledge base for isolation
    cache_key = f"{embedding_model}#{knowledge_base}"
    
    if cache_key not in _in_memory_vector_stores:
        with _in_memory_vector_stores_lock:
            # Double-check locking pattern to prevent race conditions
            if cache_key not in _in_memory_vector_stores:
                # Use clean embedding model name (no pollution)
                clean_embedding = get_embedding(embedding_model)
                _in_memory_vector_stores[cache_key] = InMemoryVectorStore(clean_embedding)

    return _in_memory_vector_stores[cache_key]


def get_in_memory_knowledge_bases() -> list[str]:
    """
    Get list of available in-memory knowledge bases.
    
    :return: List of knowledge base names
    """
    with _in_memory_vector_stores_lock:
        # Extract knowledge base names from cache keys (format: "embedding_model#knowledge_base")
        kb_names = set()
        for cache_key in _in_memory_vector_stores.keys():
            if '#' in cache_key:
                kb_name = cache_key.split('#', 1)[1]
                kb_names.add(kb_name)
        return sorted(list(kb_names))


def clear_embedding_cache() -> None:
    """
    Clear all embedding caches. Useful for testing or memory management.
    """
    with _embedding_cache_lock:
        _embedding_cache.clear()
    with _vector_size_cache_lock:
        _vector_size_cache.clear()
    with _in_memory_vector_stores_lock:
        _in_memory_vector_stores.clear()


def get_available_embeddings() -> list[str]:
    """
    Retrieves available string embedding models.

    :return:
    """
    ollama_models = langchain_llms.get_available_llms()
    huggingface_models = [
        f'huggingface:{config.huggingface_default_embedding_model}',
    ]

    # If we have a local model trained, we prioritize that one.
    #
    finetuned_model_path = os.path.join(
        config.huggingface_model_cache_folder,
        config.huggingface_finetuned_embedding_model
    )

    if os.path.exists(finetuned_model_path):
        huggingface_models.insert(0, f'huggingface:{finetuned_model_path}')

    return [*huggingface_models, *ollama_models]


def get_embedding(embedding_model: str) -> Embeddings:
    """
    Returns a cached embedding instance for the provided model. Thread-safe implementation
    prevents multiple instantiation of the same model across concurrent requests.
    
    When USE_MOCK_EMBEDDINGS=true, returns MockEmbeddings for performance testing.

    :param embedding_model: Model identifier (e.g., "huggingface:bert-base")
    :return: Cached embedding instance
    """
    # Check if mock embeddings are enabled for performance testing
    if config.use_mock_embeddings:
        mock_key = f"mock_{embedding_model}"
        if mock_key not in _embedding_cache:
            with _embedding_cache_lock:
                if mock_key not in _embedding_cache:
                    print(f"🧪 Using MOCK embeddings for {embedding_model} (performance testing)")
                    _embedding_cache[mock_key] = MockEmbeddings()
        return _embedding_cache[mock_key]
    
    # Normal embedding logic (original implementation)
    if embedding_model not in _embedding_cache:
        with _embedding_cache_lock:
            # Double-check locking pattern
            if embedding_model not in _embedding_cache:
                model_source, model_name = langchain_utils.parse_model_name(embedding_model)
                
                # Create embedding instance (only once per model)
                if model_source == 'huggingface':
                    _embedding_cache[embedding_model] = HuggingFaceEmbeddings(
                        model_name=model_name,
                        cache_folder=config.huggingface_model_cache_folder,
                    )
                else:
                    _embedding_cache[embedding_model] = OllamaEmbeddings(
                        model=model_name, 
                        base_url=config.ollama_base_url
                    )

    return _embedding_cache[embedding_model]


def get_vector_size(embedding_model: str) -> int:
    """
    Thread-safe method to get vector size for an embedding model.
    Caches the result to avoid repeated embed_query("hi") calls.
    
    :param embedding_model: Model identifier
    :return: Vector dimension size
    """
    # Handle mock embeddings
    if config.use_mock_embeddings:
        return 768  # MockEmbeddings uses 768 dimensions (PhysBERT compatible)
    
    if embedding_model not in _vector_size_cache:
        with _vector_size_cache_lock:
            if embedding_model not in _vector_size_cache:
                embedding = get_embedding(embedding_model)  # Uses embedding cache
                _vector_size_cache[embedding_model] = len(embedding.embed_query("hi"))
    
    return _vector_size_cache[embedding_model]

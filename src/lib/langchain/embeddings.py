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


def get_local_custom_models() -> list[dict]:
    """
    Discover all local custom models by scanning subdirectories under models/ folder.
    Each subdirectory becomes a namespace (e.g., models/finetuned/my-model → finetuned/my-model).

    :return: List of model info dictionaries with 'path', 'namespace', 'name', and 'display_name'
    """
    local_models = []
    models_base_dir = config.huggingface_model_cache_folder

    if not os.path.exists(models_base_dir):
        return local_models

    try:
        # Scan all subdirectories in models/ folder (each becomes a namespace)
        for namespace_folder in os.listdir(models_base_dir):
            # Skip hidden files and HuggingFace cache directories
            if namespace_folder.startswith('.') or namespace_folder.startswith('models--'):
                continue

            namespace_path = os.path.join(models_base_dir, namespace_folder)

            # Only process directories
            if not os.path.isdir(namespace_path):
                continue

            # Scan for models within this namespace folder
            for model_name in os.listdir(namespace_path):
                model_path = os.path.join(namespace_path, model_name)

                # Skip hidden files and non-directories
                if model_name.startswith('.') or not os.path.isdir(model_path):
                    continue

                # Validate model directory (check for required files)
                if _is_valid_model_directory(model_path):
                    # Create friendly display name
                    display_name = _get_model_display_name(model_name, model_path)

                    local_models.append({
                        'path': model_path,
                        'namespace': namespace_folder,
                        'name': model_name,
                        'namespaced_name': f'{namespace_folder}/{model_name}',
                        'display_name': f'{namespace_folder.title()}: {display_name}'
                    })

    except Exception as e:
        print(f"Warning: Error scanning local custom models directory: {e}")

    # Sort by namespace first, then by name
    return sorted(local_models, key=lambda x: (x['namespace'], x['name']))


def _is_valid_model_directory(model_path: str) -> bool:
    """
    Check if a directory contains a valid HuggingFace model.

    :param model_path: Path to the model directory
    :return: True if valid, False otherwise
    """
    required_files = ['config.json']

    # Check for model weights (either .bin or .safetensors)
    weight_files = [
        'pytorch_model.bin',
        'model.safetensors',
        'pytorch_model.safetensors'
    ]

    try:
        files_in_dir = os.listdir(model_path)

        # Check required files
        for required_file in required_files:
            if required_file not in files_in_dir:
                return False

        # Check for at least one weight file that is NOT empty
        has_valid_weights = False
        for weight_file in weight_files:
            if weight_file in files_in_dir:
                weight_path = os.path.join(model_path, weight_file)
                # Check file exists and is not empty (size > 0)
                if os.path.exists(weight_path) and os.path.getsize(weight_path) > 0:
                    has_valid_weights = True
                    break

        if not has_valid_weights:
            return False

        return True

    except Exception:
        return False


def _get_model_display_name(model_name: str, model_path: str) -> str:
    """
    Generate a friendly display name for a model.

    :param model_name: Directory name of the model
    :param model_path: Full path to model directory
    :return: User-friendly display name
    """
    # Try to load metadata file for custom display name
    metadata_file = os.path.join(model_path, 'model_metadata.json')
    if os.path.exists(metadata_file):
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if 'display_name' in metadata:
                    return metadata['display_name']
        except Exception:
            pass  # Fall back to automatic naming

    # Generate display name from directory name
    display_name = model_name.replace('-', ' ').replace('_', ' ')

    # Capitalize words
    display_name = ' '.join(word.capitalize() for word in display_name.split())

    # Add "Fine-tuned" suffix if not already present
    if 'fine' not in display_name.lower() and 'tuned' not in display_name.lower():
        display_name += ' (Fine-tuned)'

    return display_name


def get_available_embeddings() -> list[str]:
    """
    Retrieves available string embedding models, including multiple fine-tuned models.

    :return: List of model identifiers
    """
    ollama_models = langchain_llms.get_available_llms()
    huggingface_models = [
        f'huggingface:{config.huggingface_default_embedding_model}',
    ]

    # Add all local custom models from directory scanning (with namespace prefixes)
    local_models = get_local_custom_models()
    for model_info in local_models:
        # Use namespace/model format (e.g., huggingface:finetuned/my-model)
        model_identifier = f'huggingface:{model_info["namespaced_name"]}'
        huggingface_models.insert(0, model_identifier)

    # Legacy support: check for single fine-tuned model via environment variable
    # This maintains backward compatibility with existing deployments
    if config.huggingface_finetuned_embedding_model and config.huggingface_finetuned_embedding_model.strip():
        legacy_finetuned_path = os.path.join(
            config.huggingface_model_cache_folder,
            config.huggingface_finetuned_embedding_model
        )

        if os.path.exists(legacy_finetuned_path):
            # Check if this path matches a directory-scanned model
            # If so, skip to avoid duplicates (directory scanning takes precedence)
            is_duplicate = any(
                model_info['path'] == legacy_finetuned_path
                for model_info in local_models
            )

            if not is_duplicate:
                # For legacy env var models, extract namespace if present
                # Format could be "finetuned/model" or just "model"
                legacy_model_path = config.huggingface_finetuned_embedding_model
                if '/' in legacy_model_path:
                    # Already has namespace
                    legacy_identifier = f'huggingface:{legacy_model_path}'
                else:
                    # No namespace, use just the model name for backward compatibility
                    legacy_model_name = os.path.basename(legacy_finetuned_path)
                    legacy_identifier = f'huggingface:{legacy_model_name}'

                if legacy_identifier not in huggingface_models:
                    huggingface_models.insert(0, legacy_identifier)

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
                    # Check if this is a local custom model (format: namespace/model-name)
                    # e.g., "finetuned/my-model", "custom/bert-variant", etc.
                    if '/' in model_name:
                        # Parse namespace and model name
                        namespace, model_base_name = model_name.split('/', 1)
                        local_model_path = os.path.join(
                            config.huggingface_model_cache_folder,
                            namespace,
                            model_base_name
                        )

                        if os.path.exists(local_model_path):
                            # Use the full path for local custom models
                            resolved_model_name = local_model_path
                        else:
                            # Fallback: treat as HuggingFace Hub model (e.g., "sentence-transformers/all-MiniLM-L6-v2")
                            resolved_model_name = model_name
                    else:
                        # Legacy support: check old finetuned directory without namespace
                        legacy_finetuned_path = os.path.join(
                            config.huggingface_model_cache_folder,
                            'finetuned',
                            model_name
                        )
                        if os.path.exists(legacy_finetuned_path):
                            # Use the full path for legacy fine-tuned models
                            resolved_model_name = legacy_finetuned_path
                        else:
                            # Use the model name as-is for HuggingFace Hub models
                            resolved_model_name = model_name

                    _embedding_cache[embedding_model] = HuggingFaceEmbeddings(
                        model_name=resolved_model_name,
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

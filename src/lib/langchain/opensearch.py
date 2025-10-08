import hashlib
import logging
import threading
from typing import Tuple

import opensearchpy.exceptions
from opensearchpy import OpenSearch

from langchain_community.vectorstores import OpenSearchVectorSearch

import lib.config as config


import lib.langchain.embeddings as langchain_embeddings

# Thread-safe cache for OpenSearch configurations
_opensearch_configs = {}
_opensearch_configs_lock = threading.Lock()

def get_opensearch_index_settings(
    embedding_model: str, knowledge_base: str = "default"
) -> Tuple[str, int]:
    """
    Populates an OpenSearch configuration template with a particular model name, knowledge base,
    and vector embedding size.

    :param embedding_model: The embedding model name
    :param knowledge_base: The knowledge base name (defaults to "default")
    :return: Tuple of (index_name, vector_size)
    """
    cache_key = f"{embedding_model}#{knowledge_base}"
    if cache_key not in _opensearch_configs:
        with _opensearch_configs_lock:
            # Double-check locking pattern for thread safety
            if cache_key not in _opensearch_configs:
                logging.debug(f"{cache_key} not found in _opensearch_configs; generating..")
                
                # Use the new thread-safe vector size function
                vector_size = langchain_embeddings.get_vector_size(embedding_model)
                model_id = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()

                # Sanitize knowledge base name for use in index name
                kb_safe = knowledge_base.lower().replace(" ", "_").replace("-", "_")
                kb_safe = "".join(c for c in kb_safe if c.isalnum() or c == "_")

                # For backward compatibility, use original format for "default" knowledge base
                if knowledge_base == "default":
                    index_name = f"{config.opensearch_index_prefix}_{vector_size}_{model_id}"
                else:
                    index_name = (
                        f"{config.opensearch_index_prefix}_{kb_safe}_{vector_size}_{model_id}"
                    )

                _opensearch_configs[cache_key] = (index_name, vector_size)

    return _opensearch_configs[cache_key]


def ensure_opensearch_index(
    embedding_model_name: str, knowledge_base: str = "default"
) -> None:
    """
    Ensures the existence of an OpenSearch index. We create one index per embedding model
    and knowledge base combination, since different models may generate various embedding 
    vector sizes (and even if two models generate vectors of the same size, they certainly 
    do not do so the same way).

    :param embedding_model_name: The embedding model name
    :param knowledge_base: The knowledge base name (defaults to "default")
    :return:
    """
    try:
        index_name, vector_size = get_opensearch_index_settings(
            embedding_model_name, knowledge_base
        )

        opensearch_client = OpenSearch([config.opensearch_base_url])
        
        # Check if index already exists before trying to create it
        if not opensearch_client.indices.exists(index=index_name):
            opensearch_client.indices.create(
                index=index_name,
                body=config.opensearch_index_settings(vector_size=vector_size),
            )
            logging.debug(f"Created new OpenSearch index: {index_name}")
        else:
            logging.debug(f"OpenSearch index already exists: {index_name}")
            
    except opensearchpy.exceptions.RequestError as e:
        if e.status_code != 400:
            raise e


def opensearch_doc_vector_store(
    embedding_model: str, knowledge_base: str = "default"
) -> OpenSearchVectorSearch:
    """
    Uses a specific embedding model to perform vector embedding for documents via
    OpenSearch in a specific knowledge base.

    :param embedding_model: Expected to follow the format <model_source>:<model_name>
                                 (e.g., ollama:deepseek-r1:latest).
    :param knowledge_base: The knowledge base name (defaults to "default")
    :return:
    """
    embeddings = langchain_embeddings.get_embedding(embedding_model)
    index_name, _ = get_opensearch_index_settings(embedding_model, knowledge_base)
    return OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=config.opensearch_base_url,
        embedding_function=embeddings,
    )


def get_available_knowledge_bases() -> list[str]:
    """
    Returns a list of available knowledge bases by examining existing OpenSearch indices.
    Knowledge bases are ordered by creation time (oldest first), with "default" always first.
    
    :return: List of knowledge base names
    """
    try:
        opensearch_client = OpenSearch([config.opensearch_base_url])
        
        # Get indices with creation timestamps using OpenSearch client
        try:
            indices_response = opensearch_client.cat.indices(
                index="*", h="index,creation.date", format="json", timeout=10
            )
            indices_info = indices_response
        except Exception as e:
            logging.warning(f"Failed to get indices with timestamps: {e}, falling back to basic listing")
            # Fallback to basic index listing without timestamps
            indices_response = opensearch_client.cat.indices(index="*", h="index", format="json")
            indices_info = [{"index": idx["index"], "creation.date": "0"} for idx in indices_response]
        
        # Filter and parse OuRAGboros indices
        kb_timestamps = {}
        prefix = config.opensearch_index_prefix + "_"
        
        for idx_info in indices_info:
            index_name = idx_info['index']
            if index_name.startswith(prefix):
                creation_time = float(idx_info['creation.date'])
                
                # Parse index name to extract knowledge base
                remaining = index_name[len(prefix):]
                parts = remaining.split("_")
                
                # If starts with a number, it's the default format
                if parts[0].isdigit():
                    kb_name = "default"
                else:
                    # Find where the vector size starts (first numeric part after kb name)
                    kb_parts = []
                    for part in parts:
                        if part.isdigit():
                            break
                        kb_parts.append(part)
                    
                    if kb_parts:
                        kb_name = "_".join(kb_parts)
                    else:
                        continue
                
                # Use the earliest timestamp for each knowledge base (in case of multiple indices)
                if kb_name not in kb_timestamps or creation_time < kb_timestamps[kb_name]:
                    kb_timestamps[kb_name] = creation_time

        if not kb_timestamps:
            return ["default"]

        # Sort by creation time, but ensure "default" is always first
        sorted_kbs = sorted(kb_timestamps.items(), key=lambda x: x[1])  # Sort by timestamp
        kb_list = [kb for kb, _ in sorted_kbs]
        
        # Ensure "default" is always first if present
        if "default" in kb_list:
            kb_list.remove("default")
            kb_list.insert(0, "default")

        logging.debug(f"Knowledge bases ordered by creation time: {kb_list}")
        return kb_list

    except Exception as e:
        logging.warning(f"Could not retrieve knowledge bases from OpenSearch: {e}")
        return ["default"]


def delete_knowledge_base(knowledge_base: str, embedding_model: str = None) -> bool:
    """
    Deletes a knowledge base by removing ALL OpenSearch indices associated with it,
    across all embedding models.

    :param knowledge_base: The knowledge base name to delete
    :param embedding_model: [DEPRECATED] No longer used - kept for backward compatibility
    :return: True if at least one index was deleted, False otherwise
    """
    if knowledge_base == "default":
        raise ValueError("Cannot delete the default knowledge base")

    try:
        opensearch_client = OpenSearch([config.opensearch_base_url])

        # Get all indices
        all_indices = opensearch_client.cat.indices(index="*", h="index", format="json")

        deleted_count = 0
        prefix = config.opensearch_index_prefix + "_"

        # Sanitize knowledge base name to match index format
        kb_safe = knowledge_base.lower().replace(" ", "_").replace("-", "_")
        kb_safe = "".join(c for c in kb_safe if c.isalnum() or c == "_")

        # Find and delete ALL indices for this knowledge base (any embedding model)
        for idx_info in all_indices:
            index_name = idx_info['index']

            # Check if this index belongs to our knowledge base
            if index_name.startswith(prefix):
                remaining = index_name[len(prefix):]

                # For non-default KBs, index format is: {prefix}_{kb_safe}_{vector_size}_{model_hash}
                # Check if KB name appears in the index name
                if remaining.startswith(kb_safe + "_"):
                    try:
                        opensearch_client.indices.delete(index=index_name)
                        deleted_count += 1
                        logging.info(f"Deleted OpenSearch index: {index_name}")
                    except Exception as e:
                        logging.warning(f"Failed to delete index {index_name}: {e}")

        # Clear all cache entries for this knowledge base (thread-safe)
        with _opensearch_configs_lock:
            keys_to_delete = [
                key for key in _opensearch_configs.keys()
                if key.endswith(f"#{knowledge_base}")
            ]
            for key in keys_to_delete:
                del _opensearch_configs[key]

        if deleted_count > 0:
            logging.info(
                f"Successfully deleted {deleted_count} index/indices for KB '{knowledge_base}'"
            )
            return True
        else:
            logging.warning(f"No indices found for KB '{knowledge_base}'")
            return False

    except Exception as e:
        logging.error(f"Failed to delete knowledge base '{knowledge_base}': {e}")
        return False

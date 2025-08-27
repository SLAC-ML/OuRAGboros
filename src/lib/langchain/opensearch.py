import hashlib
import logging
from typing import Tuple

import opensearchpy.exceptions
from opensearchpy import OpenSearch

from langchain_community.vectorstores import OpenSearchVectorSearch

import lib.config as config


import lib.langchain.embeddings as langchain_embeddings

_opensearch_configs = {}

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
        logging.debug(f"{cache_key} not found in _opensearch_configs; generating..")
        vector_size = len(
            langchain_embeddings.get_embedding(embedding_model).embed_query("hi")
        )
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
        opensearch_client.indices.create(
            index=index_name,
            body=config.opensearch_index_settings(vector_size=vector_size),
        )
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
    
    :return: List of knowledge base names
    """
    try:
        opensearch_client = OpenSearch([config.opensearch_base_url])
        indices = opensearch_client.indices.get_alias(
            index=f"{config.opensearch_index_prefix}_*"
        )

        knowledge_bases = set()
        prefix = config.opensearch_index_prefix + "_"

        logging.debug(f"Found indices: {list(indices.keys())}")

        for index_name in indices.keys():
            if index_name.startswith(prefix):
                # Parse index name to extract knowledge base
                # Format examples:
                # - ouragboros_768_abc123... (old format = "default")
                # - ouragboros_test_kb_768_abc123... (new format = "test_kb")
                
                remaining = index_name[len(prefix):]
                parts = remaining.split("_")
                
                # If starts with a number, it's the old format (default knowledge base)
                if parts[0].isdigit():
                    knowledge_bases.add("default")
                else:
                    # Find where the vector size starts (first numeric part after kb name)
                    kb_parts = []
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            break
                        kb_parts.append(part)
                    
                    if kb_parts:
                        kb_name = "_".join(kb_parts)
                        knowledge_bases.add(kb_name)

        result = sorted(list(knowledge_bases))
        logging.debug(f"Discovered knowledge bases: {result}")
        return result

    except Exception as e:
        logging.warning(f"Could not retrieve knowledge bases from OpenSearch: {e}")
        return ["default"]
